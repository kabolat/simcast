import json
from pathlib import Path

import pytest
import yaml

import simcast.cli.evaluate_composite as evaluate_composite_module
from simcast.cli.evaluate_composite import evaluate_composite
from simcast.config import load_base_config, load_method_config, resolve_run_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
BASES = CONFIGS / "bases" / "liander2024"
METHODS = CONFIGS / "methods"


def _write_fit(fit_dir: Path, *, base_file: str, method_file: str, seed: int | None) -> None:
    fit_dir.mkdir(parents=True)
    base = load_base_config(BASES / base_file)
    method = load_method_config(METHODS / method_file)
    resolved = resolve_run_config(base, method, seed=seed)
    (fit_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )


def _synthetic_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "runs" / "lab" / "quick_shot" / "fixed"
    cache = tmp_path / "cache"
    cache.mkdir()
    m0_dir = run_root / "models" / "transformer" / "m0" / "deterministic"
    m4_dir = run_root / "models" / "transformer" / "m4" / "seed_1"
    _write_fit(m0_dir, base_file="transformer.yaml", method_file="m0_independent.yaml", seed=None)
    _write_fit(m4_dir, base_file="transformer.yaml", method_file="m4_conditional_kernel.yaml", seed=1)
    manifest = {
        "fits": [
            {
                "fit_id": "transformer/m0/deterministic",
                "base_id": "transformer",
                "method_id": "m0",
                "seed": None,
                "method_family": "independent",
                "cache_path": str(cache),
                "fit_path": str(m0_dir),
                "status": "complete",
            },
            {
                "fit_id": "transformer/m4/seed_1",
                "base_id": "transformer",
                "method_id": "m4",
                "seed": 1,
                "method_family": "conditional_kernel",
                "cache_path": str(cache),
                "fit_path": str(m4_dir),
                "status": "complete",
            },
        ],
        "evaluations": {},
    }
    (run_root / "composite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_root


def _patch_evaluate(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"evaluate": 0}

    def fake_evaluate(config, *, methods, method_runs, cache_dir, output_dir):
        del config, methods, method_runs, cache_dir
        calls["evaluate"] += 1
        path = Path(output_dir)
        path.mkdir(parents=True)
        (path / "evaluation_manifest.json").write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(evaluate_composite_module, "evaluate_from_config", fake_evaluate)
    return calls


def test_standalone_evaluation_document_evaluates_all_selected_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = _synthetic_run_root(tmp_path)
    calls = _patch_evaluate(monkeypatch)
    document_path = tmp_path / "variogram_power_1.yaml"
    document_path.write_text(
        yaml.safe_dump(
            {
                "kind": "evaluation",
                "id": "variogram_power_1",
                "evaluation": {"variogram_power": 1.0},
                "run_root": str(run_root),
            }
        ),
        encoding="utf-8",
    )

    destination = evaluate_composite(document_path)

    assert destination == run_root / "evaluations" / "variogram_power_1"
    assert calls["evaluate"] == 2
    manifest = json.loads((run_root / "composite_manifest.json").read_text(encoding="utf-8"))
    cells = manifest["evaluations"]["variogram_power_1"]
    assert [cell["method_id"] for cell in cells] == ["m0", "m4"]


def test_standalone_evaluation_is_idempotent_on_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = _synthetic_run_root(tmp_path)
    calls = _patch_evaluate(monkeypatch)
    document_path = tmp_path / "variogram_power_1.yaml"
    document_path.write_text(
        yaml.safe_dump(
            {
                "kind": "evaluation",
                "id": "variogram_power_1",
                "run_root": str(run_root),
            }
        ),
        encoding="utf-8",
    )

    evaluate_composite(document_path)
    first_calls = dict(calls)
    evaluate_composite(document_path)

    assert calls == first_calls


def test_composite_mode_requires_evaluation_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = _synthetic_run_root(tmp_path)
    _patch_evaluate(monkeypatch)
    venue_dir = tmp_path / "configs" / "venues" / "lab"
    venue_dir.mkdir(parents=True)
    composite_path = venue_dir / "composite.yaml"
    composite_path.write_text(
        "kind: composite\nname: quick_shot\nvenue: lab\n"
        "bases: [{id: transformer, config: transformer.yaml}]\n"
        "methods: [{id: m0, method: m0.yaml}, {id: m4, method: m4.yaml}]\n"
        "evaluations: [{id: variogram_power_1, config: variogram_power_1.yaml, method_ids: [m4]}]\n",
        encoding="utf-8",
    )
    (venue_dir / "variogram_power_1.yaml").write_text(
        yaml.safe_dump({"kind": "evaluation", "id": "variogram_power_1"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="--evaluation is required"):
        evaluate_composite(composite_path)

    destination = evaluate_composite(composite_path, evaluation_id="variogram_power_1", run_root=run_root)
    assert destination == run_root / "evaluations" / "variogram_power_1"


def test_evaluation_does_not_require_a_reference_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = _synthetic_run_root(tmp_path)
    calls = _patch_evaluate(monkeypatch)
    document_path = tmp_path / "independent.yaml"
    document_path.write_text(
        yaml.safe_dump(
            {
                "kind": "evaluation",
                "id": "independent",
                "method_ids": ["m4"],
                "run_root": str(run_root),
            }
        ),
        encoding="utf-8",
    )

    evaluate_composite(document_path)
    assert calls["evaluate"] == 1
