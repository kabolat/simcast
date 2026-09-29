import json
from pathlib import Path

import pytest

import simcast.cli.run_composite as composite_module
import simcast.cli.run_singular as singular_module
from simcast.cli.run_composite import expand_methods, run_composite
from simcast.config import load_composite_config

CONFIGS = Path(__file__).parents[1] / "configs"


def test_singular_fits_only_selected_method_and_explicit_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    fitted: list[str] = []
    evaluated: list[str] = []

    def fake_train(config, *, cache_dir, output_dir):
        assert cache_dir == cache
        path = Path(output_dir)
        path.mkdir(parents=True)
        fitted.append(config.dependence.method)
        return path

    def fake_evaluate(
        config, *, methods, metrics=None, figures=None, base_figures_dir=None, method_runs, cache_dir, output_dir
    ):
        del config, method_runs
        assert cache_dir == cache
        evaluated.extend(methods)
        path = Path(output_dir)
        path.mkdir(parents=True)
        return path

    monkeypatch.setattr(singular_module, "locate_compatible_cache", lambda _base: cache)
    monkeypatch.setattr(singular_module, "train_from_config", fake_train)
    monkeypatch.setattr(singular_module, "evaluate_from_config", fake_evaluate)
    monkeypatch.setattr(
        singular_module,
        "build_cache_from_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must reuse cache")),
    )
    destination = tmp_path / "singular"
    result = singular_module.run_singular(
        CONFIGS / "bases/liander2024/transformer.yaml",
        CONFIGS / "methods/m4_conditional_kernel.yaml",
        reference_method_path=CONFIGS / "methods/m0_independent.yaml",
        output_dir=destination,
    )

    assert result == destination.resolve()
    assert fitted == ["conditional_kernel", "independent"]
    assert evaluated == ["conditional_kernel", "independent"]
    manifest = json.loads((destination / "singular_manifest.json").read_text(encoding="utf-8"))
    assert manifest["primary_method"] == "m4"
    assert (destination / "resolved_base.yaml").is_file()
    assert (destination / "resolved_method.yaml").is_file()


def test_composite_expansion_is_exact_and_includes_full_m4() -> None:
    principal_venue = next(path for path in (CONFIGS / "venues").iterdir() if path.name != "lab")
    path = principal_venue / "main.yaml"
    composite = load_composite_config(path)
    cells = expand_methods(path, composite)
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.method.id] = counts.get(cell.method.id, 0) + 1
    assert counts == {"m0": 5, "m1": 5, "m2": 50, "m3": 50, "m4": 50}
    m4 = [cell for cell in cells if cell.method.id == "m4"]
    assert all(cell.method.optimization.epochs == 100 for cell in m4)  # type: ignore[union-attr]


def test_lab_budget_is_an_equal_composite_override() -> None:
    path = CONFIGS / "venues/lab/quick_all_methods.yaml"
    composite = load_composite_config(path)
    cells = expand_methods(path, composite)
    optimized = [cell for cell in cells if cell.method.id in {"m2", "m3", "m4"}]
    assert [cell.method.optimization.epochs for cell in optimized] == [2, 2, 2]  # type: ignore[union-attr]
    assert [cell.method.optimization.patience for cell in optimized] == [1, 1, 1]  # type: ignore[union-attr]


def test_resume_rejects_changed_composite_before_any_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "runs/lab/quick_all_methods/fixed/composite_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"configuration_sha256": "different", "fits": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="resolved composite configuration has changed"):
        run_composite(CONFIGS / "venues/lab/quick_all_methods.yaml", run_id="fixed", resume=True)


def test_resume_preserves_validated_completed_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    calls = {"fit": 0, "evaluate": 0}

    def fake_train(config, *, cache_dir, output_dir):
        assert cache_dir == cache
        calls["fit"] += 1
        path = Path(output_dir)
        path.mkdir(parents=True)
        model_name = (
            "best.pt"
            if config.dependence.method
            in {"conditional_low_rank", "set_aware_low_rank", "conditional_kernel"}
            else "model.npz"
        )
        (path / model_name).write_bytes(b"model")
        (path / "run_metadata.json").write_text("{}\n", encoding="utf-8")
        return path

    def fake_evaluate(
        config, *, methods, metrics, figures=None, base_figures_dir=None, method_runs, cache_dir, output_dir
    ):
        del config, methods, metrics, method_runs
        assert cache_dir == cache
        calls["evaluate"] += 1
        path = Path(output_dir)
        path.mkdir(parents=True)
        (path / "evaluation_manifest.json").write_text(
            json.dumps({"metrics": ["mean_pinball"]}) + "\n", encoding="utf-8"
        )
        return path

    monkeypatch.setattr(composite_module, "locate_compatible_cache", lambda _base: cache)
    monkeypatch.setattr(composite_module, "train_from_config", fake_train)
    monkeypatch.setattr(composite_module, "evaluate_from_config", fake_evaluate)
    path = CONFIGS / "venues/lab/quick_all_methods.yaml"

    composite_module.run_composite(path, run_id="resume_test")
    first_counts = dict(calls)
    composite_module.run_composite(path, run_id="resume_test", resume=True)

    assert calls == first_counts


def test_composite_runs_declared_reports_after_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    config_dir = tmp_path / "configs" / "venues" / "lab"
    config_dir.mkdir(parents=True)
    base = (CONFIGS / "bases/liander2024/transformer.yaml").resolve()
    method = (CONFIGS / "methods/m0_independent.yaml").resolve()
    evaluation = (CONFIGS / "evaluations/standard.yaml").resolve()
    report = (CONFIGS / "reports/lab_main.yaml").resolve()
    composite_path = config_dir / "with_report.yaml"
    composite_path.write_text(
        "\n".join(
            (
                "kind: composite",
                "name: with_report",
                "venue: lab",
                f"bases: [{'{'}id: transformer, config: {base}{'}'}]",
                f"methods: [{'{'}id: m0, method: {method}{'}'}]",
                f"evaluations: [{'{'}id: standard, config: {evaluation}, method_ids: [m0]{'}'}]",
                f"reports: [{'{'}id: lab_main, config: {report}, evaluation_ids: [standard]{'}'}]",
                "",
            )
        ),
        encoding="utf-8",
    )
    events: list[str] = []

    def fake_train(config, *, cache_dir, output_dir):
        del config
        assert cache_dir == cache
        path = Path(output_dir)
        path.mkdir(parents=True)
        (path / "model.npz").write_bytes(b"model")
        (path / "run_metadata.json").write_text("{}\n", encoding="utf-8")
        events.append("fit")
        return path

    def fake_evaluate(config, *, methods, metrics, figures, base_figures_dir, method_runs, cache_dir, output_dir):
        del config, methods, metrics, figures, base_figures_dir, method_runs
        assert cache_dir == cache
        path = Path(output_dir)
        path.mkdir(parents=True)
        (path / "evaluation_manifest.json").write_text('{"metrics": ["mean_pinball"]}\n', encoding="utf-8")
        events.append("evaluate")
        return path

    def fake_report(*, composite_config_path, run_id, force, report_id=None):
        assert composite_config_path == composite_path.resolve()
        assert report_id is None
        assert run_id == "with_report_test"
        assert force is True
        manifest = json.loads(
            (tmp_path / "runs/lab/with_report/with_report_test/composite_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["evaluations"]["standard"][0]["status"] == "complete"
        events.append("report")

    monkeypatch.setattr(composite_module, "locate_compatible_cache", lambda _base: cache)
    monkeypatch.setattr(composite_module, "train_from_config", fake_train)
    monkeypatch.setattr(composite_module, "evaluate_from_config", fake_evaluate)
    monkeypatch.setattr(composite_module, "report_composite", fake_report)

    run_composite(composite_path, run_id="with_report_test")

    assert events == ["fit", "evaluate", "report"]


@pytest.mark.parametrize("run_id", ["../escape", "UPPER", "spaces are unsafe"])
def test_run_identifier_cannot_escape_venue_paths(run_id: str) -> None:
    with pytest.raises(ValueError, match="safe slug"):
        run_composite(CONFIGS / "venues/lab/quick_all_methods.yaml", run_id=run_id)
