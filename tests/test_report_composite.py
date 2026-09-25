import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from simcast.cli.report_composite import report_composite


def _synthetic_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "runs" / "lab" / "quick_all_methods" / "fixed"
    evaluation_root = run_root / "evaluations" / "standard" / "transformer"
    reference_evaluation = evaluation_root / "m0" / "deterministic"
    method_evaluation = evaluation_root / "m4" / "seed_1"
    reference_evaluation.mkdir(parents=True)
    method_evaluation.mkdir(parents=True)
    rows = [
        {
            "method": method,
            "origin": origin,
            "mean_pinball": 1.0 + offset,
            "crps": 2.0 + offset,
            "coverage_0.9": 0.88 if method == "independent" else 0.92,
            "observed_aggregate": 1.0,
            "aggregate_q0.05": 0.8 + offset,
            "aggregate_q0.50": 1.0 + offset,
            "aggregate_q0.95": 1.2 + offset,
        }
        for method, offset in (("independent", 0.1), ("conditional_kernel", 0.0))
        for origin in range(8)
    ]
    frame = pd.DataFrame(rows)
    frame[frame["method"] == "independent"].to_parquet(
        reference_evaluation / "per_origin_metrics.parquet", index=False
    )
    frame[frame["method"] == "conditional_kernel"].to_parquet(
        method_evaluation / "per_origin_metrics.parquet", index=False
    )
    manifest = {
        "fits": [],
        "evaluations": {
            "standard": [
                {
                    "base_id": "transformer",
                    "method_id": "m0",
                    "seed": None,
                    "method_family": "independent",
                    "evaluation_path": str(reference_evaluation),
                    "status": "complete",
                },
                {
                    "base_id": "transformer",
                    "method_id": "m4",
                    "seed": 1,
                    "method_family": "conditional_kernel",
                    "evaluation_path": str(method_evaluation),
                    "status": "complete",
                },
                {
                    "base_id": "transformer",
                    "method_id": "m4",
                    "seed": 2,
                    "method_family": "conditional_kernel",
                    "evaluation_path": str(method_evaluation),
                    "status": "running",
                },
            ]
        },
    }
    (run_root / "composite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_root


def _write_report_config(path: Path, run_root: Path, *, metrics: list[str], output_dir: Path | None = None) -> Path:
    payload: dict[str, object] = {
        "kind": "report",
        "reference": "m0",
        "metrics": metrics,
        "evaluation_ids": ["standard"],
        "analysis": {"bootstrap_replicates": 200, "primary_block_length": 3},
    }
    if output_dir is not None:
        payload["output_dir"] = str(output_dir)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_report_composite_only_reads_completed_cells_and_defaults_output_under_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = _synthetic_run_root(tmp_path)
    config_path = _write_report_config(tmp_path / "report.yaml", run_root, metrics=["mean_pinball"])

    destination = report_composite(config_path, run_root=run_root)

    assert destination == run_root / "reports" / "report" / "standard"
    report_destination = destination
    assert (report_destination / "method_comparison_mean_pinball.png").is_file()
    assert (report_destination / "paired_effect_mean_pinball.png").is_file()
    assert (report_destination / "report_summary.md").is_file()
    assert (report_destination / "summary_coverage.png").is_file()
    assert (report_destination / "summary_quantile_calibration.png").is_file()
    per_origin = pd.read_parquet(report_destination / "per_origin_metrics.parquet")
    assert len(per_origin) == 16  # only the one completed cell's rows, not the "running" one


def test_report_composite_supports_alternate_metrics_and_explicit_output_dir(tmp_path: Path) -> None:
    run_root = _synthetic_run_root(tmp_path)
    output_dir = tmp_path / "custom_report"
    config_path = _write_report_config(
        tmp_path / "report.yaml", run_root, metrics=["mean_pinball", "crps"], output_dir=output_dir
    )

    destination = report_composite(config_path, run_root=run_root)

    report_destination = output_dir / "standard"
    assert destination == report_destination
    assert (report_destination / "method_comparison_mean_pinball.png").is_file()
    assert (report_destination / "method_comparison_crps.png").is_file()
    assert (report_destination / "paired_effect_crps.png").is_file()
    summary = pd.read_csv(report_destination / "method_summary.csv")
    assert set(summary["metric"]) == {"mean_pinball", "crps"}


def test_report_composite_rejects_unknown_metric_without_writing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = _synthetic_run_root(tmp_path)
    config_path = _write_report_config(tmp_path / "report.yaml", run_root, metrics=["not_a_real_metric"])

    with pytest.raises(ValueError, match="not_a_real_metric"):
        report_composite(config_path, run_root=run_root)

    assert not (run_root / "reports").exists()


def test_report_composite_plots_noncovering_bootstrap_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = _synthetic_run_root(tmp_path)
    config_path = _write_report_config(tmp_path / "report.yaml", run_root, metrics=["mean_pinball"])
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("bootstrap_replicates: 200", "bootstrap_replicates: 1"),
        encoding="utf-8",
    )
    report_composite(config_path, run_root=run_root)

    assert (run_root / "reports" / "report" / "standard").is_dir()


def test_force_replaces_existing_report_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = _synthetic_run_root(tmp_path)
    config_path = _write_report_config(tmp_path / "report.yaml", run_root, metrics=["mean_pinball"])

    destination = report_composite(config_path, run_root=run_root)
    stale = destination / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    report_composite(config_path, run_root=run_root, force=True)

    assert not stale.exists()


def test_composite_mode_selects_the_only_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = _synthetic_run_root(tmp_path)
    venue_dir = tmp_path / "configs" / "venues" / "lab"
    report_dir = tmp_path / "configs" / "reports"
    venue_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    (report_dir / "main.yaml").write_text(
        yaml.safe_dump(
            {
                "kind": "report",
                "reference": "m0",
                "metrics": ["mean_pinball"],
                "evaluation_ids": ["standard"],
                "analysis": {"bootstrap_replicates": 10, "primary_block_length": 3},
            }
        ),
        encoding="utf-8",
    )
    composite_path = venue_dir / "composite.yaml"
    composite_path.write_text(
        "kind: composite\nname: quick_all_methods\nvenue: lab\n"
        "bases: [{id: transformer, config: transformer.yaml}]\n"
        "methods: [{id: m0, method: m0.yaml}, {id: m4, method: m4.yaml}]\n"
        "evaluations: [{id: standard, config: standard.yaml}]\n"
        "reports: [{id: main, config: ../../reports/main.yaml, evaluation_ids: [standard]}]\n",
        encoding="utf-8",
    )
    (venue_dir / "standard.yaml").write_text(
        yaml.safe_dump({"kind": "evaluation", "id": "standard"}), encoding="utf-8"
    )

    destination = report_composite(composite_config_path=composite_path)

    assert destination == run_root / "reports" / "main" / "standard"


def test_composite_mode_runs_all_reports_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = _synthetic_run_root(tmp_path)
    venue_dir = tmp_path / "configs" / "venues" / "lab"
    report_dir = tmp_path / "configs" / "reports"
    venue_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    report_payload = {
        "kind": "report",
        "reference": "m0",
        "metrics": ["mean_pinball"],
        "evaluation_ids": ["standard"],
        "analysis": {"bootstrap_replicates": 10, "primary_block_length": 3},
    }
    for name in ("first", "second"):
        (report_dir / f"{name}.yaml").write_text(yaml.safe_dump(report_payload), encoding="utf-8")
    composite_path = venue_dir / "composite.yaml"
    composite_path.write_text(
        "kind: composite\nname: quick_all_methods\nvenue: lab\n"
        "bases: [{id: transformer, config: transformer.yaml}]\n"
        "methods: [{id: m0, method: m0.yaml}, {id: m4, method: m4.yaml}]\n"
        "evaluations: [{id: standard, config: standard.yaml}]\n"
        "reports: [{id: first, config: ../../reports/first.yaml, evaluation_ids: [standard]}, "
        "{id: second, config: ../../reports/second.yaml, evaluation_ids: [standard]}]\n",
        encoding="utf-8",
    )
    (venue_dir / "standard.yaml").write_text(
        yaml.safe_dump({"kind": "evaluation", "id": "standard"}), encoding="utf-8"
    )

    destination = report_composite(composite_config_path=composite_path)

    assert destination == run_root / "reports"
    assert (destination / "first" / "standard" / "report_summary.md").is_file()
    assert (destination / "second" / "standard" / "report_summary.md").is_file()

