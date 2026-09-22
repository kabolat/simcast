import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from simcast.cli.report_composite import report_composite


def _synthetic_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "runs" / "lab" / "quick_all_methods" / "fixed"
    evaluation = run_root / "transformer" / "m4" / "seed_1" / "evaluation"
    evaluation.mkdir(parents=True)
    rows = [
        {"method": method, "origin": origin, "mean_pinball": 1.0 + offset, "crps": 2.0 + offset}
        for method, offset in (("independent", 0.1), ("conditional_kernel", 0.0))
        for origin in range(8)
    ]
    pd.DataFrame(rows).to_parquet(evaluation / "per_origin_metrics.parquet", index=False)
    manifest = {
        "cells": [
            {
                "cell_id": "transformer/m4/seed_1",
                "base_id": "transformer",
                "experiment_id": "m4",
                "seed": 1,
                "method_family": "conditional_kernel",
                "reference_family": "independent",
                "evaluation_path": str(evaluation),
                "status": "complete",
            },
            {
                "cell_id": "transformer/m4/seed_2",
                "base_id": "transformer",
                "experiment_id": "m4",
                "seed": 2,
                "method_family": "conditional_kernel",
                "reference_family": "independent",
                "evaluation_path": str(evaluation),
                "status": "running",
            },
        ]
    }
    (run_root / "composite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_root


def _write_report_config(path: Path, run_root: Path, *, metrics: list[str], output_dir: Path | None = None) -> Path:
    payload: dict[str, object] = {
        "kind": "report",
        "run_root": str(run_root),
        "metrics": metrics,
        "analysis": {"reference": "independent", "bootstrap_replicates": 200, "primary_block_length": 3},
    }
    if output_dir is not None:
        payload["output_dir"] = str(output_dir)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_report_composite_only_reads_completed_cells_and_defaults_output_under_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = _synthetic_run_root(tmp_path)
    config_path = _write_report_config(tmp_path / "report.yaml", run_root, metrics=["mean_pinball"])

    destination = report_composite(config_path)

    assert destination == tmp_path / "reports" / "lab" / "quick_all_methods" / "fixed" / "report"
    assert (destination / "method_comparison_mean_pinball.png").is_file()
    assert (destination / "paired_effect_mean_pinball.png").is_file()
    assert (destination / "report_summary.md").is_file()
    per_origin = pd.read_parquet(destination / "per_origin_metrics.parquet")
    assert len(per_origin) == 16  # only the one completed cell's rows, not the "running" one


def test_report_composite_supports_alternate_metrics_and_explicit_output_dir(tmp_path: Path) -> None:
    run_root = _synthetic_run_root(tmp_path)
    output_dir = tmp_path / "custom_report"
    config_path = _write_report_config(
        tmp_path / "report.yaml", run_root, metrics=["mean_pinball", "crps"], output_dir=output_dir
    )

    destination = report_composite(config_path)

    assert destination == output_dir
    assert (destination / "method_comparison_mean_pinball.png").is_file()
    assert (destination / "method_comparison_crps.png").is_file()
    assert (destination / "paired_effect_crps.png").is_file()
    summary = pd.read_csv(destination / "method_summary.csv")
    assert set(summary["metric"]) == {"mean_pinball", "crps"}


def test_report_composite_rejects_unknown_metric_without_writing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = _synthetic_run_root(tmp_path)
    config_path = _write_report_config(tmp_path / "report.yaml", run_root, metrics=["not_a_real_metric"])

    with pytest.raises(ValueError, match="not_a_real_metric"):
        report_composite(config_path)

    assert not (tmp_path / "reports").exists()
