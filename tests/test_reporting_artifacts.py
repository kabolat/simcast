import json
from pathlib import Path

import pandas as pd

from simcast.config import CompositeAnalysisConfig
from simcast.reporting import read_evaluation_artifacts, write_composite_report


def test_current_and_legacy_evaluation_manifests_are_read_without_rewriting(tmp_path: Path) -> None:
    current = tmp_path / "current" / "evaluation_manifest.json"
    legacy = tmp_path / "legacy" / "nested" / "evaluation_manifest.json"
    current.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    current_payload = {"schema": "simcast.evaluation.v2", "methods": ["conditional_kernel"]}
    legacy_payload = {"experimental_protocol": {"name": "historical"}, "methods": ["independent"]}
    current.write_text(json.dumps(current_payload), encoding="utf-8")
    legacy.write_text(json.dumps(legacy_payload), encoding="utf-8")
    before = {path: path.read_bytes() for path in (current, legacy)}

    artifacts = read_evaluation_artifacts(tmp_path)

    assert [artifact.manifest for artifact in artifacts] == [current_payload, legacy_payload]
    assert {path: path.read_bytes() for path in (current, legacy)} == before


def test_generic_composite_report_includes_m4(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    rows = [
        {"method": method, "origin": origin, "mean_pinball": 1.0 + 0.02 * origin + offset}
        for method, offset in (("independent", 0.1), ("conditional_kernel", 0.0))
        for origin in range(12)
    ]
    pd.DataFrame(rows).to_parquet(evaluation / "per_origin_metrics.parquet", index=False)
    cells: list[dict[str, object]] = [
        {
            "evaluation_path": str(evaluation),
            "base_id": "transformer",
            "method_id": "m0",
            "seed": None,
            "method_family": "independent",
            "reference_family": "independent",
        },
        {
            "evaluation_path": str(evaluation),
            "base_id": "transformer",
            "method_id": "m4",
            "seed": 42,
            "method_family": "conditional_kernel",
            "reference_family": "independent",
        }
    ]

    report = tmp_path / "report"
    analysis = CompositeAnalysisConfig(reference="m0", bootstrap_replicates=200, primary_block_length=7)
    write_composite_report(report, cells, analysis)

    summary = pd.read_csv(report / "method_summary.csv")
    effects = pd.read_csv(report / "paired_effects.csv")
    assert set(summary["method"]) == {"independent", "conditional_kernel"}
    assert set(summary["method_id"]) == {"m0", "m4"}
    assert set(effects["method"]) == {"conditional_kernel"}
    assert (report / "method_comparison_mean_pinball.png").is_file()
    assert (report / "paired_effect_mean_pinball.png").is_file()
    assert not (report / "method_comparison_mean_pinball.pdf").exists()
    assert (report / "report_summary.md").is_file()
    assert "method_summary.csv" in (report / "report_summary.md").read_text(encoding="utf-8")
