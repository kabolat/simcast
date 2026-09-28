from pathlib import Path

import pandas as pd

from simcast.config import CompositeAnalysisConfig
from simcast.reporting import write_composite_report


def test_generic_composite_report_includes_m4(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    rows = [
        {
            "method": method,
            "origin": origin,
            "mean_pinball": 1.0 + 0.02 * origin + offset,
            "coverage_0.9": 0.88 if method == "independent" else 0.92,
            "observed_aggregate": 1.0,
            "aggregate_q0.05": 0.8 + offset,
            "aggregate_q0.50": 1.0 + offset,
            "aggregate_q0.95": 1.2 + offset,
        }
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
        },
        {
            "evaluation_path": str(evaluation),
            "base_id": "transformer",
            "method_id": "m4",
            "seed": 42,
            "method_family": "conditional_kernel",
        }
    ]

    report = tmp_path / "report"
    analysis = CompositeAnalysisConfig(bootstrap_replicates=200, primary_block_length=7)
    write_composite_report(report, cells, analysis, reference="m0")

    summary = pd.read_csv(report / "method_summary.csv")
    effects = pd.read_csv(report / "paired_effects.csv")
    assert set(summary["method"]) == {"independent", "conditional_kernel"}
    assert set(summary["method_id"]) == {"m0", "m4"}
    assert set(effects["method"]) == {"conditional_kernel"}
    assert (report / "method_comparison_mean_pinball.png").is_file()
    assert (report / "paired_effect_mean_pinball.png").is_file()
    assert (report / "summary_coverage.png").is_file()
    assert (report / "summary_quantile_calibration.png").is_file()
    assert not (report / "method_comparison_mean_pinball.pdf").exists()
    assert (report / "report_summary.md").is_file()
    assert "method_summary.csv" in (report / "report_summary.md").read_text(encoding="utf-8")
