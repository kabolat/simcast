from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from simcast.config import CompositeAnalysisConfig
from simcast.evaluation.plots import plot_aggregate_fan
from simcast.reporting import write_composite_report


def test_generic_composite_report_includes_m4(tmp_path: Path) -> None:
    rows = [
        {
            "method": method,
            "origin": origin,
            "origin_index": origin,
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
    cells: list[dict[str, object]] = []
    for method_id, method, seeds in (
        ("m0", "independent", [(None, 0.0)]),
        ("m4", "conditional_kernel", [(42, 0.0), (43, 0.04)]),
    ):
        for seed, seed_offset in seeds:
            evaluation = tmp_path / "evaluation" / f"{method_id}-{seed or 'deterministic'}"
            evaluation.mkdir(parents=True)
            method_rows = [
                {**row, "mean_pinball": row["mean_pinball"] + seed_offset}
                for row in rows
                if row["method"] == method
            ]
            pd.DataFrame(method_rows).to_parquet(evaluation / "per_origin_metrics.parquet", index=False)
            pd.DataFrame(method_rows).assign(
                lead=1,
                valid=True,
                cross_entity_statistic="sum",
                crps=0.5,
                **{
                    "interval_score_0.9": 0.0,
                    "interval_width_0.9": 1.0,
                    "interval_lower_0.9": 0.5,
                    "interval_upper_0.9": 1.5,
                },
            ).to_parquet(evaluation / "per_origin_lead_metrics.parquet", index=False)
            cells.append(
                {
                    "evaluation_path": str(evaluation),
                    "base_id": "transformer",
                    "method_id": method_id,
                    "seed": seed,
                    "method_family": method,
                }
            )

    report = tmp_path / "report"
    analysis = CompositeAnalysisConfig(bootstrap_replicates=200, primary_block_length=7)
    write_composite_report(report, cells, analysis, reference="m0")

    summary = pd.read_csv(report / "method_summary.csv")
    effects = pd.read_csv(report / "paired_effect_bs.csv")
    assert set(summary["method"]) == {"independent", "conditional_kernel"}
    assert set(summary["method_id"]) == {"m0", "m4"}
    assert set(effects["method"]) == {"conditional_kernel"}
    assert (report / "method_comparison_mean_pinball.png").is_file()
    assert (report / "paired_effect_bs_mean_pinball.png").is_file()
    assert (report / "paired_effect_dist_mean_pinball.png").is_file()
    differences = pd.read_parquet(report / "paired_effect_dist.parquet")
    assert len(differences) == 12
    assert set(differences["method_id"]) == {"m4"}
    assert np.allclose(differences["score_improvement"], 0.08)
    assert differences["method_better"].all()
    assert (report / "summary_coverage.png").is_file()
    assert (report / "summary_quantile_calibration.png").is_file()
    assert (report / "aggregate_fan_gallery_transformer.png").is_file()
    report_cases = pd.read_parquet(report / "per_origin_lead_metrics.parquet")
    assert {"origin_index", "lead", "valid", "interval_lower_0.9", "interval_upper_0.9"} <= set(
        report_cases.columns
    )
    assert not (report / "method_comparison_mean_pinball.pdf").exists()
    assert (report / "report_summary.md").is_file()
    assert "method_summary.csv" in (report / "report_summary.md").read_text(encoding="utf-8")


def test_calibration_figures_facet_multiple_bases(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    rows = [
        {
            "method": method,
            "origin": origin,
            "origin_index": origin,
            "mean_pinball": 1.0 + offset,
            "coverage_0.9": 0.9,
            "observed_aggregate": 1.0,
            "aggregate_q0.05": 0.8 + offset,
            "aggregate_q0.50": 1.0 + offset,
            "aggregate_q0.95": 1.2 + offset,
        }
        for method, offset in (("independent", 0.1), ("conditional_kernel", 0.0))
        for origin in range(4)
    ]
    pd.DataFrame(rows).to_parquet(evaluation / "per_origin_metrics.parquet", index=False)
    pd.DataFrame(rows).assign(
        lead=1,
        valid=True,
        cross_entity_statistic="sum",
        **{
            "interval_score_0.9": 0.0,
            "interval_width_0.9": 1.0,
            "interval_lower_0.9": 0.5,
            "interval_upper_0.9": 1.5,
        },
    ).to_parquet(
        evaluation / "per_origin_lead_metrics.parquet", index=False
    )
    cells: list[dict[str, object]] = [
        {
            "evaluation_path": str(evaluation),
            "base_id": base_id,
            "method_id": method_id,
            "seed": None if method_id == "m0" else 42,
            "method_family": method,
        }
        for base_id in ("transformer", "solar_park")
        for method_id, method in (("m0", "independent"), ("m4", "conditional_kernel"))
    ]

    report = tmp_path / "report"
    write_composite_report(
        report,
        cells,
        CompositeAnalysisConfig(bootstrap_replicates=10, primary_block_length=3),
        reference="m0",
    )

    assert mpimg.imread(report / "summary_coverage.png").shape[0] == 1620
    assert mpimg.imread(report / "summary_quantile_calibration.png").shape[0] == 1620
    assert (report / "aggregate_fan_gallery_transformer.png").is_file()
    assert (report / "aggregate_fan_gallery_solar_park.png").is_file()


def test_fan_and_score_plots_include_crps_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "method": method,
            "origin": origin,
            "origin_index": origin,
            "mean_pinball": 1.0,
            "crps": 2.1 if method == "independent" else 2.0,
            "observed_aggregate": 1.0,
            "aggregate_q0.5": 1.0,
        }
        for method in ("independent", "conditional_kernel")
        for origin in range(3)
    ]
    cells = []
    for method_id, method in (("m0", "independent"), ("m4", "conditional_kernel")):
        evaluation = tmp_path / "evaluation" / method_id
        evaluation.mkdir(parents=True)
        method_rows = [row for row in rows if row["method"] == method]
        pd.DataFrame(method_rows).to_parquet(evaluation / "per_origin_metrics.parquet", index=False)
        pd.DataFrame(method_rows).assign(
            lead=1,
            valid=True,
            cross_entity_statistic="absolute_max",
            **{"interval_lower_0.8": 0.2, "interval_upper_0.8": 1.8},
        ).to_parquet(evaluation / "per_origin_lead_metrics.parquet", index=False)
        cells.append(
            {
                "evaluation_path": str(evaluation),
                "base_id": "transformer",
                "method_id": method_id,
                "seed": None,
                "method_family": method,
            }
        )
    captured: dict[str, str] = {}
    captured_legends: dict[str, list[str]] = {}
    captured_annotations: dict[str, list[tuple[str, tuple[float, float], str]]] = {}
    captured_title_gaps: dict[str, bool] = {}

    def capture_figure(figure, path, **_kwargs):
        figure.canvas.draw()
        filename = Path(path).name
        text = [
            item.get_text()
            for axis in figure.axes
            for item in [*axis.texts, axis.title, axis.xaxis.label, axis.yaxis.label]
        ]
        text.extend(item.get_text() for item in figure.texts)
        text.extend(label.get_text() for axis in figure.axes for label in axis.get_xticklabels())
        captured[filename] = "\n".join(text)
        captured_legends[filename] = [
            label.get_text()
            for axis in figure.axes
            if axis.get_legend() is not None
            for label in axis.get_legend().get_texts()
        ]
        captured_annotations[filename] = [
            (item.get_text(), item.get_position(), item.get_ha())
            for axis in figure.axes
            for item in axis.texts
        ]
        suptitle = next(
            (item for item in figure.texts if item.get_text().startswith("Paired relative")),
            None,
        )
        if suptitle is not None:
            captured_title_gaps[filename] = (
                figure.axes[0].title.get_window_extent().y1 < suptitle.get_window_extent().y0
            )

    monkeypatch.setattr(plt.Figure, "savefig", capture_figure)

    report = tmp_path / "report"
    write_composite_report(
        report,
        cells,
        CompositeAnalysisConfig(bootstrap_replicates=10, primary_block_length=3),
        reference="m0",
        metrics=("mean_pinball", "crps"),
    )
    plot_aggregate_fan(
        np.array([[0.8, 1.0, 1.2], [0.9, 1.1, 1.3]]),
        (0.1, 0.5, 0.9),
        np.array([1.0, 1.1]),
        tmp_path / "aggregate_fan.png",
        interval_predictions=np.array([[[0.5, 1.5]], [[0.6, 1.6]]]),
        interval_levels=(0.8,),
        crps=1.25,
        title="Aggregate forecast",
    )

    assert "CRPS" in captured["aggregate_fan_gallery_transformer.png"]
    assert "Maximum absolute entity value" in captured["aggregate_fan_gallery_transformer.png"]
    assert not any("interval" in label.lower() for label in captured_legends["aggregate_fan_gallery_transformer.png"])
    assert "Mean CRPS 1.25" in captured["aggregate_fan.png"]
    assert "Mean CRPS across valid leads" not in captured["aggregate_fan.png"]
    assert not any("interval" in label.lower() for label in captured_legends["aggregate_fan.png"])
    assert "2.1" in captured["method_comparison_crps.png"]
    assert "+4.8%" in captured["paired_effect_bs_crps.png"]
    improvement_annotation = next(
        item for item in captured_annotations["paired_effect_bs_crps.png"] if item[0] == "+4.8%"
    )
    assert improvement_annotation[1:] == ((5, 0), "left")
    assert captured_title_gaps["paired_effect_bs_crps.png"]
    assert "100% origins better" in captured["paired_effect_dist_crps.png"]
    assert not captured_legends["paired_effect_dist_crps.png"]
