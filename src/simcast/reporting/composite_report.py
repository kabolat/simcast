"""Consolidate a completed composite run's per-cell records into a report."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from simcast.config import CompositeAnalysisConfig
from simcast.evaluation.uncertainty import paired_moving_block_bootstrap

_FIXED_FILE_DESCRIPTIONS = {
    "per_origin_metrics.parquet": "concatenated per-origin, per-method evaluation records; input to every file below",
    "method_summary.csv": "mean value per (base, experiment, method, seed), one row per metric",
    "paired_effects.csv": "moving-block bootstrap paired contrasts vs. the reference, one row per metric/comparison",
    "report_summary.md": "this file",
}


def write_composite_report(
    report_dir: Path,
    cells: list[dict[str, object]],
    analysis: CompositeAnalysisConfig,
    metrics: Sequence[str] = ("mean_pinball",),
) -> None:
    """Write pooled per-origin metrics, paired effects, and per-metric figures."""

    origin_frames: list[pd.DataFrame] = []
    primary_frames: list[pd.DataFrame] = []
    comparisons: set[tuple[str, str, str, str]] = set()
    for cell in cells:
        evaluation = Path(str(cell["evaluation_path"]))
        frame = pd.read_parquet(evaluation / "per_origin_metrics.parquet")
        frame["base_id"] = str(cell["base_id"])
        frame["experiment_id"] = str(cell["experiment_id"])
        frame["configured_seed"] = "deterministic" if cell["seed"] is None else str(cell["seed"])
        origin_frames.append(frame)
        primary_family = str(cell["method_family"])
        primary_frames.append(frame[frame["method"] == primary_family])
        reference_family = str(cell["reference_family"])
        if primary_family == reference_family:
            continue
        comparisons.add(
            (str(cell["base_id"]), str(cell["experiment_id"]), primary_family, reference_family)
        )
    per_origin = pd.concat(origin_frames, ignore_index=True)
    unknown_metrics = [metric for metric in metrics if metric not in per_origin.columns]
    if unknown_metrics:
        raise ValueError(f"metrics {unknown_metrics} are not columns of per_origin_metrics.parquet")

    report_dir.mkdir(parents=True, exist_ok=True)
    per_origin.to_parquet(report_dir / "per_origin_metrics.parquet", index=False)
    primary_origin = pd.concat(primary_frames, ignore_index=True)

    summary_frames = []
    effect_rows: list[dict[str, object]] = []
    for metric in metrics:
        frame = primary_origin.groupby(
            ["base_id", "experiment_id", "method", "configured_seed"], dropna=False, as_index=False
        ).agg(value=(metric, "mean"))
        frame.insert(0, "metric", metric)
        summary_frames.append(frame)

        for base_id, experiment_id, primary_family, reference_family in sorted(comparisons):
            comparison = per_origin[
                (per_origin["base_id"] == base_id) & (per_origin["experiment_id"] == experiment_id)
            ]
            # Averaging here treats optimization seeds as repeated fits, not as
            # additional test observations. Repeated deterministic reference rows
            # collapse to their common value in the same operation.
            averaged = comparison.groupby(["method", "origin"], as_index=False)[metric].mean()
            wide = averaged.pivot(index="origin", columns="method", values=metric).dropna()
            if primary_family not in wide or reference_family not in wide:
                continue
            for block_length in [analysis.primary_block_length, *analysis.sensitivity_block_lengths]:
                if len(wide) < block_length:
                    continue
                effect = paired_moving_block_bootstrap(
                    wide[primary_family].to_numpy(),
                    wide[reference_family].to_numpy(),
                    block_length=block_length,
                    bootstrap_replicates=analysis.bootstrap_replicates,
                    seed=2027,
                )
                effect_rows.append(
                    {
                        "metric": metric,
                        "base_id": base_id,
                        "experiment_id": experiment_id,
                        "method": primary_family,
                        "reference": reference_family,
                        **asdict(effect),
                    }
                )
    method_summary = pd.concat(summary_frames, ignore_index=True)
    method_summary.to_csv(report_dir / "method_summary.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(report_dir / "paired_effects.csv", index=False)

    for metric in metrics:
        figure_data = method_summary[method_summary["metric"] == metric].groupby(
            ["experiment_id", "method"], as_index=False
        )["value"].mean()
        figure, axis = plt.subplots(figsize=(max(7.0, 0.7 * len(figure_data)), 4.5))
        labels = [f"{row.experiment_id}\n{row.method}" for row in figure_data.itertuples()]
        axis.bar(labels, figure_data["value"])
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(report_dir / f"method_comparison_{metric}.png", dpi=180)
        figure.savefig(report_dir / f"method_comparison_{metric}.pdf")
        plt.close(figure)

    _write_report_summary(report_dir, metrics, analysis)


def _write_report_summary(report_dir: Path, metrics: Sequence[str], analysis: CompositeAnalysisConfig) -> None:
    """Plain-text index so a report directory can be understood without opening every file."""

    block_lengths = [analysis.primary_block_length, *analysis.sensitivity_block_lengths]
    lines = [
        "# Report summary",
        "",
        f"Metrics: {', '.join(metrics)}.",
        f"Reference: `{analysis.reference}`. Bootstrap: {analysis.bootstrap_replicates} replicates, "
        f"block lengths {block_lengths}.",
        "",
        "| File | Contents |",
        "|---|---|",
    ]
    for path in sorted(report_dir.iterdir()):
        if path.name == "report_summary.md":
            continue
        description = _FIXED_FILE_DESCRIPTIONS.get(path.name, "")
        if not description and path.stem.startswith("method_comparison_"):
            metric = path.stem.removeprefix("method_comparison_")
            description = f"bar chart of `{metric}` per experiment/method ({path.suffix.lstrip('.')})"
        lines.append(f"| `{path.name}` | {description} |")
    (report_dir / "report_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
