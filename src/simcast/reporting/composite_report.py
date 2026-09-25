"""Consolidate a completed composite run's per-cell records into a report."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from simcast.config import CompositeAnalysisConfig
from simcast.evaluation.uncertainty import paired_moving_block_bootstrap

_FIXED_FILE_DESCRIPTIONS = {
    "per_origin_metrics.parquet": "concatenated per-origin, per-method evaluation records; input to every file below",
    "method_summary.csv": "mean value per (base, evaluated method, seed), one row per metric",
    "paired_effects.csv": "moving-block bootstrap paired contrasts vs. the reference, one row per metric/comparison",
    "paired_effect_<metric>.png": (
        "vertical base panels with relative-improvement intervals for all block lengths"
    ),
    "report_summary.md": "this file",
}


def write_composite_report(
    report_dir: Path,
    cells: list[dict[str, object]],
    analysis: CompositeAnalysisConfig,
    reference: str,
    metrics: Sequence[str] = ("mean_pinball",),
) -> None:
    """Write pooled per-origin metrics, paired effects, and per-metric figures."""

    origin_frames: list[pd.DataFrame] = []
    primary_frames: list[pd.DataFrame] = []
    comparisons: set[tuple[str, str, str, str]] = set()
    reference_families: dict[str, str] = {}
    for cell in cells:
        evaluation = Path(str(cell["evaluation_path"]))
        frame = pd.read_parquet(evaluation / "per_origin_metrics.parquet")
        frame["base_id"] = str(cell["base_id"])
        frame["method_id"] = str(cell["method_id"])
        frame["configured_seed"] = "deterministic" if cell["seed"] is None else str(cell["seed"])
        origin_frames.append(frame)
        primary_frames.append(frame)
        if str(cell["method_id"]) == reference:
            reference_families[str(cell["base_id"])] = str(cell["method_family"])
    for cell in cells:
        method_id = str(cell["method_id"])
        if method_id == reference:
            continue
        reference_family = reference_families.get(str(cell["base_id"]))
        if reference_family is None:
            continue
        comparisons.add(
            (str(cell["base_id"]), method_id, str(cell["method_family"]), reference_family)
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
            ["base_id", "method_id", "method", "configured_seed"], dropna=False, as_index=False
        ).agg(value=(metric, "mean"))
        frame.insert(0, "metric", metric)
        summary_frames.append(frame)

        for base_id, method_id, primary_family, reference_family in sorted(comparisons):
            comparison = per_origin[per_origin["base_id"] == base_id]
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
                        "method_id": method_id,
                        "method": primary_family,
                        "reference": reference_family,
                        **asdict(effect),
                    }
                )
    method_summary = pd.concat(summary_frames, ignore_index=True)
    method_summary.to_csv(report_dir / "method_summary.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(report_dir / "paired_effects.csv", index=False)
    _write_coverage_figure(per_origin, report_dir)

    for metric in metrics:
        figure_data = method_summary[method_summary["metric"] == metric]
        base_ids = list(figure_data["base_id"].drop_duplicates())
        figure, axes = plt.subplots(len(base_ids), 1, squeeze=False, figsize=(7.0, 4.5 * len(base_ids)))
        for axis, base_id in zip(axes[:, 0], base_ids, strict=True):
            base_data = figure_data[figure_data["base_id"] == base_id]
            axis.bar(
                [f"{row.method_id}\n{row.method}" for row in base_data.itertuples()],
                base_data["value"],
            )
            axis.set_title(str(base_id))
            axis.set_ylabel(metric)
            axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(report_dir / f"method_comparison_{metric}.png", dpi=180)
        plt.close(figure)

        primary_effects = pd.DataFrame(effect_rows)
        primary_effects = primary_effects[primary_effects["metric"] == metric]
        if not primary_effects.empty:
            base_ids = list(primary_effects["base_id"].drop_duplicates())
            block_lengths = [analysis.primary_block_length, *analysis.sensitivity_block_lengths]
            markers = ["o", "s", "^", "D", "P", "X"]
            jitter = 0.12
            figure, axes = plt.subplots(len(base_ids), 1, squeeze=False, figsize=(7.0, 4.5 * len(base_ids)))
            for axis, base_id in zip(axes[:, 0], base_ids, strict=True):
                base_effects = primary_effects[primary_effects["base_id"] == base_id]
                comparison_keys = base_effects[["method_id", "method"]].drop_duplicates()
                for block_index, (marker, block_length) in enumerate(zip(markers, block_lengths, strict=False)):
                    block_effects = base_effects[base_effects["block_length"] == block_length].merge(
                        comparison_keys, on=["method_id", "method"], how="inner"
                    )
                    if block_effects.empty:
                        continue
                    labels = [f"{row.method_id}\n{row.method}" for row in block_effects.itertuples()]
                    improvements = -block_effects["percentage_difference"].to_numpy()
                    lower_errors = np.maximum(
                        improvements + block_effects["percentage_ci_upper"].to_numpy(), 0.0
                    )
                    upper_errors = np.maximum(
                        -block_effects["percentage_ci_lower"].to_numpy() - improvements, 0.0
                    )
                    axis.errorbar(
                        [index + (block_index - (len(block_lengths) - 1) / 2) * jitter
                         for index in range(len(block_effects))],
                        improvements,
                        yerr=[lower_errors, upper_errors],
                        fmt=marker,
                        capsize=4,
                        label=f"block {block_length}",
                    )
                    if block_length == analysis.primary_block_length:
                        axis.set_xticks(range(len(block_effects)), labels, rotation=45, ha="right")
                axis.axhline(0.0, color="black", linewidth=1.0)
                axis.set_title(str(base_id))
                axis.set_ylabel("relative improvement vs reference (%)")
                axis.legend()
            figure.suptitle("Paired relative improvement by bootstrap block length")
            figure.tight_layout()
            figure.savefig(report_dir / f"paired_effect_{metric}.png", dpi=180)
            plt.close(figure)

    for old_pdf in report_dir.glob("method_comparison_*.pdf"):
        old_pdf.unlink()

    _write_report_summary(report_dir, metrics, analysis, reference)


def _write_coverage_figure(per_origin: pd.DataFrame, report_dir: Path) -> None:
    coverage_columns = sorted(
        column for column in per_origin.columns if column.startswith("coverage_")
    )
    if not coverage_columns:
        return
    methods = list(per_origin["method"].drop_duplicates())
    figure, axes = plt.subplots(
        len(coverage_columns),
        1,
        squeeze=False,
        figsize=(7.0, 3.5 * len(coverage_columns)),
    )
    for axis, column in zip(axes[:, 0], coverage_columns, strict=True):
        nominal = float(column.removeprefix("coverage_"))
        values = [
            float(per_origin.loc[per_origin["method"] == method, column].mean())
            for method in methods
        ]
        axis.axhline(nominal, color="black", linestyle="--", label="nominal")
        axis.plot(methods, values, marker="o")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("empirical coverage")
        axis.set_title(f"Nominal interval coverage: {nominal:g}")
        axis.tick_params(axis="x", rotation=30)
        axis.legend()
    figure.supxlabel("method")
    figure.tight_layout()
    figure.savefig(report_dir / "summary_coverage.png", dpi=180)
    plt.close(figure)


def _write_report_summary(
    report_dir: Path, metrics: Sequence[str], analysis: CompositeAnalysisConfig, reference: str
) -> None:
    """Plain-text index so a report directory can be understood without opening every file."""

    block_lengths = [analysis.primary_block_length, *analysis.sensitivity_block_lengths]
    lines = [
        "# Report summary",
        "",
        f"Metrics: {', '.join(metrics)}.",
        f"Reference: `{reference}`. Bootstrap: {analysis.bootstrap_replicates} replicates, "
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
            description = f"base-panel bar charts of `{metric}` per evaluated method (png)"
        if not description and path.stem.startswith("paired_effect_"):
            metric = path.stem.removeprefix("paired_effect_")
            description = f"vertical base-panel relative-improvement plot of `{metric}` for all block lengths (png)"
        lines.append(f"| `{path.name}` | {description} |")
    (report_dir / "report_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
