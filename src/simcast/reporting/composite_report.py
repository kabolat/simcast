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
from simcast.evaluation.plots import cross_entity_statistic_label
from simcast.evaluation.uncertainty import paired_moving_block_bootstrap

_FIXED_FILE_DESCRIPTIONS = {
    "per_origin_metrics.parquet": "concatenated per-origin, per-method evaluation records; input to every file below",
    "per_origin_lead_metrics.parquet": (
        "concatenated valid-case forecasts and scores used for calibration and forecast galleries"
    ),
    "method_summary.csv": "mean value per (base, evaluated method, seed), one row per metric",
    "paired_effect_bs.csv": "moving-block bootstrap paired contrasts vs. the reference, one row per metric/comparison",
    "paired_effect_dist.parquet": "seed-averaged paired score improvements per origin; positive favors the method",
    "paired_effect_bs_<metric>.png": (
        "vertical base panels with relative-improvement intervals for all block lengths"
    ),
    "paired_effect_dist_<metric>.png": (
        "per-origin score-improvement box plots with better-origin fractions and bootstrap mean intervals"
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
    case_frames: list[pd.DataFrame] = []
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
        cases = pd.read_parquet(evaluation / "per_origin_lead_metrics.parquet")
        cases["base_id"] = str(cell["base_id"])
        cases["method_id"] = str(cell["method_id"])
        cases["configured_seed"] = "deterministic" if cell["seed"] is None else str(cell["seed"])
        case_frames.append(cases)
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
    per_origin_lead = pd.concat(case_frames, ignore_index=True)
    per_origin_lead.to_parquet(report_dir / "per_origin_lead_metrics.parquet", index=False)
    primary_origin = pd.concat(primary_frames, ignore_index=True)

    summary_frames = []
    effect_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for metric in metrics:
        frame = primary_origin.groupby(
            ["base_id", "method_id", "method", "configured_seed"], dropna=False, as_index=False
        ).agg(value=(metric, "mean"))
        frame.insert(0, "metric", metric)
        summary_frames.append(frame)

        for base_id, method_id, primary_family, reference_family in sorted(comparisons):
            comparison = per_origin[
                (per_origin["base_id"] == base_id)
                & per_origin["method_id"].isin([method_id, reference])
            ]
            # Averaging here treats optimization seeds as repeated fits, not as
            # additional test observations. Repeated deterministic reference rows
            # collapse to their common value in the same operation.
            averaged = comparison.groupby(["method_id", "origin"], as_index=False)[metric].mean()
            wide = averaged.pivot(index="origin", columns="method_id", values=metric).dropna()
            if method_id not in wide or reference not in wide:
                continue
            paired = wide[[method_id, reference]].replace([np.inf, -np.inf], np.nan).dropna()
            block_lengths = [analysis.primary_block_length, *analysis.sensitivity_block_lengths]
            eligible_block_lengths = [block_length for block_length in block_lengths if len(paired) >= block_length]
            if not eligible_block_lengths:
                continue
            origin_indices = (
                comparison.drop_duplicates("origin").set_index("origin")["origin_index"].reindex(paired.index)
            )
            improvements = paired[reference] - paired[method_id]
            for origin, improvement in improvements.items():
                distribution_rows.append(
                    {
                        "metric": metric,
                        "base_id": base_id,
                        "method_id": method_id,
                        "method": primary_family,
                        "reference": reference_family,
                        "origin_index": origin_indices.loc[origin],
                        "origin": origin,
                        "score_improvement": float(improvement),
                        "method_better": bool(improvement > 0),
                    }
                )
            for block_length in eligible_block_lengths:
                effect = paired_moving_block_bootstrap(
                    paired[method_id].to_numpy(),
                    paired[reference].to_numpy(),
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
    pd.DataFrame(effect_rows).to_csv(report_dir / "paired_effect_bs.csv", index=False)
    if distribution_rows:
        pd.DataFrame(distribution_rows).to_parquet(report_dir / "paired_effect_dist.parquet", index=False)
    _write_coverage_figure(per_origin_lead, report_dir)
    _write_quantile_coverage_figure(per_origin_lead, report_dir)
    _write_aggregate_fan_galleries(
        per_origin_lead,
        report_dir,
        reference=reference,
        metrics=metrics,
    )

    for metric in metrics:
        figure_data = method_summary[method_summary["metric"] == metric]
        base_ids = list(figure_data["base_id"].drop_duplicates())
        figure, axes = plt.subplots(len(base_ids), 1, squeeze=False, figsize=(7.0, 4.5 * len(base_ids)))
        for axis, base_id in zip(axes[:, 0], base_ids, strict=True):
            base_data = figure_data[figure_data["base_id"] == base_id]
            bars = axis.bar(
                [f"{row.method_id}\n{row.method}" for row in base_data.itertuples()],
                base_data["value"],
            )
            axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
            axis.margins(y=0.15)
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
            figure, axes = plt.subplots(
                len(base_ids), 1, squeeze=False, figsize=(7.0, 4.5 * len(base_ids)), layout="constrained"
            )
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
                    x_positions = [
                        index + (block_index - (len(block_lengths) - 1) / 2) * jitter
                        for index in range(len(block_effects))
                    ]
                    axis.errorbar(
                        x_positions,
                        improvements,
                        yerr=[lower_errors, upper_errors],
                        fmt=marker,
                        capsize=4,
                        label=f"block {block_length}",
                    )
                    if block_length == analysis.primary_block_length:
                        axis.set_xticks(range(len(block_effects)), labels, rotation=45, ha="right")
                        for x_position, improvement in zip(x_positions, improvements, strict=True):
                            axis.annotate(
                                f"{improvement:+.1f}%",
                                (x_position, improvement),
                                xytext=(5, 0),
                                textcoords="offset points",
                                ha="left",
                                va="center",
                                fontsize=7,
                            )
                axis.axhline(0.0, color="black", linewidth=1.0)
                axis.margins(x=0.2)
                axis.set_title(str(base_id))
                axis.set_ylabel("relative improvement vs reference (%)")
                axis.legend()
            figure.suptitle("Paired relative improvement by bootstrap block length")
            figure.savefig(report_dir / f"paired_effect_bs_{metric}.png", dpi=180)
            plt.close(figure)

        distribution = pd.DataFrame(distribution_rows)
        distribution = distribution[distribution["metric"] == metric]
        if not distribution.empty:
            base_ids = list(distribution["base_id"].drop_duplicates())
            figure, axes = plt.subplots(
                len(base_ids),
                1,
                squeeze=False,
                figsize=(7.0, 4.5 * len(base_ids)),
                layout="constrained",
            )
            for axis, base_id in zip(axes[:, 0], base_ids, strict=True):
                base_distribution = distribution[distribution["base_id"] == base_id]
                comparisons_for_base = base_distribution[["method_id", "method"]].drop_duplicates()
                values = []
                labels = []
                for method_id, method in comparisons_for_base.itertuples(index=False, name=None):
                    method_rows = base_distribution[base_distribution["method_id"] == method_id]
                    values.append(method_rows["score_improvement"].to_numpy())
                    better_fraction = method_rows["method_better"].mean()
                    labels.append(f"{method_id}\n{method}\n{better_fraction:.0%} origins better")
                positions = np.arange(1, len(values) + 1)
                axis.boxplot(values, positions=positions, patch_artist=True)
                axis.axhline(0.0, color="black", linewidth=1.0)
                axis.set_xticks(positions, labels, rotation=35, ha="right")
                axis.set_title(str(base_id))
                axis.set_ylabel("Per-origin score improvement (M0 - method)")
            figure.suptitle(f"Per-origin {metric} improvement; positive favors the method")
            figure.savefig(report_dir / f"paired_effect_dist_{metric}.png", dpi=180)
            plt.close(figure)

    old_effect_figures = [
        path
        for path in report_dir.glob("paired_effect_*.png")
        if not path.stem.startswith(("paired_effect_bs_", "paired_effect_dist_"))
    ]
    for old_artifact in [report_dir / "paired_effects.csv", *old_effect_figures]:
        old_artifact.unlink(missing_ok=True)
    for old_pdf in report_dir.glob("method_comparison_*.pdf"):
        old_pdf.unlink()

    _write_report_summary(report_dir, metrics, analysis, reference)


def _representative_origins(reference_rows: pd.DataFrame) -> list[tuple[object, str]]:
    rows = reference_rows[reference_rows["valid"]]
    if rows.empty:
        return []
    origin_values = (
        rows.groupby(["origin_index", "origin"], as_index=False)
        .agg(observed_aggregate=("observed_aggregate", "mean"))
        .sort_values("origin")
        .reset_index(drop=True)
    )
    targets = np.quantile(origin_values["observed_aggregate"], [0.1, 0.5, 0.9])
    labels = ("low observed", "median observed", "high observed")
    selected: list[tuple[object, str]] = []
    seen: set[object] = set()
    values = origin_values["observed_aggregate"].to_numpy()
    for target, label in zip(targets, labels, strict=True):
        index = int(np.argmin(np.abs(values - target)))
        origin_index = origin_values.iloc[index]["origin_index"]
        if origin_index not in seen:
            selected.append((origin_index, label))
            seen.add(origin_index)
    return selected


def _write_aggregate_fan_galleries(
    per_origin_lead: pd.DataFrame,
    report_dir: Path,
    *,
    reference: str,
    metrics: Sequence[str],
) -> None:
    quantile_columns = sorted(
        (column for column in per_origin_lead.columns if column.startswith("aggregate_q")),
        key=lambda column: float(column.removeprefix("aggregate_q")),
    )
    interval_columns = sorted(
        (
            column.removeprefix("interval_lower_")
            for column in per_origin_lead.columns
            if column.startswith("interval_lower_")
            and f"interval_upper_{column.removeprefix('interval_lower_')}" in per_origin_lead
        ),
        key=float,
        reverse=True,
    )
    if not quantile_columns or "observed_aggregate" not in per_origin_lead:
        return
    median_column = min(quantile_columns, key=lambda column: abs(float(column.removeprefix("aggregate_q")) - 0.5))
    method_colors = plt.get_cmap("tab10")

    for base_id in per_origin_lead["base_id"].drop_duplicates():
        base_rows = per_origin_lead[per_origin_lead["base_id"] == base_id]
        reference_rows = base_rows[base_rows["method_id"] == reference]
        if reference_rows.empty:
            continue
        examples = _representative_origins(reference_rows)
        if not examples:
            continue
        methods = list(base_rows[["method_id", "method"]].drop_duplicates().itertuples(index=False, name=None))
        figure, axes = plt.subplots(
            len(examples),
            len(methods),
            squeeze=False,
            figsize=(4.2 * len(methods), 2.8 * len(examples)),
            sharex=True,
            sharey="row",
        )
        for row_index, (origin_index, example_label) in enumerate(examples):
            origin_rows = base_rows[
                (base_rows["origin_index"] == origin_index) & base_rows["valid"]
            ]
            for method_index, (method_id, method_name) in enumerate(methods):
                axis = axes[row_index, method_index]
                method_rows = origin_rows[origin_rows["method_id"] == method_id]
                if method_rows.empty:
                    axis.set_visible(False)
                    continue
                by_lead = method_rows.groupby("lead", sort=True)
                leads = np.asarray(list(by_lead.groups))
                color = method_colors(method_index % 10)
                for interval_level in interval_columns:
                    lower = by_lead[f"interval_lower_{interval_level}"].mean().to_numpy()
                    upper = by_lead[f"interval_upper_{interval_level}"].mean().to_numpy()
                    axis.fill_between(
                        leads,
                        lower,
                        upper,
                        color=color,
                        alpha=0.08,
                    )
                median = by_lead[median_column].mean().to_numpy()
                observed = by_lead["observed_aggregate"].mean().to_numpy()
                axis.plot(leads, median, color=color, label="Seed-mean median")
                axis.plot(leads, observed, color="black", linewidth=1.0, label="Observed")
                crps_label = ""
                if "crps" in metrics and "crps" in method_rows:
                    crps_label = f" | mean CRPS {method_rows['crps'].mean():.3g}"
                origin_time = str(method_rows["origin"].iloc[0])[:10]
                axis.set_title(f"{example_label}: {origin_time}\n{method_id} ({method_name}){crps_label}")
                axis.set_xlabel("Lead")
                if method_index == 0:
                    axis.set_ylabel(cross_entity_statistic_label(str(method_rows["cross_entity_statistic"].iloc[0])))
                axis.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5))
        statistic_label = (
            cross_entity_statistic_label(str(base_rows["cross_entity_statistic"].iloc[0]))
            if "cross_entity_statistic" in base_rows
            else "Cross-entity statistic"
        )
        figure.suptitle(
            f"{base_id}: observed-outcome examples ({statistic_label}); seed-mean forecasts are illustrative",
            y=1.02,
        )
        figure.tight_layout()
        figure.savefig(report_dir / f"aggregate_fan_gallery_{base_id}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def _write_coverage_figure(per_origin: pd.DataFrame, report_dir: Path) -> None:
    if "valid" in per_origin:
        per_origin = per_origin[per_origin["valid"]]
    coverage_columns = sorted(
        (column for column in per_origin.columns if column.startswith("coverage_")),
        key=lambda column: float(column.removeprefix("coverage_")),
    )
    if not coverage_columns:
        return
    nominal = np.asarray([float(column.removeprefix("coverage_")) for column in coverage_columns])
    base_ids = list(per_origin["base_id"].drop_duplicates())
    figure, axes = plt.subplots(len(base_ids), 1, squeeze=False, figsize=(7.0, 4.5 * len(base_ids)))
    for axis, base_id in zip(axes[:, 0], base_ids, strict=True):
        base_rows = per_origin[per_origin["base_id"] == base_id]
        axis.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", label="nominal")
        for method in base_rows["method"].drop_duplicates():
            empirical = [
                float(base_rows.loc[base_rows["method"] == method, column].mean())
                for column in coverage_columns
            ]
            axis.plot(nominal, empirical, marker="o", label=method)
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="nominal coverage",
            ylabel="empirical coverage",
            title=str(base_id),
        )
        axis.set_aspect("equal", adjustable="box")
        axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "summary_coverage.png", dpi=180)
    plt.close(figure)


def _write_quantile_coverage_figure(per_origin: pd.DataFrame, report_dir: Path) -> None:
    quantile_columns = sorted(
        (column for column in per_origin.columns if column.startswith("aggregate_q")),
        key=lambda column: float(column.removeprefix("aggregate_q")),
    )
    if "observed_aggregate" not in per_origin or not quantile_columns:
        return
    if "valid" in per_origin:
        per_origin = per_origin[per_origin["valid"]]
    nominal = np.asarray([float(column.removeprefix("aggregate_q")) for column in quantile_columns])
    base_ids = list(per_origin["base_id"].drop_duplicates())
    figure, axes = plt.subplots(len(base_ids), 1, squeeze=False, figsize=(7.0, 4.5 * len(base_ids)))
    for axis, base_id in zip(axes[:, 0], base_ids, strict=True):
        base_rows = per_origin[per_origin["base_id"] == base_id]
        axis.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", label="nominal")
        for method in base_rows["method"].drop_duplicates():
            method_rows = base_rows[base_rows["method"] == method]
            empirical = [
                float((method_rows["observed_aggregate"] <= method_rows[column]).mean())
                for column in quantile_columns
            ]
            axis.plot(nominal, empirical, marker="o", label=method)
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="nominal aggregate quantile",
            ylabel="empirical aggregate coverage",
            title=str(base_id),
        )
        axis.set_aspect("equal", adjustable="box")
        axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "summary_quantile_calibration.png", dpi=180)
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
        if not description and path.name.startswith("aggregate_fan_gallery_"):
            description = (
                "aggregate fan examples selected by observed outcomes, with one panel per method and origin"
            )
        if not description and path.stem.startswith("method_comparison_"):
            metric = path.stem.removeprefix("method_comparison_")
            description = f"base-panel bar charts of `{metric}` per evaluated method, with values labeled (png)"
        if not description and path.stem.startswith("paired_effect_bs_"):
            metric = path.stem.removeprefix("paired_effect_bs_")
            description = (
                f"vertical base-panel relative-improvement plot of `{metric}` for all block lengths; "
                "primary-block means are labeled (png)"
            )
        if not description and path.stem.startswith("paired_effect_dist_"):
            metric = path.stem.removeprefix("paired_effect_dist_")
            description = (
                f"per-origin score-improvement box plots for `{metric}` with better-origin fractions "
                "and moving-block mean intervals (png)"
            )
        lines.append(f"| `{path.name}` | {description} |")
    (report_dir / "report_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
