"""Consolidate a completed composite run's per-cell records into a report."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from simcast.config import CompositeExperimentConfig
from simcast.evaluation.uncertainty import paired_moving_block_bootstrap


def write_composite_report(
    report_dir: Path,
    cells: list[dict[str, object]],
    config: CompositeExperimentConfig,
) -> None:
    """Write pooled per-origin metrics, paired effects, and a summary figure."""

    report_dir.mkdir(parents=True, exist_ok=True)
    origin_frames: list[pd.DataFrame] = []
    effect_rows: list[dict[str, object]] = []
    comparisons: set[tuple[str, str, str, str]] = set()
    for cell in cells:
        evaluation = Path(str(cell["evaluation_path"]))
        frame = pd.read_parquet(evaluation / "per_origin_metrics.parquet")
        frame["base_id"] = str(cell["base_id"])
        frame["experiment_id"] = str(cell["experiment_id"])
        frame["configured_seed"] = "deterministic" if cell["seed"] is None else str(cell["seed"])
        origin_frames.append(frame)
        primary_family = str(cell["method_family"])
        reference_family = str(cell["reference_family"])
        if primary_family == reference_family:
            continue
        comparisons.add(
            (str(cell["base_id"]), str(cell["experiment_id"]), primary_family, reference_family)
        )
    per_origin = pd.concat(origin_frames, ignore_index=True)
    per_origin.to_parquet(report_dir / "per_origin_metrics.parquet", index=False)
    for base_id, experiment_id, primary_family, reference_family in sorted(comparisons):
        comparison = per_origin[
            (per_origin["base_id"] == base_id) & (per_origin["experiment_id"] == experiment_id)
        ]
        # Averaging here treats optimization seeds as repeated fits, not as
        # additional test observations. Repeated deterministic reference rows
        # collapse to their common value in the same operation.
        averaged = comparison.groupby(["method", "origin"], as_index=False)["mean_pinball"].mean()
        wide = averaged.pivot(index="origin", columns="method", values="mean_pinball").dropna()
        if primary_family not in wide or reference_family not in wide:
            continue
        for block_length in [config.analysis.primary_block_length, *config.analysis.sensitivity_block_lengths]:
            if len(wide) < block_length:
                continue
            effect = paired_moving_block_bootstrap(
                wide[primary_family].to_numpy(),
                wide[reference_family].to_numpy(),
                block_length=block_length,
                bootstrap_replicates=config.analysis.bootstrap_replicates,
                seed=2027,
            )
            effect_rows.append(
                {
                    "base_id": base_id,
                    "experiment_id": experiment_id,
                    "method": primary_family,
                    "reference": reference_family,
                    **asdict(effect),
                }
            )
    summary = per_origin.groupby(
        ["base_id", "experiment_id", "method", "configured_seed"],
        dropna=False,
        as_index=False,
    ).agg(
        primary_metric=("mean_pinball", "mean")
    )
    summary.to_csv(report_dir / "method_summary.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(report_dir / "paired_effects.csv", index=False)
    figure_data = summary.groupby(["experiment_id", "method"], as_index=False)["primary_metric"].mean()
    figure, axis = plt.subplots(figsize=(max(7.0, 0.7 * len(figure_data)), 4.5))
    labels = [f"{row.experiment_id}\n{row.method}" for row in figure_data.itertuples()]
    axis.bar(labels, figure_data["primary_metric"])
    axis.set_ylabel("mean aggregate pinball loss")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(report_dir / "method_comparison.png", dpi=180)
    figure.savefig(report_dir / "method_comparison.pdf")
    plt.close(figure)
