"""Aggregate saved confirmatory artifacts without fitting or retraining models."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml  # type: ignore[import-untyped]

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from simcast.evaluation.uncertainty import paired_moving_block_bootstrap, select_representative_origins
from simcast.fm.cache import load_pit_library
from simcast.fm.pit import nominal_cell_midpoints

METHOD_LABELS = {
    "independent": "M0",
    "static_gaussian": "M1",
    "conditional_low_rank": "M2",
    "set_aware_low_rank": "M3",
    "conditional_kernel": "M4",
}
METHOD_ORDER = tuple(METHOD_LABELS)
PAIR_COMPARISONS = (
    ("static_gaussian", "independent"),
    ("conditional_low_rank", "independent"),
    ("set_aware_low_rank", "independent"),
    ("conditional_low_rank", "static_gaussian"),
    ("set_aware_low_rank", "static_gaussian"),
    ("set_aware_low_rank", "conditional_low_rank"),
)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def _artifact_frames(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    per_case: list[pd.DataFrame] = []
    per_origin: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for manifest_path in sorted(input_root.rglob("evaluation_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol = manifest.get("experimental_protocol", {})
        if protocol.get("name") != "powertech2027" or protocol.get("full_group_only") is not True:
            continue
        directory = manifest_path.parent
        case_path = directory / "per_origin_lead_metrics.parquet"
        origin_path = directory / "per_origin_metrics.parquet"
        if not case_path.is_file() or not origin_path.is_file():
            continue
        config = _read_yaml(directory / "resolved_config.yaml")
        confirmatory = manifest.get("confirmatory", config.get("confirmatory", {}))
        dependence = config["dependence"]
        labels = {
            "experiment_family": confirmatory["experiment_family"],
            "feature_set": confirmatory["feature_set"],
            "pit_transform": config["pit"]["dependence_transform"],
            "m2_rank": dependence["conditional_low_rank"]["latent_rank"],
            "m3_rank": dependence["set_aware_low_rank"]["latent_rank"],
            "static_variant": "pooled" if dependence["static_gaussian"]["share_across_leads"] else "lead_specific",
            "evaluation_dir": str(directory),
        }
        per_case.append(pd.read_parquet(case_path).assign(**labels))
        per_origin.append(pd.read_parquet(origin_path).assign(**labels))
        records.append({"manifest": manifest, "config": config, **labels})
    if not records:
        raise FileNotFoundError(f"no complete powertech2027 evaluation artifacts found under {input_root}")
    return pd.concat(per_case, ignore_index=True), pd.concat(per_origin, ignore_index=True), records


def _main_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["experiment_family"] == "main")
        & (frame["feature_set"] == "full")
        & (frame["pit_transform"] == "nominal_cells")
        & (frame["m2_rank"] == 4)
        & (frame["m3_rank"] == 4)
        & (frame["static_variant"] == "lead_specific")
    ].copy()


def _deduplicate_deterministic(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    neural = frame[frame["method"].isin(["conditional_low_rank", "set_aware_low_rank", "conditional_kernel"])]
    deterministic = frame[~frame.index.isin(neural.index)].drop_duplicates(list(keys))
    return pd.concat([deterministic, neural], ignore_index=True)


def _method_seed_summary(per_origin: pd.DataFrame) -> pd.DataFrame:
    fixed = {
        "mean_pinball",
        "crps",
        "weighted_interval_score",
        "energy_score",
        "variogram_score",
        "test_pseudo_nll",
    }
    metrics = [
        column
        for raw_column in per_origin.columns
        for column in (str(raw_column),)
        if column in fixed
        or column.startswith("coverage_")
        or column.startswith("interval_width_")
        or column.startswith("interval_score_")
    ]
    return (
        per_origin.groupby(["group", "method", "neural_seed"], dropna=False, as_index=False)[metrics]
        .mean()
        .rename(columns={"mean_pinball": "primary_metric"})
    )


def _training_records(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        manifest = record["manifest"]
        for method, raw_path in manifest.get("method_runs", {}).items():
            run_path = Path(raw_path)
            metadata_path = run_path / "run_metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            row = {
                "group": manifest["group"]["name"],
                "method": method,
                "neural_seed": (
                    metadata.get("seed") if method in {"conditional_low_rank", "set_aware_low_rank"} else None
                ),
                "experiment_family": record["experiment_family"],
                "feature_set": record["feature_set"],
                "pit_transform": record["pit_transform"],
                "m2_rank": record["m2_rank"],
                "m3_rank": record["m3_rank"],
                "static_variant": record["static_variant"],
                "selected_epoch": np.nan,
                "validation_pseudo_nll": np.nan,
                "parameter_count": np.nan,
                "run_path": str(run_path),
            }
            summary_path = run_path / "training_summary.json"
            checkpoint_path = run_path / "best.pt"
            if summary_path.is_file():
                training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row["selected_epoch"] = training_summary["best_epoch"]
                row["validation_pseudo_nll"] = training_summary["best_validation_nll"]
            if checkpoint_path.is_file():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                state = checkpoint.get("model_state_dict", {})
                row["parameter_count"] = sum(
                    value.numel() for value in state.values() if isinstance(value, torch.Tensor)
                )
            rows.append(row)
    return pd.DataFrame(rows).drop_duplicates()


def _group_method_summary(per_case: pd.DataFrame, per_origin: pd.DataFrame) -> pd.DataFrame:
    seed = _method_seed_summary(per_origin)
    metrics = [column for column in seed.columns if column not in {"group", "method", "neural_seed"}]
    mean = seed.groupby(["group", "method"], as_index=False)[metrics].mean()
    seed_std = (
        seed.groupby(["group", "method"])["primary_metric"].std(ddof=1).rename("neural_seed_std").reset_index()
    )
    valid = (
        per_case[per_case["valid"]]
        .drop_duplicates(["group", "method", "origin", "lead"])
        .groupby(["group", "method"], as_index=False)
        .size()
        .rename(columns={"size": "number_of_valid_cases"})
    )
    cardinality = per_case.groupby(["group", "method"], as_index=False)["K"].first()
    result = mean.merge(seed_std, on=["group", "method"]).merge(valid, on=["group", "method"]).merge(
        cardinality, on=["group", "method"]
    )
    comparisons = (("independent", "percentage_change_vs_M0"), ("static_gaussian", "percentage_change_vs_M1"))
    for baseline, output in comparisons:
        reference = result[result["method"] == baseline].set_index("group")["primary_metric"]
        result[output] = 100.0 * (
            result["primary_metric"] - result["group"].map(reference)
        ) / result["group"].map(reference)
    result["method_label"] = result["method"].map(METHOD_LABELS)
    result["method_order"] = result["method"].map({name: index for index, name in enumerate(METHOD_ORDER)})
    return result.sort_values(["group", "method_order"]).drop(columns="method_order")


def _paired_effects(
    per_origin: pd.DataFrame,
    *,
    block_lengths: Sequence[int],
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    averaged = per_origin.groupby(["group", "method", "origin"], as_index=False)["mean_pinball"].mean()
    rows: list[dict[str, Any]] = []
    for group, group_rows in averaged.groupby("group", sort=False):
        wide = group_rows.pivot(index="origin", columns="method", values="mean_pinball").sort_index()
        for method_a, method_b in PAIR_COMPARISONS:
            if method_a not in wide or method_b not in wide:
                continue
            paired = wide[[method_a, method_b]].dropna()
            for block_length in block_lengths:
                if len(paired) < block_length:
                    continue
                effect = paired_moving_block_bootstrap(
                    paired[method_a].to_numpy(),
                    paired[method_b].to_numpy(),
                    block_length=block_length,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=seed,
                )
                rows.append(
                    {
                        "group": group,
                        "method_A": method_a,
                        "method_B": method_b,
                        "absolute_paired_difference": effect.mean_difference,
                        "percentage_difference": effect.percentage_difference,
                        "block_length": effect.block_length,
                        "ci_lower": effect.ci_lower,
                        "ci_upper": effect.ci_upper,
                        "percentage_ci_lower": effect.percentage_ci_lower,
                        "percentage_ci_upper": effect.percentage_ci_upper,
                        "number_of_origins": effect.number_of_origins,
                        "bootstrap_replicates": effect.bootstrap_replicates,
                    }
                )
    return pd.DataFrame(rows)


def _main_results_table(summary: pd.DataFrame, effects: pd.DataFrame, block_length: int) -> pd.DataFrame:
    table = summary.copy()
    primary = effects[effects["block_length"] == block_length]
    for baseline, label in (("independent", "M0"), ("static_gaussian", "M1")):
        lookup = primary[primary["method_B"] == baseline].set_index(["group", "method_A"])
        keys = list(zip(table["group"], table["method"], strict=True))
        table[f"paired_percentage_ci_vs_{label}_lower"] = [
            lookup.loc[key, "percentage_ci_lower"] if key in lookup.index else np.nan for key in keys
        ]
        table[f"paired_percentage_ci_vs_{label}_upper"] = [
            lookup.loc[key, "percentage_ci_upper"] if key in lookup.index else np.nan for key in keys
        ]
    columns = [
        "group",
        "K",
        "method_label",
        "primary_metric",
        "percentage_change_vs_M0",
        "percentage_change_vs_M1",
        "paired_percentage_ci_vs_M0_lower",
        "paired_percentage_ci_vs_M0_upper",
        "paired_percentage_ci_vs_M1_lower",
        "paired_percentage_ci_vs_M1_upper",
        "crps",
        "weighted_interval_score",
        "coverage_0.9",
        "interval_width_0.9",
    ]
    return table[[column for column in columns if column in table]]


def _sensitivity_summary(
    frame: pd.DataFrame,
    training: pd.DataFrame,
    family: str,
    dimensions: Sequence[str],
) -> pd.DataFrame:
    selected = frame[frame["experiment_family"] == family]
    if selected.empty:
        return pd.DataFrame()
    keys = ["group", "method", *dimensions, "neural_seed"]
    per_seed = selected.groupby(keys, dropna=False, as_index=False)["mean_pinball"].mean()
    training_columns = [*keys, "validation_pseudo_nll", "selected_epoch", "parameter_count"]
    if training.empty:
        training_selected = pd.DataFrame(columns=training_columns)
    else:
        training_selected = training[training["experiment_family"] == family]
    per_seed = per_seed.merge(
        training_selected[training_columns].drop_duplicates(keys),
        on=keys,
        how="left",
    )
    keys_without_seed = [key for key in keys if key != "neural_seed"]
    result = per_seed.groupby(keys_without_seed, as_index=False).agg(
        aggregate_pinball=("mean_pinball", "mean"),
        seed_std=("mean_pinball", "std"),
        seed_min=("mean_pinball", "min"),
        seed_max=("mean_pinball", "max"),
        validation_pseudo_nll=("validation_pseudo_nll", "mean"),
        selected_epoch=("selected_epoch", "mean"),
        parameter_count=("parameter_count", "mean"),
    )
    return cast(
        pd.DataFrame,
        result.reset_index(drop=True).rename(
            columns={"mean": "aggregate_pinball", "std": "seed_std", "min": "seed_min", "max": "seed_max"}
        ),
    )


def _ablation_deltas(ablation: pd.DataFrame, main_summary: pd.DataFrame) -> pd.DataFrame:
    if ablation.empty:
        return ablation
    result = ablation.copy()
    m1 = main_summary[main_summary["method"] == "static_gaussian"].set_index("group")["primary_metric"]
    full = result[result["feature_set"] == "full"].set_index(["group", "method"])["aggregate_pinball"]
    result["delta_vs_M1"] = result["aggregate_pinball"] - result["group"].map(m1)
    result["percentage_change_vs_M1"] = 100.0 * result["delta_vs_M1"] / result["group"].map(m1)
    result["percentage_seed_std"] = 100.0 * result["seed_std"] / result["group"].map(m1)
    result["delta_vs_full_conditional"] = [
        row.aggregate_pinball - full.get((row.group, row.method), np.nan) for row in result.itertuples()
    ]
    return result


def _cache_characteristics(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        manifest = record["manifest"]
        group = str(manifest["group"]["name"])
        if group in seen:
            continue
        seen.add(group)
        library = load_pit_library(manifest["cache_path"], access="training")
        dataset = library.dataset
        split = np.asarray(dataset["split"].values)
        valid = np.asarray(dataset["pit_valid"].values, dtype=bool)
        train_z = np.asarray(dataset["pit_z"].values)[split == "train"]
        correlations: list[np.ndarray] = []
        for lead in range(train_z.shape[-1]):
            values = train_z[:, :, lead]
            values = values[np.isfinite(values).all(axis=1)]
            if len(values) >= 2:
                correlations.append(np.corrcoef(values, rowvar=False))
        off_diagonal = np.concatenate(
            [matrix[~np.eye(matrix.shape[0], dtype=bool)] for matrix in correlations]
        )
        pit_metadata = library.metadata["pit"]
        rows.append(
            {
                "group": group,
                "K": dataset.sizes["entity"],
                "valid_train_cases": int(valid[split == "train"].sum()),
                "valid_test_cases": int(valid[split == "test"].sum()),
                "raw_quantile_crossing_rate": float(pit_metadata["crossing_frequency"]),
                "mean_absolute_training_pit_correlation": float(np.mean(np.abs(off_diagonal))),
                "median_absolute_training_pit_correlation": float(np.median(np.abs(off_diagonal))),
            }
        )
    return pd.DataFrame(rows).sort_values("group")


def _write_table(table: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    table.to_csv(csv_path, index=False)
    tex_path.write_text(table.to_latex(index=False, float_format=lambda value: f"{value:.4g}"), encoding="utf-8")


def _write_main_table(table: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    """Write numeric CSV and bold the lowest primary score per group in LaTeX."""

    table.to_csv(csv_path, index=False)
    formatted = table.copy()
    best = formatted.groupby("group")["primary_metric"].idxmin()
    formatted["primary_metric"] = formatted["primary_metric"].map(lambda value: f"{value:.4g}")
    formatted.loc[best, "primary_metric"] = formatted.loc[best, "primary_metric"].map(
        lambda value: rf"\textbf{{{value}}}"
    )
    tex_path.write_text(formatted.to_latex(index=False, escape=False), encoding="utf-8")


def _save_figure(figure: Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _pipeline_figure(output: Path) -> None:
    labels = [
        "Frozen\nChronos-2",
        "Quantiles +\nembeddings",
        "Finite PIT\npseudo-data",
        "M0 / M1 /\nM2 / M3",
        "Correlated\nuniforms",
        "Fixed entity\nmarginals",
        "Full spatial\naggregate",
        "Proper\nscores",
    ]
    figure, axis = plt.subplots(figsize=(13, 2.2))
    axis.set_axis_off()
    for index, label in enumerate(labels):
        x = index / (len(labels) - 1)
        axis.text(x, 0.5, label, ha="center", va="center", bbox={"boxstyle": "round", "fc": "white"})
        if index:
            previous = (index - 1) / (len(labels) - 1)
            axis.annotate("", xy=(x - 0.055, 0.5), xytext=(previous + 0.055, 0.5), arrowprops={"arrowstyle": "->"})
    _save_figure(figure, output / "figure_A_method_pipeline")


def _forest_figure(effects: pd.DataFrame, primary_block_length: int, output: Path) -> None:
    selected = effects[
        (effects["block_length"] == primary_block_length)
        & effects["method_B"].isin(["independent", "static_gaussian"])
    ].copy()
    if selected.empty:
        return
    selected["label"] = selected["group"] + ": " + selected["method_A"].map(METHOD_LABELS) + " vs " + selected[
        "method_B"
    ].map(METHOD_LABELS)
    positions = np.arange(len(selected))
    figure, axis = plt.subplots(figsize=(8, max(3, 0.3 * len(selected))))
    axis.errorbar(
        selected["percentage_difference"],
        positions,
        xerr=np.vstack(
            (
                selected["percentage_difference"] - selected["percentage_ci_lower"],
                selected["percentage_ci_upper"] - selected["percentage_difference"],
            )
        ),
        fmt="o",
        capsize=3,
    )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(positions, selected["label"])
    axis.set_xlabel("Aggregate pinball change (%) — negative favours method A")
    axis.invert_yaxis()
    _save_figure(figure, output / "figure_B_paired_effect_forest")


def _calibration_figure(summary: pd.DataFrame, output: Path) -> None:
    coverage_columns = [str(column) for column in summary.columns if str(column).startswith("coverage_")]
    if not coverage_columns:
        return
    groups = [str(value) for value in summary["group"].drop_duplicates()]
    figure, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.5), squeeze=False)
    for axis, group in zip(axes[0], groups, strict=True):
        subset = summary[summary["group"] == group]
        nominal = np.asarray([float(column.removeprefix("coverage_")) for column in coverage_columns])
        for _, row in subset.iterrows():
            axis.plot(nominal, row[coverage_columns].to_numpy(dtype=float), marker="o", label=row["method_label"])
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        axis.set_title(group)
        axis.set_xlabel("Nominal coverage")
        axis.set_ylabel("Empirical coverage")
    axes[0, -1].legend()
    _save_figure(figure, output / "figure_C_aggregate_calibration")


def _ablation_figure(ablation: pd.DataFrame, output: Path) -> None:
    if ablation.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 4.5))
    feature_order = list(dict.fromkeys(ablation["feature_set"]))
    series = list(ablation.groupby(["group", "method"], sort=False))
    width = 0.8 / max(len(series), 1)
    for index, ((raw_group, raw_method), rows) in enumerate(series):
        group, method = str(raw_group), str(raw_method)
        subset = rows.set_index("feature_set").reindex(feature_order)
        axis.bar(
            np.arange(len(feature_order)) + index * width,
            subset["percentage_change_vs_M1"],
            width,
            yerr=subset["percentage_seed_std"],
            label=f"{group}:{METHOD_LABELS[method]}",
            capsize=2,
        )
    axis.set_xticks(np.arange(len(feature_order)) + width * (len(series) - 1) / 2, feature_order, rotation=20)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_ylabel("Aggregate pinball change vs M1 (%)")
    axis.legend(ncol=2)
    _save_figure(figure, output / "figure_F_feature_ablation")


def _pit_frequency_figure(records: Sequence[dict[str, Any]], output: Path) -> None:
    record = records[0]
    library = load_pit_library(record["manifest"]["cache_path"], access="training")
    dataset = library.dataset
    train = np.asarray(dataset["split"].values) == "train"
    u = torch.tensor(dataset["pit_u"].values[train])
    levels = torch.tensor(dataset["quantile"].values)
    midpoints = nominal_cell_midpoints(levels)
    cells = torch.abs(u.unsqueeze(-1) - midpoints).nan_to_num(float("inf")).argmin(dim=-1)
    frequencies = torch.stack(
        [((cells == index) & torch.isfinite(u)).sum() for index in range(len(midpoints))]
    ).float()
    frequencies /= frequencies.sum()
    nominal_edges = torch.cat((levels.new_tensor([0.0]), levels, levels.new_tensor([1.0])))
    nominal = nominal_edges[1:] - nominal_edges[:-1]
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.bar(np.arange(len(midpoints)) - 0.2, nominal.numpy(), width=0.4, label="Nominal")
    axis.bar(np.arange(len(midpoints)) + 0.2, frequencies.numpy(), width=0.4, label="Training empirical")
    axis.set_xlabel("Finite PIT cell")
    axis.set_ylabel("Probability")
    axis.set_title(str(record["manifest"]["group"]["name"]))
    axis.legend()
    _save_figure(figure, output / "figure_H_pit_cell_calibration")


def _main_record(records: Sequence[dict[str, Any]], group: str) -> dict[str, Any] | None:
    matches = [
        record
        for record in records
        if (
            record["manifest"]["group"]["name"] == group
            and record["experiment_family"] == "main"
            and record["feature_set"] == "full"
            and record["pit_transform"] == "nominal_cells"
            and record["m2_rank"] == 4
            and record["m3_rank"] == 4
            and record["static_variant"] == "lead_specific"
        )
    ]
    return matches[0] if matches else None


def _mean_main_correlations(records: Sequence[dict[str, Any]], group: str, method: str) -> np.ndarray:
    matching = [
        record
        for record in records
        if _main_record([record], group) is not None
        and (Path(str(record["evaluation_dir"])) / f"{method}_aggregate_predictions.npz").is_file()
    ]
    if not matching:
        raise FileNotFoundError(f"no main {method} correlation artifacts for {group}")
    return np.stack([_load_correlations(record, method) for record in matching]).mean(axis=0)


def _load_correlations(record: Mapping[str, Any], method: str) -> np.ndarray:
    path = Path(str(record["evaluation_dir"])) / f"{method}_aggregate_predictions.npz"
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["correlations"])


def _correlation_heatmaps(records: Sequence[dict[str, Any]], output: Path) -> None:
    for group in ("transformer", "station_installation"):
        record = _main_record(records, group)
        if record is None:
            continue
        library = load_pit_library(record["manifest"]["cache_path"], access="training")
        dataset = library.dataset
        train = np.asarray(dataset["split"].values) == "train"
        z = np.asarray(dataset["pit_z"].values)[train]
        complete = z[np.isfinite(z).all(axis=1).any(axis=1)]
        flattened = np.transpose(complete, (0, 2, 1)).reshape(-1, z.shape[1])
        flattened = flattened[np.isfinite(flattened).all(axis=1)]
        matrices = {
            "Empirical training PIT": np.corrcoef(flattened, rowvar=False),
            "M1 fitted": _mean_main_correlations(records, group, "static_gaussian").mean(axis=(0, 1)),
            "M2 test mean": _mean_main_correlations(records, group, "conditional_low_rank").mean(axis=(0, 1)),
            "M3 test mean": _mean_main_correlations(records, group, "set_aware_low_rank").mean(axis=(0, 1)),
        }
        figure, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
        image = None
        for axis, (title, matrix) in zip(axes, matrices.items(), strict=True):
            image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
            axis.set_title(title, fontsize=9)
            axis.set_xlabel("Entity index")
            axis.set_ylabel("Entity index")
        if image is not None:
            figure.colorbar(image, ax=axes, shrink=0.75)
        _save_figure(figure, output / f"figure_D_pit_correlations_{group}")


def _dependence_dynamics(records: Sequence[dict[str, Any]], output: Path) -> None:
    for group in ("transformer", "station_installation"):
        record = _main_record(records, group)
        if record is None:
            continue
        static = _mean_main_correlations(records, group, "static_gaussian")
        leads = sorted({0, static.shape[1] // 2, static.shape[1] - 1})
        figure, axes = plt.subplots(2, len(leads), figsize=(4 * len(leads), 6), squeeze=False)
        for method in ("conditional_low_rank", "set_aware_low_rank"):
            matrices = _mean_main_correlations(records, group, method)
            entity_count = matrices.shape[-1]
            mask = ~np.eye(entity_count, dtype=bool)
            for column, lead in enumerate(leads):
                mean_rho = matrices[:, lead][:, mask].mean(axis=1)
                distance = np.linalg.norm(matrices[:, lead] - static[:, lead], axis=(1, 2))
                axes[0, column].plot(mean_rho, label=METHOD_LABELS[method])
                axes[1, column].plot(distance, label=METHOD_LABELS[method])
                axes[0, column].set_title(f"Lead {lead + 1}")
                axes[1, column].set_xlabel("Test origin index")
        axes[0, 0].set_ylabel("Mean off-diagonal correlation")
        axes[1, 0].set_ylabel("Frobenius distance from M1")
        axes[0, -1].legend()
        _save_figure(figure, output / f"figure_E_dependence_dynamics_{group}")


def _fan_forecasts(main_case: pd.DataFrame, output: Path) -> None:
    group = str(main_case["group"].iloc[0])
    selected = main_case[main_case["group"] == group].copy()
    origin_values = selected.groupby("origin", as_index=False)[["observed_aggregate"]].mean()
    origin_values = origin_values.sort_values(by="origin")
    chosen = select_representative_origins(origin_values)
    quantiles = [str(column) for column in selected.columns if str(column).startswith("aggregate_q")]
    if not chosen or not quantiles:
        return
    methods = [
        method
        for method in ("independent", "static_gaussian", "conditional_low_rank")
        if method in set(selected["method"])
    ]
    figure, axes = plt.subplots(len(chosen), 1, figsize=(9, 2.8 * len(chosen)), squeeze=False)
    for axis, origin in zip(axes[:, 0], chosen, strict=True):
        frame = selected[selected["origin"] == origin]
        truth = frame.groupby("lead")["observed_aggregate"].mean().sort_index()
        axis.plot(truth.index, truth.values, color="black", linewidth=1.2, label="Observed")
        for method in methods:
            method_rows = frame[frame["method"] == method].groupby("lead")[quantiles].mean().sort_index()
            lower = method_rows.get("aggregate_q0.05", method_rows.iloc[:, 0])
            upper = method_rows.get("aggregate_q0.95", method_rows.iloc[:, -1])
            median = method_rows.get("aggregate_q0.5", method_rows.iloc[:, len(method_rows.columns) // 2])
            line = axis.plot(method_rows.index, median, label=METHOD_LABELS[method])[0]
            axis.fill_between(method_rows.index, lower, upper, color=line.get_color(), alpha=0.12)
        axis.set_title(str(origin))
        axis.set_xlabel("Lead")
        axis.set_ylabel("Aggregate")
    axes[0, 0].legend(ncol=4)
    _save_figure(figure, output / f"figure_G_representative_fans_{group}")


def _diagnostics(
    main_case: pd.DataFrame,
    main_origin: pd.DataFrame,
    records: Sequence[dict[str, Any]],
    output: Path,
) -> None:
    lead = main_case.groupby(["group", "method", "lead"], as_index=False)["mean_pinball"].mean()
    origin = main_origin.groupby(["group", "method", "origin"], as_index=False)["mean_pinball"].mean()
    lead.to_csv(output / "per_lead_mean_pinball.csv", index=False)
    origin.to_csv(output / "per_origin_mean_pinball.csv", index=False)
    for group in main_case["group"].drop_duplicates():
        record = _main_record(records, str(group))
        if record is None:
            continue
        rows: list[dict[str, Any]] = []
        for method in ("static_gaussian", "conditional_low_rank", "set_aware_low_rank"):
            try:
                matrices = _load_correlations(record, method)
            except FileNotFoundError:
                continue
            eigenvalues = np.linalg.eigvalsh(matrices)
            mask = ~np.eye(matrices.shape[-1], dtype=bool)
            rows.append(
                {
                    "method": method,
                    "mean_off_diagonal": float(matrices[..., mask].mean()),
                    "min_eigenvalue": float(eigenvalues.min()),
                    "median_condition_number": float(np.median(eigenvalues[..., -1] / eigenvalues[..., 0])),
                }
            )
        pd.DataFrame(rows).to_csv(output / f"correlation_numerics_{group}.csv", index=False)

        figure, axis = plt.subplots(figsize=(7, 3.5))
        for method in ("static_gaussian", "conditional_low_rank", "set_aware_low_rank"):
            try:
                matrices = _load_correlations(record, method)
            except FileNotFoundError:
                continue
            mask = ~np.eye(matrices.shape[-1], dtype=bool)
            axis.hist(matrices[..., mask].reshape(-1), bins=40, alpha=0.4, label=METHOD_LABELS[method])
        axis.set_xlabel("Predicted off-diagonal correlation")
        axis.set_ylabel("Count")
        axis.legend()
        _save_figure(figure, output / f"off_diagonal_correlations_{group}")

    width_columns = [str(column) for column in main_case.columns if str(column).startswith("interval_width_")]
    if width_columns:
        figure, axis = plt.subplots(figsize=(8, 4))
        coverage = str(width_columns[-1])
        labels: list[str] = []
        values: list[np.ndarray] = []
        for (raw_group, raw_method), group_rows in main_case[main_case["valid"]].groupby(
            ["group", "method"], sort=False
        ):
            group, method = str(raw_group), str(raw_method)
            labels.append(f"{group}:{METHOD_LABELS[method]}")
            values.append(group_rows[coverage].dropna().to_numpy())
        axis.boxplot(values, tick_labels=labels, showfliers=False)
        axis.tick_params(axis="x", rotation=75)
        axis.set_ylabel(coverage)
        _save_figure(figure, output / "aggregate_interval_width_distributions")


def _training_diagnostics(training: pd.DataFrame, output: Path) -> None:
    if training.empty:
        return
    neural = training[training["method"].isin(["conditional_low_rank", "set_aware_low_rank"])]
    if neural.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 4))
    grouped = list(neural.groupby(["group", "method"], sort=False))
    axis.boxplot(
        [rows["validation_pseudo_nll"].dropna().to_numpy() for _, rows in grouped],
        tick_labels=[
            f"{str(group)}:{METHOD_LABELS[str(method)]}" for (group, method), _ in grouped
        ],
    )
    axis.tick_params(axis="x", rotation=60)
    axis.set_ylabel("Selected validation pseudo-NLL")
    _save_figure(figure, output / "validation_pseudo_nll_across_seeds")

    figure, axis = plt.subplots(figsize=(8, 4))
    for row in neural.itertuples():
        path = Path(str(row.run_path)) / "training_metrics.csv"
        if not path.is_file():
            continue
        history = pd.read_csv(path)
        axis.plot(history["epoch"], history["validation_nll"], alpha=0.45, linewidth=0.8)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation pseudo-NLL")
    _save_figure(figure, output / "neural_training_curves_all_seeds")


def _solar_diagnostic(records: Sequence[dict[str, Any]], output: Path) -> None:
    record = next((item for item in records if item["manifest"]["group"]["name"] == "solar_park"), None)
    if record is None:
        return
    library = load_pit_library(record["manifest"]["cache_path"], access="training")
    pit = library.metadata["pit"]
    by_lead = pit.get("crossing_frequency_by_lead", [])
    pd.DataFrame(
        {
            "lead": np.arange(1, len(by_lead) + 1),
            "raw_crossing_rate": by_lead,
            "raw_crossing_rate_overall": pit["crossing_frequency"],
            "complete_cases_without_repair": pit.get("complete_case_count_without_repair", np.nan),
            "complete_cases_with_repair": pit.get("complete_case_count_with_configured_repair", np.nan),
        }
    ).to_csv(output / "solar_marginal_diagnostic.csv", index=False)


def _scientific_summary(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    characteristics: pd.DataFrame,
    ablation: pd.DataFrame,
    pit_sensitivity: pd.DataFrame,
    primary_block_length: int,
) -> str:
    lines = [
        "# Confirmatory scientific summary",
        "",
        "The fixed Chronos-2 marginal quantile grids are identical across dependence methods within each run.",
        "Negative paired score differences favour method A. Confidence intervals use forecast-origin moving blocks.",
        "",
        "## Dependence and independence",
        "",
    ]
    for characteristic in characteristics.itertuples():
        lines.append(
            f"- {characteristic.group}: mean absolute off-diagonal training PIT correlation is "
            f"{characteristic.mean_absolute_training_pit_correlation:.3f}."
        )
    independent = summary[summary["method"] == "independent"]
    if "coverage_0.9" in independent:
        for _, summary_row in independent.iterrows():
            lines.append(
                f"- {summary_row['group']}: M0 empirical 90% aggregate coverage is "
                f"{summary_row['coverage_0.9']:.3f}."
            )
    lines.extend(["", "## Paired method effects", ""])
    primary = effects[effects["block_length"] == primary_block_length]
    for _, effect_row in primary.iterrows():
        relation = "excludes" if effect_row["ci_lower"] > 0 or effect_row["ci_upper"] < 0 else "includes"
        method_a, method_b = str(effect_row["method_A"]), str(effect_row["method_B"])
        lines.append(
            f"- {effect_row['group']}: {METHOD_LABELS[method_a]} versus {METHOD_LABELS[method_b]} "
            f"has estimated aggregate-pinball change {effect_row['percentage_difference']:.2f}%; "
            f"the 95% interval [{effect_row['ci_lower']:.4g}, {effect_row['ci_upper']:.4g}] {relation} zero."
        )
    if primary.empty:
        lines.append("- No complete paired confirmatory comparison was available.")
    seed_variability = summary[summary["method"].isin(["conditional_low_rank", "set_aware_low_rank"])]
    if not seed_variability.empty:
        lines.append("")
        lines.append(
            "Neural-seed variability is reported in `results/method_seed_summary.csv`; no best-seed selection is used."
        )
    lines.extend(["", "## Feature and PIT sensitivities", ""])
    if ablation.empty:
        lines.append("- Feature-ablation experiments were not available in the supplied artifacts.")
    else:
        for ablation_row in ablation.itertuples():
            method = str(ablation_row.method)
            lines.append(
                f"- {ablation_row.group} {METHOD_LABELS[method]} {ablation_row.feature_set}: "
                f"aggregate-pinball change versus M1 is {ablation_row.percentage_change_vs_M1:.2f}%."
            )
    if pit_sensitivity.empty:
        lines.append("- PIT-frequency sensitivity experiments were not available in the supplied artifacts.")
    else:
        for (raw_group, raw_method), pit_rows in pit_sensitivity.groupby(["group", "method"], sort=False):
            group, method = str(raw_group), str(raw_method)
            values = pit_rows.set_index("pit_transform")["aggregate_pinball"]
            if {"nominal_cells", "training_frequency"}.issubset(values.index):
                change = values["training_frequency"] - values["nominal_cells"]
                lines.append(f"- {group} {METHOD_LABELS[method]} PIT-sensitivity pinball change is {change:.4g}.")
    lines.extend(
        [
            "",
            "The existing test period was inspected during exploratory development. The present protocol is therefore "
            "confirmatory from its declaration onward, not a claim of a previously untouched test set.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_powertech_report(
    input_root: str | Path,
    output_dir: str | Path = "reports/powertech2027",
) -> Path:
    """Build tidy results, tables, figures, and narrative from saved evaluations."""

    source = Path(input_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"report directory already exists: {output}")
    results_dir = output / "results"
    figures_dir = output / "figures"
    diagnostics_dir = output / "diagnostics"
    for directory in (results_dir, figures_dir, diagnostics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    per_case, per_origin, records = _artifact_frames(source)
    main_case = _deduplicate_deterministic(
        _main_rows(per_case), ["group", "method", "origin", "lead", "experiment_family"]
    )
    main_origin = _deduplicate_deterministic(
        _main_rows(per_origin), ["group", "method", "origin", "experiment_family"]
    )
    if main_case.empty or main_origin.empty:
        raise ValueError("no main full-feature, nominal-PIT, rank-4 confirmatory rows were found")

    training = _training_records(records)
    method_seed = _method_seed_summary(main_origin)
    if not training.empty:
        main_training = training[training["experiment_family"] == "main"]
        method_seed = method_seed.merge(
            main_training[
                ["group", "method", "neural_seed", "selected_epoch", "validation_pseudo_nll", "parameter_count"]
            ].drop_duplicates(["group", "method", "neural_seed"]),
            on=["group", "method", "neural_seed"],
            how="left",
        )
    group_summary = _group_method_summary(main_case, main_origin)
    first_confirmatory = records[0]["config"]["confirmatory"]
    block_lengths = [first_confirmatory["primary_block_length"], *first_confirmatory["sensitivity_block_lengths"]]
    effects = _paired_effects(
        main_origin,
        block_lengths=block_lengths,
        bootstrap_replicates=int(first_confirmatory["bootstrap_replicates"]),
        seed=int(records[0]["manifest"]["evaluation_seed"]),
    )
    main_results = _main_results_table(
        group_summary,
        effects,
        int(first_confirmatory["primary_block_length"]),
    )
    ablation = _ablation_deltas(
        _sensitivity_summary(per_origin, training, "ablation", ["feature_set"]),
        group_summary,
    )
    pit_sensitivity = _sensitivity_summary(per_origin, training, "pit_sensitivity", ["pit_transform"])
    rank_sensitivity = _sensitivity_summary(
        per_origin,
        training,
        "rank_sensitivity",
        ["m2_rank", "m3_rank"],
    )
    static_sensitivity = _sensitivity_summary(
        per_origin,
        training,
        "static_sensitivity",
        ["static_variant"],
    )

    main_case.to_parquet(results_dir / "per_origin_lead_metrics.parquet", index=False)
    main_origin.to_parquet(results_dir / "per_origin_metrics.parquet", index=False)
    method_seed.to_csv(results_dir / "method_seed_summary.csv", index=False)
    group_summary.to_csv(results_dir / "group_method_summary.csv", index=False)
    effects.to_csv(results_dir / "paired_effects.csv", index=False)
    ablation.to_csv(results_dir / "ablation_summary.csv", index=False)
    pit_sensitivity.to_csv(results_dir / "pit_sensitivity_summary.csv", index=False)
    rank_sensitivity.to_csv(results_dir / "rank_sensitivity_summary.csv", index=False)
    static_sensitivity.to_csv(results_dir / "static_sensitivity_summary.csv", index=False)

    characteristics = _cache_characteristics(records)
    characteristics.to_csv(results_dir / "data_dependence_characteristics.csv", index=False)
    _write_table(characteristics, output / "table_1_data_dependence.csv", output / "table_1_data_dependence.tex")
    _write_main_table(main_results, output / "main_results.csv", output / "main_results.tex")
    _write_table(effects, output / "paired_effects.csv", output / "paired_effects.tex")
    _write_table(ablation, output / "ablation_results.csv", output / "ablation_results.tex")

    _pipeline_figure(figures_dir)
    _forest_figure(effects, int(first_confirmatory["primary_block_length"]), figures_dir)
    _calibration_figure(group_summary, figures_dir)
    _correlation_heatmaps(records, figures_dir)
    _dependence_dynamics(records, figures_dir)
    _ablation_figure(ablation, figures_dir)
    _fan_forecasts(main_case, figures_dir)
    _pit_frequency_figure(records, figures_dir)
    _diagnostics(main_case, main_origin, records, diagnostics_dir)
    _training_diagnostics(training, diagnostics_dir)
    _solar_diagnostic(records, diagnostics_dir)

    groups_by_name = {
        str(record["manifest"]["group"]["name"]): record["manifest"]["group"] for record in records
    }
    protocol = {
        "source_root": str(source),
        "config": records[0]["config"],
        "groups": list(groups_by_name.values()),
        "git_commits": sorted(
            {str(value) for record in records if (value := record["manifest"].get("git_commit")) is not None}
        ),
        "config_hashes": sorted(
            {str(value) for record in records if (value := record["manifest"].get("config_sha256")) is not None}
        ),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "scientific_summary.md").write_text(
        _scientific_summary(
            group_summary,
            effects,
            characteristics,
            ablation,
            pit_sensitivity,
            int(first_confirmatory["primary_block_length"]),
        ),
        encoding="utf-8",
    )
    groups = ("transformer", "solar_park", "wind_park", "mv_feeder", "station_installation")
    phases = ("main", "ablation", "pit_sensitivity", "rank_sensitivity", "static_sensitivity")
    commands = [
        "uv run python -m simcast.cli.run_powertech_experiments "
        f"--config configs/powertech2027/{group}.yaml --phase {phase} "
        f"--output-dir runs/powertech2027/{group}/{phase}"
        + (" --rebuild-cache" if phase == "main" else "")
        for group in groups
        for phase in phases
    ]
    commands.append(
        "uv run python -m simcast.cli.build_powertech_report --input-root runs/powertech2027 "
        "--output-dir reports/powertech2027"
    )
    (output / "reproduction_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    shutil.copy2(results_dir / "paired_effects.csv", output / "paired_effects_tidy.csv")
    return output
