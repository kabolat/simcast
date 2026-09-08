"""Origin-level aggregation and paired moving-block uncertainty analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PairedEffect:
    mean_difference: float
    percentage_difference: float
    ci_lower: float
    ci_upper: float
    percentage_ci_lower: float
    percentage_ci_upper: float
    block_length: int
    bootstrap_replicates: int
    number_of_origins: int


def moving_block_indices(
    number_of_origins: int,
    block_length: int,
    bootstrap_replicates: int,
    *,
    seed: int,
) -> np.ndarray:
    """Draw non-circular moving blocks and truncate each replicate to ``N``."""

    if min(number_of_origins, block_length, bootstrap_replicates) <= 0:
        raise ValueError("origin count, block length, and replicate count must be positive")
    if block_length > number_of_origins:
        raise ValueError("block length cannot exceed the number of origins")
    generator = np.random.default_rng(seed)
    blocks_per_replicate = int(np.ceil(number_of_origins / block_length))
    starts = generator.integers(
        0,
        number_of_origins - block_length + 1,
        size=(bootstrap_replicates, blocks_per_replicate),
    )
    offsets = np.arange(block_length)
    return (starts[..., None] + offsets).reshape(bootstrap_replicates, -1)[:, :number_of_origins]


def paired_moving_block_bootstrap(
    scores_a: Sequence[float] | np.ndarray,
    scores_b: Sequence[float] | np.ndarray,
    *,
    block_length: int = 7,
    bootstrap_replicates: int = 10_000,
    seed: int = 2027,
) -> PairedEffect:
    """Bootstrap chronological paired origin scores; negative favours method A."""

    left = np.asarray(scores_a, dtype=np.float64)
    right = np.asarray(scores_b, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("paired score vectors must be one-dimensional with equal shape")
    finite = np.isfinite(left) & np.isfinite(right)
    difference = left[finite] - right[finite]
    denominator = right[finite]
    if difference.size < block_length:
        raise ValueError("too few paired finite origins for the requested block length")
    indices = moving_block_indices(
        difference.size,
        block_length,
        bootstrap_replicates,
        seed=seed,
    )
    bootstrap_means = difference[indices].mean(axis=1)
    baseline_mean = denominator.mean()
    percentage = np.nan if baseline_mean == 0 else 100.0 * difference.mean() / baseline_mean
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return PairedEffect(
        mean_difference=float(difference.mean()),
        percentage_difference=float(percentage),
        ci_lower=float(lower),
        ci_upper=float(upper),
        percentage_ci_lower=float(np.nan if baseline_mean == 0 else 100.0 * lower / baseline_mean),
        percentage_ci_upper=float(np.nan if baseline_mean == 0 else 100.0 * upper / baseline_mean),
        block_length=block_length,
        bootstrap_replicates=bootstrap_replicates,
        number_of_origins=int(difference.size),
    )


def mean_neural_seeds_per_origin(per_origin: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average neural repetitions within each method/origin before temporal inference."""

    required = {"group", "method", "origin", metric}
    if missing := required - set(per_origin.columns):
        raise ValueError(f"per-origin table is missing columns: {sorted(missing)}")
    result = per_origin.groupby(["group", "method", "origin"], as_index=False, sort=False)[[metric]].mean()
    return result.sort_values(by=["group", "method", "origin"])


def seed_summary(per_origin: pd.DataFrame, metric: str = "mean_pinball") -> pd.DataFrame:
    """Summarize each method/seed after averaging its chronological origins."""

    required = {"group", "method", "neural_seed", metric}
    if missing := required - set(per_origin.columns):
        raise ValueError(f"per-origin table is missing columns: {sorted(missing)}")
    result = per_origin.groupby(["group", "method", "neural_seed"], dropna=False, as_index=False)[[metric]].mean()
    return result.rename(columns={metric: "primary_metric"})


def select_representative_origins(origin_table: pd.DataFrame) -> list[pd.Timestamp]:
    """Choose examples from timestamps and observed aggregates, never method error."""

    required = {"origin", "observed_aggregate"}
    if missing := required - set(origin_table.columns):
        raise ValueError(f"origin table is missing columns: {sorted(missing)}")
    frame = origin_table.dropna(subset=["origin", "observed_aggregate"]).copy()
    if frame.empty:
        return []
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame = frame.sort_values("origin").drop_duplicates("origin")
    winter = frame[frame["origin"].dt.month.isin([12, 1, 2])]
    summer = frame[frame["origin"].dt.month.isin([6, 7, 8])]
    median_value = frame["observed_aggregate"].median()
    median_row = frame.loc[(frame["observed_aggregate"] - median_value).abs().idxmin(), "origin"]
    high_row = frame.loc[frame["observed_aggregate"].idxmax(), "origin"]
    candidates = [
        winter.iloc[0]["origin"] if not winter.empty else frame.iloc[0]["origin"],
        summer.iloc[0]["origin"] if not summer.empty else frame.iloc[0]["origin"],
        median_row,
        high_row,
    ]
    return list(dict.fromkeys(pd.Timestamp(value) for value in candidates))
