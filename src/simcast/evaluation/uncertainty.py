"""Paired moving-block uncertainty analysis of origin-level scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


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
