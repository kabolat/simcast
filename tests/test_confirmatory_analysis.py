from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simcast.evaluation.uncertainty import (
    mean_neural_seeds_per_origin,
    moving_block_indices,
    paired_moving_block_bootstrap,
    seed_summary,
    select_representative_origins,
)


def test_moving_blocks_are_chronological_and_reproducible() -> None:
    first = moving_block_indices(20, 7, 50, seed=42)
    second = moving_block_indices(20, 7, 50, seed=42)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (50, 20)
    for row in first:
        for start in range(0, 14, 7):
            np.testing.assert_array_equal(np.diff(row[start : start + 7]), np.ones(6))


def test_paired_difference_sign_and_temporal_ci() -> None:
    baseline = np.arange(1.0, 31.0)
    improved = baseline - 2.0
    effect = paired_moving_block_bootstrap(improved, baseline, block_length=7, bootstrap_replicates=500, seed=4)
    assert effect.mean_difference == pytest.approx(-2.0)
    assert effect.ci_lower == pytest.approx(-2.0)
    assert effect.ci_upper == pytest.approx(-2.0)
    assert effect.percentage_difference < 0


def test_seed_aggregation_preserves_individual_seeds_then_averages_by_origin() -> None:
    frame = pd.DataFrame(
        {
            "group": ["g"] * 4,
            "method": ["conditional_low_rank"] * 4,
            "neural_seed": [11, 23, 11, 23],
            "origin": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"], utc=True),
            "mean_pinball": [1.0, 3.0, 2.0, 4.0],
        }
    )
    summary = seed_summary(frame)
    assert sorted(summary["primary_metric"]) == [1.5, 3.5]
    averaged = mean_neural_seeds_per_origin(frame, "mean_pinball")
    assert averaged["mean_pinball"].tolist() == [2.0, 3.0]


def test_representative_origins_depend_only_on_timestamp_and_observed_aggregate() -> None:
    frame = pd.DataFrame(
        {
            "origin": pd.to_datetime(["2024-01-02", "2024-06-01", "2024-07-01", "2024-12-01"], utc=True),
            "observed_aggregate": [5.0, 10.0, 20.0, 6.0],
        }
    )
    selected = select_representative_origins(frame)
    assert selected[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert pd.Timestamp("2024-06-01", tz="UTC") in selected
    assert pd.Timestamp("2024-07-01", tz="UTC") in selected
