from __future__ import annotations

import numpy as np
import pytest

from simcast.evaluation.uncertainty import moving_block_indices, paired_moving_block_bootstrap


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
