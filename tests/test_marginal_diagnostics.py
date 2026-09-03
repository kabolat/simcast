import numpy as np
import pytest

from simcast.fm.diagnostics import compute_marginal_diagnostics


def test_marginal_diagnostics_report_entity_and_overall_metrics() -> None:
    truth = np.asarray([[[2.0, 3.0]], [[4.0, 5.0]]])
    median = truth + np.asarray([[[0.0, 1.0]], [[-1.0, 0.0]]])
    predictions = np.stack((median - 2.0, median - 1.0, median, median + 1.0, median + 2.0), axis=-1)
    pit = np.asarray([[[0.05, 0.3]], [[0.7, 0.95]]])

    report = compute_marginal_diagnostics(
        truth,
        predictions,
        [0.1, 0.25, 0.5, 0.75, 0.9],
        pit,
        entity_ids=["transformer::a"],
        interval_levels=[0.5, 0.8],
    )

    assert report.overall.median_mae == pytest.approx(0.5)
    assert report.overall.pit_mean == pytest.approx(0.5)
    assert report.overall.lower_tail_fraction == pytest.approx(0.25)
    assert report.overall.upper_tail_fraction == pytest.approx(0.25)
    assert report.overall.interval_coverage["0.5"] == 1.0
    assert report.by_entity["transformer::a"] == report.overall


def test_marginal_diagnostics_never_interpolate_interval_endpoints() -> None:
    truth = np.ones((1, 1, 1))
    predictions = np.ones((1, 1, 1, 3))
    with pytest.raises(ValueError, match="do not interpolate"):
        compute_marginal_diagnostics(
            truth,
            predictions,
            [0.1, 0.5, 0.9],
            np.full_like(truth, 0.5),
            interval_levels=[0.5],
        )
