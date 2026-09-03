import pytest
import torch

from simcast.evaluation.aggregate import evaluate_aggregate_ensemble
from simcast.evaluation.metrics import (
    crps_ensemble,
    empirical_quantiles,
    energy_score,
    interval_score,
    pinball_loss,
    variogram_score,
)


def test_empirical_quantiles_use_nearest_order_statistic() -> None:
    samples = torch.tensor([[0.0, 10.0, 20.0, 30.0]])
    actual = empirical_quantiles(samples, torch.tensor([0.1, 0.5, 0.9]))
    torch.testing.assert_close(actual, torch.tensor([[0.0, 20.0, 30.0]]))


def test_pinball_and_interval_score_known_values() -> None:
    truth = torch.tensor([2.0])
    forecasts = torch.tensor([[1.0, 3.0]])
    losses = pinball_loss(truth, forecasts, torch.tensor([0.25, 0.75]), reduction="none")
    torch.testing.assert_close(losses, torch.tensor([[0.25, 0.25]]))
    assert interval_score(truth, torch.tensor([1.0]), torch.tensor([3.0]), 0.8).item() == pytest.approx(2.0)
    outside_score = interval_score(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([3.0]), 0.8)
    assert outside_score.item() == pytest.approx(12.0)


def test_crps_for_deterministic_ensemble_is_absolute_error() -> None:
    samples = torch.tensor([[3.0, 3.0, 3.0]])
    assert crps_ensemble(samples, torch.tensor([1.0])).item() == pytest.approx(2.0)


def test_energy_and_variogram_scores_are_zero_for_perfect_ensemble() -> None:
    truth = torch.tensor([1.0, 4.0, 9.0])
    samples = truth.expand(5, -1)
    assert energy_score(samples, truth).item() == pytest.approx(0.0)
    assert variogram_score(samples, truth).item() == pytest.approx(0.0)


def test_aggregate_evaluation_reports_overall_and_one_based_leads() -> None:
    samples = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]],
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        ]
    )
    truth = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    result = evaluate_aggregate_ensemble(samples, truth, torch.tensor([0.1, 0.5, 0.9]))
    assert result.quantile_predictions.shape == (2, 2, 3)
    assert list(result.by_lead.index) == [1, 2]
    assert result.overall["crps"] >= 0
    assert result.overall["coverage_0.9"] == 1.0
