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


def test_variogram_score_sums_over_all_ordered_pairs() -> None:
    torch.manual_seed(3)
    samples = torch.randn(7, 4)
    truth = torch.randn(4)
    observed = (truth[:, None] - truth[None, :]).abs().sqrt()
    predicted = (samples[:, :, None] - samples[:, None, :]).abs().sqrt().mean(dim=0)
    expected = (observed - predicted).square().sum()
    torch.testing.assert_close(variogram_score(samples, truth, power=0.5), expected)


def test_weighted_interval_score_is_twice_mean_pinball_over_implied_levels() -> None:
    torch.manual_seed(5)
    samples = torch.randn(2, 3, 401)
    truth = torch.randn(2, 3)
    coverages = (0.5, 0.8, 0.9)
    implied = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    result = evaluate_aggregate_ensemble(
        samples,
        truth,
        implied,
        metrics=("mean_pinball", "weighted_interval_score"),
        interval_coverages=coverages,
    )
    torch.testing.assert_close(
        result.case_metrics["weighted_interval_score"], 2.0 * result.case_metrics["mean_pinball"]
    )


def test_energy_and_variogram_scores_support_batched_ensembles() -> None:
    torch.manual_seed(17)
    samples = torch.randn(3, 11, 4)
    truth = torch.randn(3, 4)
    expected_energy = torch.stack(
        [energy_score(sample, observation) for sample, observation in zip(samples, truth, strict=True)]
    )
    expected_variogram = torch.stack(
        [variogram_score(sample, observation, power=0.5) for sample, observation in zip(samples, truth, strict=True)]
    )
    torch.testing.assert_close(energy_score(samples, truth), expected_energy, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(variogram_score(samples, truth, power=0.5), expected_variogram, atol=1e-6, rtol=1e-6)


def test_aggregate_evaluation_reports_overall_and_one_based_leads() -> None:
    samples = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]],
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        ]
    )
    truth = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    result = evaluate_aggregate_ensemble(
        samples,
        truth,
        torch.tensor([0.1, 0.5, 0.9]),
        metrics=("mean_pinball", "crps", "weighted_interval_score"),
    )
    assert result.quantile_predictions.shape == (2, 2, 3)
    assert list(result.by_lead.index) == [1, 2]
    assert result.overall["crps"] >= 0
    assert result.overall["coverage_0.9"] == 1.0
