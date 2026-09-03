"""Proper scores and calibration summaries for aggregate and joint ensembles."""

from __future__ import annotations

from typing import Literal

import torch


def empirical_quantiles(samples: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Empirical nearest-order-statistic quantiles along the final axis."""

    draws = torch.as_tensor(samples)
    probabilities = torch.as_tensor(levels, dtype=draws.dtype, device=draws.device)
    if draws.ndim < 1 or draws.shape[-1] == 0:
        raise ValueError("samples must have a non-empty final ensemble axis")
    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("levels must be a non-empty vector")
    if not bool(((probabilities > 0) & (probabilities < 1)).all()):
        raise ValueError("levels must lie in (0, 1)")
    # torch.quantile puts quantiles first; move them to the final dimension.
    return torch.quantile(draws, probabilities, dim=-1, interpolation="nearest").movedim(0, -1)


def pinball_loss(
    observation: torch.Tensor,
    forecast_quantiles: torch.Tensor,
    levels: torch.Tensor,
    *,
    reduction: Literal["none", "mean"] = "mean",
) -> torch.Tensor:
    """Quantile (pinball) loss, retaining quantile as the final dimension."""

    forecasts = torch.as_tensor(forecast_quantiles)
    truth = torch.as_tensor(observation, dtype=forecasts.dtype, device=forecasts.device)
    probabilities = torch.as_tensor(levels, dtype=forecasts.dtype, device=forecasts.device)
    if forecasts.shape[:-1] != truth.shape or forecasts.shape[-1] != probabilities.numel():
        raise ValueError("forecast_quantiles must have shape observation.shape + [num_levels]")
    error = truth.unsqueeze(-1) - forecasts
    loss = torch.maximum(probabilities * error, (probabilities - 1.0) * error)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction: {reduction}")


def interval_score(
    observation: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    coverage: float,
) -> torch.Tensor:
    """Winkler interval score for a central interval with nominal coverage."""

    if not 0 < coverage < 1:
        raise ValueError("coverage must lie in (0, 1)")
    truth = torch.as_tensor(observation)
    lower_bound = torch.as_tensor(lower, dtype=truth.dtype, device=truth.device)
    upper_bound = torch.as_tensor(upper, dtype=truth.dtype, device=truth.device)
    if truth.shape != lower_bound.shape or truth.shape != upper_bound.shape:
        raise ValueError("observation and interval bounds must have equal shapes")
    if bool((lower_bound > upper_bound).any()):
        raise ValueError("lower interval bounds must not exceed upper bounds")
    alpha = 1.0 - coverage
    below = (truth < lower_bound) * (lower_bound - truth)
    above = (truth > upper_bound) * (truth - upper_bound)
    return upper_bound - lower_bound + (2.0 / alpha) * (below + above)


def weighted_interval_score(
    observation: torch.Tensor,
    median: torch.Tensor,
    lowers: torch.Tensor,
    uppers: torch.Tensor,
    coverages: torch.Tensor,
) -> torch.Tensor:
    """Weighted interval score over central intervals, preserving case axes."""

    truth = torch.as_tensor(observation)
    med = torch.as_tensor(median, dtype=truth.dtype, device=truth.device)
    lower = torch.as_tensor(lowers, dtype=truth.dtype, device=truth.device)
    upper = torch.as_tensor(uppers, dtype=truth.dtype, device=truth.device)
    levels = torch.as_tensor(coverages, dtype=truth.dtype, device=truth.device)
    if med.shape != truth.shape or lower.shape[:-1] != truth.shape or upper.shape != lower.shape:
        raise ValueError("interval tensors must have shape observation.shape + [num_coverages]")
    if lower.shape[-1] != levels.numel():
        raise ValueError("interval count does not match coverages")
    alphas = 1.0 - levels
    scores = torch.stack(
        [interval_score(truth, lower[..., idx], upper[..., idx], float(levels[idx])) for idx in range(levels.numel())],
        dim=-1,
    )
    weights = alphas / 2.0
    numerator = 0.5 * torch.abs(truth - med) + (scores * weights).sum(dim=-1)
    return numerator / (0.5 + weights.sum())


def crps_ensemble(samples: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
    """Empirical CRPS in ``O(M log M)`` along the final sample dimension."""

    draws = torch.as_tensor(samples)
    truth = torch.as_tensor(observation, dtype=draws.dtype, device=draws.device)
    if draws.shape[:-1] != truth.shape or draws.shape[-1] == 0:
        raise ValueError("samples must have shape observation.shape + [num_samples]")
    sorted_draws = torch.sort(draws, dim=-1).values
    sample_count = draws.shape[-1]
    ranks = torch.arange(1, sample_count + 1, dtype=draws.dtype, device=draws.device)
    coefficients = 2.0 * ranks - sample_count - 1.0
    pair_term = (sorted_draws * coefficients).sum(dim=-1) / (sample_count * sample_count)
    return torch.abs(draws - truth.unsqueeze(-1)).mean(dim=-1) - pair_term


def energy_score(
    samples: torch.Tensor,
    observation: torch.Tensor,
    *,
    beta: float = 1.0,
    pair_chunk_size: int = 512,
) -> torch.Tensor:
    """Energy score for one multivariate ensemble ``[M, K]``."""

    draws = torch.as_tensor(samples)
    truth = torch.as_tensor(observation, dtype=draws.dtype, device=draws.device)
    if draws.ndim != 2 or truth.shape != (draws.shape[1],) or draws.shape[0] == 0:
        raise ValueError("expected samples [num_samples, entities] and observation [entities]")
    if not 0 < beta <= 2 or pair_chunk_size <= 0:
        raise ValueError("beta must be in (0, 2] and pair_chunk_size must be positive")
    first = torch.linalg.vector_norm(draws - truth, dim=-1).pow(beta).mean()
    pair_sum = draws.new_zeros(())
    for start in range(0, draws.shape[0], pair_chunk_size):
        distances = torch.cdist(draws[start : start + pair_chunk_size], draws, p=2).pow(beta)
        pair_sum = pair_sum + distances.sum()
    second = pair_sum / (2.0 * draws.shape[0] ** 2)
    return torch.as_tensor(first - second)


def variogram_score(
    samples: torch.Tensor,
    observation: torch.Tensor,
    *,
    power: float = 0.5,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variogram score summed over unique entity pairs for one ensemble."""

    draws = torch.as_tensor(samples)
    truth = torch.as_tensor(observation, dtype=draws.dtype, device=draws.device)
    if draws.ndim != 2 or truth.shape != (draws.shape[1],) or draws.shape[1] < 2:
        raise ValueError("expected samples [num_samples, K] and observation [K], with K >= 2")
    if not 0 < power <= 2:
        raise ValueError("power must lie in (0, 2]")
    entities = draws.shape[1]
    if weights is None:
        pair_weights = torch.ones((entities, entities), dtype=draws.dtype, device=draws.device)
    else:
        pair_weights = torch.as_tensor(weights, dtype=draws.dtype, device=draws.device)
        if pair_weights.shape != (entities, entities):
            raise ValueError("weights must have shape [K, K]")
    row, column = torch.triu_indices(entities, entities, offset=1, device=draws.device)
    observed = torch.abs(truth[row] - truth[column]).pow(power)
    predicted = torch.abs(draws[:, row] - draws[:, column]).pow(power).mean(dim=0)
    return (pair_weights[row, column] * (observed - predicted).square()).sum()
