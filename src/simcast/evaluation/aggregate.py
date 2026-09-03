"""Aggregate-ensemble evaluation overall and by one-based forecast lead."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from simcast.evaluation.metrics import (
    crps_ensemble,
    empirical_quantiles,
    interval_score,
    pinball_loss,
    weighted_interval_score,
)


@dataclass(frozen=True, slots=True)
class AggregateEvaluation:
    """Summary tables and empirical aggregate quantile forecasts."""

    overall: dict[str, float]
    by_lead: pd.DataFrame
    quantile_predictions: torch.Tensor


def sum_marginal_quantiles(quantile_predictions: torch.Tensor) -> torch.Tensor:
    """Naive comonotonic-style baseline: sum equally labelled marginal quantiles.

    This is deliberately *not* presented as a generally valid aggregate
    quantile. It is retained only as an intuitive strong-dependence baseline.
    """

    predictions = torch.as_tensor(quantile_predictions)
    if predictions.ndim < 2:
        raise ValueError("quantile_predictions must have shape [..., entity, quantile]")
    return predictions.sum(dim=-2)


def evaluate_aggregate_ensemble(
    aggregate_samples: torch.Tensor,
    true_aggregate: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    interval_coverages: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> AggregateEvaluation:
    """Evaluate samples shaped ``[origin, lead, sample]`` against ``[origin, lead]``."""

    samples = torch.as_tensor(aggregate_samples)
    truth = torch.as_tensor(true_aggregate, dtype=samples.dtype, device=samples.device)
    levels = torch.as_tensor(quantile_levels, dtype=samples.dtype, device=samples.device)
    if samples.ndim != 3 or samples.shape[:-1] != truth.shape:
        raise ValueError("expected aggregate_samples [origin, lead, sample] and truth [origin, lead]")
    if not bool(torch.isfinite(samples).all()) or not bool(torch.isfinite(truth).all()):
        raise ValueError("aggregate evaluation requires complete finite cases")

    quantiles = empirical_quantiles(samples, levels)
    pinball = pinball_loss(truth, quantiles, levels, reduction="none")
    crps = crps_ensemble(samples, truth)
    median = empirical_quantiles(samples, samples.new_tensor([0.5]))[..., 0]
    overall: dict[str, float] = {
        "mean_pinball": float(pinball.mean()),
        "crps": float(crps.mean()),
    }
    for index, level in enumerate(levels):
        overall[f"pinball_q{float(level):g}"] = float(pinball[..., index].mean())

    lower_columns: list[torch.Tensor] = []
    upper_columns: list[torch.Tensor] = []
    coverages = samples.new_tensor(interval_coverages)
    interval_scores: list[torch.Tensor] = []
    coverage_arrays: list[torch.Tensor] = []
    width_arrays: list[torch.Tensor] = []
    for coverage in interval_coverages:
        alpha = 1.0 - coverage
        bounds = empirical_quantiles(samples, samples.new_tensor([alpha / 2.0, 1.0 - alpha / 2.0]))
        lower, upper = bounds[..., 0], bounds[..., 1]
        covered = (truth >= lower) & (truth <= upper)
        width = upper - lower
        score = interval_score(truth, lower, upper, coverage)
        lower_columns.append(lower)
        upper_columns.append(upper)
        coverage_arrays.append(covered)
        width_arrays.append(width)
        interval_scores.append(score)
        suffix = f"{coverage:g}"
        overall[f"coverage_{suffix}"] = float(covered.float().mean())
        overall[f"interval_width_{suffix}"] = float(width.mean())
        overall[f"interval_score_{suffix}"] = float(score.mean())
    lowers = torch.stack(lower_columns, dim=-1)
    uppers = torch.stack(upper_columns, dim=-1)
    wis = weighted_interval_score(truth, median, lowers, uppers, coverages)
    overall["weighted_interval_score"] = float(wis.mean())

    rows: list[dict[str, float | int]] = []
    for lead_index in range(samples.shape[1]):
        row: dict[str, float | int] = {
            "lead": lead_index + 1,
            "mean_pinball": float(pinball[:, lead_index].mean()),
            "crps": float(crps[:, lead_index].mean()),
            "weighted_interval_score": float(wis[:, lead_index].mean()),
        }
        for coverage, covered, width, score in zip(
            interval_coverages, coverage_arrays, width_arrays, interval_scores, strict=True
        ):
            suffix = f"{coverage:g}"
            row[f"coverage_{suffix}"] = float(covered[:, lead_index].float().mean())
            row[f"interval_width_{suffix}"] = float(width[:, lead_index].mean())
            row[f"interval_score_{suffix}"] = float(score[:, lead_index].mean())
        rows.append(row)
    return AggregateEvaluation(
        overall=overall,
        by_lead=pd.DataFrame(rows).set_index("lead"),
        quantile_predictions=quantiles,
    )
