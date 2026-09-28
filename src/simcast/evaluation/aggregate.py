"""Aggregate-ensemble evaluation overall and by one-based forecast lead."""

from __future__ import annotations

from collections.abc import Collection
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
    case_metrics: dict[str, torch.Tensor]


def evaluate_aggregate_ensemble(
    aggregate_samples: torch.Tensor,
    true_aggregate: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    metrics: Collection[str],
    interval_coverages: tuple[float, ...] = (0.5, 0.8, 0.9),
    valid_mask: torch.Tensor | None = None,
) -> AggregateEvaluation:
    """Evaluate samples shaped ``[origin, lead, sample]`` against ``[origin, lead]``."""

    samples = torch.as_tensor(aggregate_samples)
    truth = torch.as_tensor(true_aggregate, dtype=samples.dtype, device=samples.device)
    levels = torch.as_tensor(quantile_levels, dtype=samples.dtype, device=samples.device)
    if samples.ndim != 3 or samples.shape[:-1] != truth.shape:
        raise ValueError("expected aggregate_samples [origin, lead, sample] and truth [origin, lead]")
    finite = torch.isfinite(samples).all(dim=-1) & torch.isfinite(truth)
    if valid_mask is None:
        valid = finite
    else:
        supplied = torch.as_tensor(valid_mask, dtype=torch.bool, device=samples.device)
        if supplied.shape != truth.shape:
            raise ValueError("valid_mask must match [origin, lead]")
        valid = supplied & finite
    if not bool(valid.any()):
        raise ValueError("aggregate evaluation has no complete finite cases")

    selected = set(metrics)
    quantiles = empirical_quantiles(samples, levels)
    overall: dict[str, float] = {}
    case_metrics: dict[str, torch.Tensor] = {}
    rows: list[dict[str, float | int]] = [{"lead": lead + 1} for lead in range(samples.shape[1])]
    if "mean_pinball" in selected:
        pinball = pinball_loss(truth, quantiles, levels, reduction="none")
        overall["mean_pinball"] = float(pinball[valid].mean())
        case_metrics["mean_pinball"] = pinball.mean(dim=-1)
        for index, level in enumerate(levels):
            suffix = f"pinball_q{float(level):g}"
            case_metrics[suffix] = pinball[..., index]
            overall[suffix] = float(pinball[..., index][valid].mean())
        for lead_index, row in enumerate(rows):
            lead_valid = valid[:, lead_index]
            if bool(lead_valid.any()):
                row["mean_pinball"] = float(pinball[:, lead_index][lead_valid].mean())
    if "crps" in selected:
        crps = crps_ensemble(samples, truth)
        overall["crps"] = float(crps[valid].mean())
        case_metrics["crps"] = crps
        for lead_index, row in enumerate(rows):
            lead_valid = valid[:, lead_index]
            if bool(lead_valid.any()):
                row["crps"] = float(crps[:, lead_index][lead_valid].mean())
    if "weighted_interval_score" in selected:
        median = empirical_quantiles(samples, samples.new_tensor([0.5]))[..., 0]
        coverages = samples.new_tensor(interval_coverages)
        interval_scores: list[torch.Tensor] = []
        lower_columns: list[torch.Tensor] = []
        upper_columns: list[torch.Tensor] = []
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
            interval_scores.append(score)
            coverage_arrays.append(covered)
            width_arrays.append(width)
            suffix = f"{coverage:g}"
            overall[f"coverage_{suffix}"] = float(covered[valid].float().mean())
            overall[f"interval_width_{suffix}"] = float(width[valid].mean())
            overall[f"interval_score_{suffix}"] = float(score[valid].mean())
            case_metrics[f"coverage_{suffix}"] = covered.to(samples.dtype)
            case_metrics[f"interval_width_{suffix}"] = width
            case_metrics[f"interval_score_{suffix}"] = score
        wis = weighted_interval_score(
            truth, median, torch.stack(lower_columns, dim=-1), torch.stack(upper_columns, dim=-1), coverages
        )
        overall["weighted_interval_score"] = float(wis[valid].mean())
        case_metrics["weighted_interval_score"] = wis
        for lead_index, row in enumerate(rows):
            lead_valid = valid[:, lead_index]
            if not bool(lead_valid.any()):
                continue
            row["weighted_interval_score"] = float(wis[:, lead_index][lead_valid].mean())
            for coverage, covered, width, score in zip(
                interval_coverages, coverage_arrays, width_arrays, interval_scores, strict=True
            ):
                suffix = f"{coverage:g}"
                row[f"coverage_{suffix}"] = float(covered[:, lead_index][lead_valid].float().mean())
                row[f"interval_width_{suffix}"] = float(width[:, lead_index][lead_valid].mean())
                row[f"interval_score_{suffix}"] = float(score[:, lead_index][lead_valid].mean())

    return AggregateEvaluation(
        overall=overall,
        by_lead=pd.DataFrame(rows).set_index("lead"),
        quantile_predictions=quantiles,
        case_metrics=case_metrics,
    )
