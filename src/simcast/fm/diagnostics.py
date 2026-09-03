"""Diagnostics for the fixed Chronos-2 marginal forecasts.

These summaries describe the entity-wise marginals only.  They do not alter,
recalibrate, or interpolate the native Chronos quantiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MarginalSummary:
    """Marginal diagnostics pooled over a selected set of forecast rows."""

    forecast_sample_count: int
    pit_sample_count: int
    median_mae: float | None
    pinball_loss: dict[str, float | None]
    interval_coverage: dict[str, float | None]
    pit_histogram: dict[str, list[float] | list[int]]
    pit_mean: float | None
    pit_variance: float | None
    lower_tail_fraction: float | None
    upper_tail_fraction: float | None


@dataclass(frozen=True, slots=True)
class MarginalDiagnostics:
    """Overall and entity-wise diagnostics for frozen FM marginals."""

    quantile_levels: list[float]
    interval_levels: list[float]
    overall: MarginalSummary
    by_entity: dict[str, MarginalSummary]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return asdict(self)


def _level_key(level: float) -> str:
    return f"{level:.8g}"


def _level_index(levels: np.ndarray, requested: float, label: str) -> int:
    matches = np.flatnonzero(np.isclose(levels, requested, rtol=0.0, atol=1.0e-6))
    if matches.size != 1:
        raise ValueError(
            f"{label} {requested:g} is not a native quantile level; marginal diagnostics do not interpolate quantiles"
        )
    return int(matches[0])


def _optional_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def _optional_variance(values: np.ndarray) -> float | None:
    return float(values.var()) if values.size else None


def _summary(
    truth: np.ndarray,
    predictions: np.ndarray,
    pit: np.ndarray,
    levels: np.ndarray,
    interval_levels: tuple[float, ...],
) -> MarginalSummary:
    median_idx = _level_index(levels, 0.5, "median")
    median = predictions[..., median_idx]
    median_mask = np.isfinite(truth) & np.isfinite(median)
    absolute_errors = np.abs(truth[median_mask] - median[median_mask])

    pinball: dict[str, float | None] = {}
    for quantile_idx, quantile in enumerate(levels):
        forecast = predictions[..., quantile_idx]
        valid = np.isfinite(truth) & np.isfinite(forecast)
        error = truth[valid] - forecast[valid]
        losses = np.maximum(quantile * error, (quantile - 1.0) * error)
        pinball[_level_key(float(quantile))] = _optional_mean(losses)

    coverage: dict[str, float | None] = {}
    for interval_level in interval_levels:
        lower_level = (1.0 - interval_level) / 2.0
        upper_level = 1.0 - lower_level
        lower = predictions[..., _level_index(levels, lower_level, "interval endpoint")]
        upper = predictions[..., _level_index(levels, upper_level, "interval endpoint")]
        valid = np.isfinite(truth) & np.isfinite(lower) & np.isfinite(upper)
        covered = (truth[valid] >= lower[valid]) & (truth[valid] <= upper[valid])
        coverage[_level_key(interval_level)] = _optional_mean(covered.astype(np.float64))

    pit_values = pit[np.isfinite(pit)].astype(np.float64, copy=False)
    probability_edges = np.concatenate(([0.0], levels.astype(np.float64), [1.0]))
    locations = 0.5 * (probability_edges[:-1] + probability_edges[1:])
    counts = np.asarray(
        [np.count_nonzero(np.isclose(pit_values, value, rtol=0.0, atol=1.0e-7)) for value in locations]
    )
    frequencies = counts / pit_values.size if pit_values.size else counts.astype(np.float64)
    lower_tail = float(counts[0] / pit_values.size) if pit_values.size else None
    upper_tail = float(counts[-1] / pit_values.size) if pit_values.size else None

    return MarginalSummary(
        forecast_sample_count=int(median_mask.sum()),
        pit_sample_count=int(pit_values.size),
        median_mae=_optional_mean(absolute_errors),
        pinball_loss=pinball,
        interval_coverage=coverage,
        pit_histogram={
            "locations": locations.tolist(),
            "counts": counts.astype(int).tolist(),
            "frequencies": frequencies.tolist(),
        },
        pit_mean=_optional_mean(pit_values),
        pit_variance=_optional_variance(pit_values),
        lower_tail_fraction=lower_tail,
        upper_tail_fraction=upper_tail,
    )


def compute_marginal_diagnostics(
    true_y: np.ndarray,
    quantile_predictions: np.ndarray,
    quantile_levels: np.ndarray | list[float],
    pit_u: np.ndarray,
    *,
    entity_ids: list[str] | tuple[str, ...] | None = None,
    interval_levels: tuple[float, ...] | list[float] = (0.5, 0.8, 0.9),
) -> MarginalDiagnostics:
    """Compute pooled and entity-wise frozen-marginal diagnostics.

    Parameters use cache layout: truth and PIT are ``[origin, entity, lead]``
    and predictions are ``[origin, entity, lead, quantile]``.  Missing or
    invalid rows are omitted metric-by-metric.  In particular, a PIT value
    dropped by complete-vector gating is not counted in the PIT summaries.
    """

    truth = np.asarray(true_y, dtype=np.float64)
    predictions = np.asarray(quantile_predictions, dtype=np.float64)
    pit = np.asarray(pit_u, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    intervals = tuple(float(value) for value in interval_levels)
    if truth.ndim != 3:
        raise ValueError("true_y must have shape [origin, entity, lead]")
    if predictions.ndim != 4 or predictions.shape[:-1] != truth.shape:
        raise ValueError("quantile_predictions must have shape [origin, entity, lead, quantile]")
    if pit.shape != truth.shape:
        raise ValueError("pit_u must have the same shape as true_y")
    if levels.ndim != 1 or levels.size != predictions.shape[-1] or np.any(np.diff(levels) <= 0):
        raise ValueError("quantile_levels must match the increasing prediction quantile axis")
    if np.any((levels <= 0.0) | (levels >= 1.0)):
        raise ValueError("quantile_levels must lie strictly between zero and one")
    if any(not 0.0 < value < 1.0 for value in intervals):
        raise ValueError("interval levels must lie strictly between zero and one")

    n_entity = truth.shape[1]
    ids = list(entity_ids) if entity_ids is not None else [str(index) for index in range(n_entity)]
    if len(ids) != n_entity or len(set(ids)) != n_entity:
        raise ValueError("entity_ids must be unique and match the entity dimension")

    overall = _summary(truth, predictions, pit, levels, intervals)
    by_entity = {
        entity_id: _summary(
            truth[:, entity_idx, :],
            predictions[:, entity_idx, :, :],
            pit[:, entity_idx, :],
            levels,
            intervals,
        )
        for entity_idx, entity_id in enumerate(ids)
    }
    return MarginalDiagnostics(
        quantile_levels=levels.tolist(),
        interval_levels=list(intervals),
        overall=overall,
        by_entity=by_entity,
    )


# Readable short alias for notebooks and analysis code.
marginal_diagnostics = compute_marginal_diagnostics


__all__ = [
    "MarginalDiagnostics",
    "MarginalSummary",
    "compute_marginal_diagnostics",
    "marginal_diagnostics",
]
