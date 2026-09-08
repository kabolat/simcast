"""Discretized PIT pseudo-observations for native quantile forecasts.

The routines in this module never interpolate the Chronos marginal CDF.  A
realization is assigned to one of the ``Q + 1`` cells induced by the native
quantile predictions and mapped to that cell's probability midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.isotonic import isotonic_regression  # type: ignore[import-untyped]

MonotoneRepair = Literal["none", "isotonic"]


@dataclass(frozen=True)
class CrossingDiagnostics:
    """Quantile-crossing rates for predictions shaped ``[origin, entity, lead, quantile]``."""

    overall: float
    by_entity: np.ndarray
    by_lead: np.ndarray
    crossed: np.ndarray


@dataclass(frozen=True)
class PITResult:
    """PIT arrays and validity of complete spatial vectors.

    ``valid_origin_lead[i, tau]`` is false if any entity has missing truth,
    non-finite forecasts, or a quantile crossing. Invalid ``(origin, lead)``
    vectors are all NaN rather than silently changing group membership.
    """

    u: torch.Tensor
    z: torch.Tensor
    valid_origin_lead: torch.Tensor
    crossing_diagnostics: CrossingDiagnostics


def _validate_levels(quantile_levels: torch.Tensor) -> torch.Tensor:
    levels = torch.as_tensor(quantile_levels)
    if levels.ndim != 1 or levels.numel() == 0:
        raise ValueError("quantile_levels must be a non-empty one-dimensional tensor")
    if not torch.is_floating_point(levels):
        levels = levels.to(torch.float32)
    if not bool(torch.all((levels > 0) & (levels < 1))):
        raise ValueError("quantile levels must lie strictly between zero and one")
    if levels.numel() > 1 and not bool(torch.all(levels[1:] > levels[:-1])):
        raise ValueError("quantile levels must be strictly increasing")
    return levels


def quantile_crossings(quantile_predictions: torch.Tensor) -> torch.Tensor:
    """Return a mask over forecast rows with at least one adjacent crossing."""

    predictions = torch.as_tensor(quantile_predictions)
    if predictions.ndim < 1 or predictions.shape[-1] < 1:
        raise ValueError("quantile_predictions must have a non-empty quantile axis")
    if predictions.shape[-1] == 1:
        return torch.zeros(predictions.shape[:-1], dtype=torch.bool, device=predictions.device)
    return (predictions[..., :-1] > predictions[..., 1:]).any(dim=-1)


def crossing_diagnostics(quantile_predictions: torch.Tensor) -> CrossingDiagnostics:
    """Summarize crossings for ``[origin, entity, lead, quantile]`` forecasts."""

    predictions = torch.as_tensor(quantile_predictions)
    if predictions.ndim != 4:
        raise ValueError("expected predictions shaped [origin, entity, lead, quantile]")
    crossed = quantile_crossings(predictions).detach().cpu().numpy()
    return CrossingDiagnostics(
        overall=float(crossed.mean()) if crossed.size else 0.0,
        by_entity=crossed.mean(axis=(0, 2)),
        by_lead=crossed.mean(axis=(0, 1)),
        crossed=crossed,
    )


def repair_quantiles_isotonic(quantile_predictions: torch.Tensor) -> torch.Tensor:
    """Least-squares isotonic repair along the final quantile dimension."""

    predictions = torch.as_tensor(quantile_predictions)
    original_shape = predictions.shape
    if predictions.ndim < 1 or original_shape[-1] == 0:
        raise ValueError("quantile_predictions must have a non-empty quantile axis")
    values = predictions.detach().to(torch.float64).cpu().numpy().reshape(-1, original_shape[-1])
    repaired = np.empty_like(values)
    for row_idx, row in enumerate(values):
        if np.isfinite(row).all():
            repaired[row_idx] = isotonic_regression(row, increasing=True)
        else:
            repaired[row_idx] = row
    return torch.as_tensor(repaired.reshape(original_shape), device=predictions.device, dtype=predictions.dtype)


def discretized_pit(
    true_y: torch.Tensor,
    quantile_predictions: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    monotone_repair: MonotoneRepair = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute deterministic discretized PIT pseudo-observations.

    Parameters
    ----------
    true_y
        Realizations shaped ``[...]``.
    quantile_predictions
        Native quantile values shaped ``[..., Q]``.
    quantile_levels
        Strictly increasing native levels shaped ``[Q]``.
    monotone_repair
        ``"none"`` marks crossed rows invalid; ``"isotonic"`` first applies
        least-squares monotonic repair.

    Returns
    -------
    u, valid
        PIT locations and a row-validity mask, both shaped ``[...]``.
    """

    predictions = torch.as_tensor(quantile_predictions)
    truth = torch.as_tensor(true_y, device=predictions.device, dtype=predictions.dtype)
    levels = _validate_levels(torch.as_tensor(quantile_levels, device=predictions.device)).to(predictions.dtype)
    if predictions.shape[:-1] != truth.shape:
        raise ValueError("true_y shape must equal quantile_predictions shape without the quantile axis")
    if predictions.shape[-1] != levels.numel():
        raise ValueError("quantile axis does not match quantile_levels")
    if monotone_repair == "isotonic":
        predictions = repair_quantiles_isotonic(predictions)
    elif monotone_repair != "none":
        raise ValueError(f"unknown monotone repair mode: {monotone_repair}")

    finite = torch.isfinite(truth) & torch.isfinite(predictions).all(dim=-1)
    valid = finite & ~quantile_crossings(predictions)
    # searchsorted returns the first quantile prediction >= y (left side).
    flat_predictions = predictions.reshape(-1, predictions.shape[-1]).contiguous()
    flat_truth = truth.reshape(-1, 1).contiguous()
    cell = torch.searchsorted(flat_predictions, flat_truth, right=False).squeeze(-1)

    edges = torch.cat((levels.new_tensor([0.0]), levels, levels.new_tensor([1.0])))
    locations = 0.5 * (edges[:-1] + edges[1:])
    u = locations[cell.clamp_max(levels.numel())].reshape(truth.shape)
    u = torch.where(valid, u, torch.full_like(u, torch.nan))
    return u, valid


def gaussianize_pit(u: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    """Map PIT locations to standard-normal pseudo-scores with ``Phi^-1``."""

    if not 0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    values = torch.as_tensor(u)
    return torch.as_tensor(torch.special.ndtri(values.clamp(min=eps, max=1.0 - eps)))


def nominal_cell_midpoints(quantile_levels: torch.Tensor) -> torch.Tensor:
    """Return the deterministic ``Q + 1`` nominal PIT-cell midpoints."""

    levels = _validate_levels(quantile_levels)
    edges = torch.cat((levels.new_tensor([0.0]), levels, levels.new_tensor([1.0])))
    return 0.5 * (edges[:-1] + edges[1:])


def fit_training_frequency_midpoints(
    training_u: torch.Tensor,
    quantile_levels: torch.Tensor,
) -> torch.Tensor:
    """Fit empirical PIT-cell midpoints using training origins only.

    ``training_u`` has shape ``[origin, entity, lead]``.  The returned table
    has shape ``[entity, lead, Q + 1]`` and contains cumulative empirical-mass
    midpoints. Missing complete vectors are ignored; each entity/lead must
    retain at least one training observation.
    """

    values = torch.as_tensor(training_u)
    if values.ndim != 3:
        raise ValueError("training_u must have shape [origin, entity, lead]")
    nominal = nominal_cell_midpoints(torch.as_tensor(quantile_levels, device=values.device)).to(values.dtype)
    distances = torch.abs(values.unsqueeze(-1) - nominal)
    cells = distances.nan_to_num(float("inf")).argmin(dim=-1)
    finite = torch.isfinite(values)
    counts = torch.stack(
        [((cells == cell) & finite).sum(dim=0) for cell in range(nominal.numel())],
        dim=-1,
    ).to(torch.float64)
    totals = counts.sum(dim=-1, keepdim=True)
    if bool((totals == 0).any()):
        raise ValueError("every entity/lead needs at least one finite training PIT cell")
    probabilities = counts / totals
    return (probabilities.cumsum(dim=-1) - 0.5 * probabilities).to(values.dtype)


def apply_training_frequency_midpoints(
    u: torch.Tensor,
    quantile_levels: torch.Tensor,
    empirical_midpoints: torch.Tensor,
) -> torch.Tensor:
    """Map nominal finite-cell PITs through a frozen training-frequency table."""

    values = torch.as_tensor(u)
    if values.ndim != 3:
        raise ValueError("u must have shape [origin, entity, lead]")
    nominal = nominal_cell_midpoints(torch.as_tensor(quantile_levels, device=values.device)).to(values.dtype)
    mapping = torch.as_tensor(empirical_midpoints, device=values.device, dtype=values.dtype)
    expected = (*values.shape[1:], nominal.numel())
    if mapping.shape != expected:
        raise ValueError(f"empirical_midpoints must have shape {expected}")
    cells = torch.abs(values.unsqueeze(-1) - nominal).nan_to_num(float("inf")).argmin(dim=-1)
    expanded = mapping.unsqueeze(0).expand(values.shape[0], -1, -1, -1)
    calibrated = expanded.gather(dim=-1, index=cells.unsqueeze(-1)).squeeze(-1)
    return torch.where(torch.isfinite(values), calibrated, torch.full_like(calibrated, torch.nan))


def dependence_pit_scores(
    pit_u: torch.Tensor,
    nominal_z: torch.Tensor,
    split_is_training: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    mode: Literal["nominal_cells", "training_frequency"] = "nominal_cells",
    eps: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return dependence scores and the optional train-only calibration map."""

    u = torch.as_tensor(pit_u)
    z = torch.as_tensor(nominal_z, device=u.device, dtype=u.dtype)
    training = torch.as_tensor(split_is_training, device=u.device, dtype=torch.bool)
    if u.shape != z.shape or u.ndim != 3 or training.shape != (u.shape[0],):
        raise ValueError("PIT arrays must be [origin, entity, lead] with one training flag per origin")
    if mode == "nominal_cells":
        return z, None
    if mode != "training_frequency":
        raise ValueError(f"unknown dependence PIT transform: {mode}")
    mapping = fit_training_frequency_midpoints(u[training], quantile_levels)
    calibrated_u = apply_training_frequency_midpoints(u, quantile_levels, mapping)
    return gaussianize_pit(calibrated_u, eps=eps), mapping


def build_group_pit(
    true_y: torch.Tensor,
    quantile_predictions: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    monotone_repair: MonotoneRepair = "none",
    eps: float = 1e-7,
) -> PITResult:
    """Build PIT arrays while retaining only complete ``(origin, lead)`` vectors.

    Expected shapes are ``true_y: [N, K, H]`` and quantile predictions
    ``[N, K, H, Q]``. Invalidity of one entity invalidates all ``K`` entries at
    the same forecast instance and lead.
    """

    truth = torch.as_tensor(true_y)
    predictions = torch.as_tensor(quantile_predictions)
    if truth.ndim != 3 or predictions.ndim != 4 or predictions.shape[:-1] != truth.shape:
        raise ValueError("expected true_y [origin, entity, lead] and predictions [origin, entity, lead, quantile]")
    diagnostics = crossing_diagnostics(predictions)
    u, valid_entity = discretized_pit(
        truth,
        predictions,
        quantile_levels,
        monotone_repair=monotone_repair,
    )
    valid_origin_lead = valid_entity.all(dim=1)  # [origin, lead]
    complete_mask = valid_origin_lead[:, None, :]
    u = torch.where(complete_mask, u, torch.full_like(u, torch.nan))
    z = gaussianize_pit(u, eps=eps)
    return PITResult(u=u, z=z, valid_origin_lead=valid_origin_lead, crossing_diagnostics=diagnostics)
