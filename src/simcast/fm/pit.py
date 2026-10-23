"""PIT pseudo-observations for finite native-quantile forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
from sklearn.isotonic import isotonic_regression  # type: ignore[import-untyped]

MonotoneRepair = Literal["none", "isotonic"]
PITMode = Literal["discretized", "linear_interpolation"]


@dataclass(frozen=True)
class CrossingDiagnostics:
    """Quantile-crossing rates for predictions shaped ``[origin, entity, lead, quantile]``."""

    overall: float
    by_entity: np.ndarray
    by_lead: np.ndarray
    crossed: np.ndarray
    eligible: np.ndarray


@dataclass(frozen=True)
class PITResult:
    """PIT arrays and validity of complete spatial vectors.

    ``valid_origin_lead[i, tau]`` is false if any entity has missing truth,
    non-finite forecasts, or a quantile crossing. Invalid ``(origin, lead)``
    vectors are all NaN rather than silently changing group membership.
    """

    u: torch.Tensor
    z: torch.Tensor
    entity_valid: torch.Tensor
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


def crossing_diagnostics(
    quantile_predictions: torch.Tensor,
    forecastable_entity: torch.Tensor | None = None,
) -> CrossingDiagnostics:
    """Summarize crossings for ``[origin, entity, lead, quantile]`` forecasts."""

    predictions = torch.as_tensor(quantile_predictions)
    if predictions.ndim != 4:
        raise ValueError("expected predictions shaped [origin, entity, lead, quantile]")
    eligible = torch.isfinite(predictions).all(dim=-1)
    if forecastable_entity is not None:
        available = torch.as_tensor(forecastable_entity, dtype=torch.bool, device=predictions.device)
        if available.shape == predictions.shape[:2]:
            available = available.unsqueeze(-1)
        if available.shape not in {predictions.shape[:3], predictions.shape[:2] + (1,)}:
            raise ValueError("forecastable_entity must have shape [origin, entity] or [origin, entity, lead]")
        eligible &= available
    crossed = (quantile_crossings(predictions) & eligible).detach().cpu().numpy()
    eligible_array = eligible.detach().cpu().numpy()

    def rate(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return cast(
            np.ndarray,
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=np.float64),
                where=denominator > 0,
            ),
        )

    return CrossingDiagnostics(
        overall=float(crossed.sum() / eligible_array.sum()) if eligible_array.any() else 0.0,
        by_entity=rate(crossed.sum(axis=(0, 2)), eligible_array.sum(axis=(0, 2))),
        by_lead=rate(crossed.sum(axis=(0, 1)), eligible_array.sum(axis=(0, 1))),
        crossed=crossed,
        eligible=eligible_array,
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


def _prepare_pit_inputs(
    true_y: torch.Tensor,
    quantile_predictions: torch.Tensor,
    quantile_levels: torch.Tensor,
    monotone_repair: MonotoneRepair,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return truth, predictions, levels, finite & ~quantile_crossings(predictions)


def discretized_pit(
    true_y: torch.Tensor,
    quantile_predictions: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    monotone_repair: MonotoneRepair = "isotonic",
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

    truth, predictions, levels, valid = _prepare_pit_inputs(
        true_y, quantile_predictions, quantile_levels, monotone_repair
    )
    # searchsorted returns the first quantile prediction >= y (left side).
    flat_predictions = predictions.reshape(-1, predictions.shape[-1]).contiguous()
    flat_truth = truth.reshape(-1, 1).contiguous()
    cell = torch.searchsorted(flat_predictions, flat_truth, right=False).squeeze(-1)

    edges = torch.cat((levels.new_tensor([0.0]), levels, levels.new_tensor([1.0])))
    locations = 0.5 * (edges[:-1] + edges[1:])
    u = locations[cell.clamp_max(levels.numel())].reshape(truth.shape)
    u = torch.where(valid, u, torch.full_like(u, torch.nan))
    return u, valid


def interpolated_pit(
    true_y: torch.Tensor,
    quantile_predictions: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    monotone_repair: MonotoneRepair = "isotonic",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert the piecewise-linear quantile function using a deterministic mid-PIT for atoms.

    The inverse quantile function is linear between native quantile points and
    constant on ``[0, q_1]`` and ``[q_Q, 1]``. Equal adjacent predicted values
    create an atom; an observation exactly at that value receives the midpoint
    of the corresponding probability interval.
    """

    truth, predictions, levels, valid = _prepare_pit_inputs(
        true_y, quantile_predictions, quantile_levels, monotone_repair
    )
    flat_predictions = predictions.reshape(-1, predictions.shape[-1]).contiguous()
    flat_truth = truth.reshape(-1, 1).contiguous()
    quantile_count = levels.numel()
    first = torch.searchsorted(flat_predictions, flat_truth, right=False).squeeze(-1)
    after = torch.searchsorted(flat_predictions, flat_truth, right=True).squeeze(-1)

    if quantile_count == 1:
        only = flat_predictions[:, 0]
        locations = torch.where(flat_truth[:, 0] < only, 0.0, 1.0)
        locations = torch.where(flat_truth[:, 0] == only, 0.5, locations)
    else:
        candidate = flat_predictions.gather(1, first.clamp_max(quantile_count - 1).unsqueeze(-1)).squeeze(-1)
        exact = (first < quantile_count) & (candidate == flat_truth[:, 0])

        lower_exact = torch.where(first == 0, levels.new_zeros(()), levels[first.clamp_max(quantile_count - 1)])
        last_equal = (after - 1).clamp_min(0)
        upper_exact = torch.where(
            last_equal == quantile_count - 1,
            levels.new_ones(()),
            levels[last_equal.clamp_max(quantile_count - 1)],
        )
        exact_location = 0.5 * (lower_exact + upper_exact)

        upper_index = first.clamp(1, quantile_count - 1)
        lower_index = upper_index - 1
        lower_value = flat_predictions.gather(1, lower_index.unsqueeze(-1)).squeeze(-1)
        upper_value = flat_predictions.gather(1, upper_index.unsqueeze(-1)).squeeze(-1)
        weight = (flat_truth[:, 0] - lower_value) / (upper_value - lower_value)
        interpolated = levels[lower_index] + weight * (levels[upper_index] - levels[lower_index])
        locations = torch.where(exact, exact_location, interpolated)
        locations = torch.where((first == 0) & ~exact, levels.new_zeros(()), locations)
        locations = torch.where(first == quantile_count, levels.new_ones(()), locations)

    u = locations.reshape(truth.shape)
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
    *,
    allow_missing: bool = False,
) -> torch.Tensor:
    """Fit empirical PIT-cell midpoints using training origins only.

    ``training_u`` has shape ``[origin, entity, lead]``.  The returned table
    has shape ``[entity, lead, Q + 1]`` and contains cumulative empirical-mass
    midpoints. Non-finite entity observations are ignored. Each entity/lead
    without training observations raises by default or uses nominal cell
    midpoints when ``allow_missing`` is enabled.
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
    fallback = totals.squeeze(-1) == 0
    if bool(fallback.any()) and not allow_missing:
        raise ValueError("every entity/lead needs at least one finite training PIT cell")
    probabilities = counts / totals.clamp_min(1.0)
    mapping = probabilities.cumsum(dim=-1) - 0.5 * probabilities
    if allow_missing:
        nominal_by_entity_lead = nominal.view(1, 1, -1).expand_as(mapping)
        mapping = torch.where(fallback.unsqueeze(-1), nominal_by_entity_lead, mapping)
    return mapping.to(values.dtype)


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
    allow_missing: bool = False,
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
    mapping = fit_training_frequency_midpoints(u[training], quantile_levels, allow_missing=allow_missing)
    calibrated_u = apply_training_frequency_midpoints(u, quantile_levels, mapping)
    return gaussianize_pit(calibrated_u, eps=eps), mapping


def build_group_pit(
    true_y: torch.Tensor,
    quantile_predictions: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    mode: PITMode = "discretized",
    monotone_repair: MonotoneRepair = "isotonic",
    eps: float = 1e-7,
    forecastable_entity: torch.Tensor | None = None,
    cardinality_mode: Literal["fixed", "variable"] = "fixed",
    min_entities_per_case: int = 2,
) -> PITResult:
    """Build PIT arrays using fixed complete vectors or variable active subsets.

    Expected shapes are ``true_y: [N, K, H]`` and quantile predictions
    ``[N, K, H, Q]``. Variable mode keeps each valid entity PIT independently;
    fixed mode retains the historical complete-vector rule.
    """

    predictions = torch.as_tensor(quantile_predictions)
    truth = torch.as_tensor(true_y, device=predictions.device, dtype=predictions.dtype)
    if truth.ndim != 3 or predictions.ndim != 4 or predictions.shape[:-1] != truth.shape:
        raise ValueError("expected true_y [origin, entity, lead] and predictions [origin, entity, lead, quantile]")
    if min_entities_per_case < 1:
        raise ValueError("min_entities_per_case must be positive")
    if cardinality_mode not in {"fixed", "variable"}:
        raise ValueError(f"unknown cardinality mode: {cardinality_mode}")
    forecastable = (
        torch.ones(truth.shape[:2], dtype=torch.bool, device=truth.device)
        if forecastable_entity is None
        else torch.as_tensor(forecastable_entity, dtype=torch.bool, device=truth.device)
    )
    if forecastable.shape != truth.shape[:2]:
        raise ValueError("forecastable_entity must have shape [origin, entity]")
    diagnostics = crossing_diagnostics(predictions, forecastable)
    pit_function = {
        "discretized": discretized_pit,
        "linear_interpolation": interpolated_pit,
    }.get(mode)
    if pit_function is None:
        raise ValueError(f"unknown PIT mode: {mode}")
    u, valid_entity = pit_function(
        truth,
        predictions,
        quantile_levels,
        monotone_repair=monotone_repair,
    )
    entity_valid = valid_entity & forecastable.unsqueeze(-1)
    if cardinality_mode == "fixed":
        valid_origin_lead = entity_valid.all(dim=1)
        entity_valid = entity_valid & valid_origin_lead[:, None, :]
    else:
        valid_origin_lead = entity_valid.sum(dim=1) >= min_entities_per_case
    u = torch.where(entity_valid, u, torch.full_like(u, torch.nan))
    z = gaussianize_pit(u, eps=eps)
    return PITResult(
        u=u,
        z=z,
        entity_valid=entity_valid,
        valid_origin_lead=valid_origin_lead,
        crossing_diagnostics=diagnostics,
    )
