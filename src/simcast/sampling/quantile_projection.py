"""Nearest-native-quantile projection in probability space."""

from __future__ import annotations

import torch


def _validate_quantile_levels(quantile_levels: torch.Tensor) -> torch.Tensor:
    levels = torch.as_tensor(quantile_levels)
    if levels.ndim != 1 or levels.numel() == 0:
        raise ValueError("quantile_levels must be a non-empty one-dimensional tensor")
    if not torch.is_floating_point(levels):
        levels = levels.to(torch.float32)
    if not bool(torch.all((levels > 0) & (levels < 1))):
        raise ValueError("quantile levels must lie strictly inside (0, 1)")
    if levels.numel() > 1 and not bool(torch.all(levels[1:] > levels[:-1])):
        raise ValueError("quantile levels must be strictly increasing")
    return levels


def quantile_cell_boundaries(quantile_levels: torch.Tensor) -> torch.Tensor:
    """Return ``[0, midpoint(q_j,q_{j+1}), ..., 1]`` cell boundaries."""

    levels = _validate_quantile_levels(quantile_levels)
    midpoints = 0.5 * (levels[:-1] + levels[1:])
    return torch.cat((levels.new_tensor([0.0]), midpoints, levels.new_tensor([1.0])))


def project_uniforms_to_quantiles(
    uniforms: torch.Tensor,
    quantile_predictions: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    return_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Project latent uniforms onto fixed native marginal quantile values.

    ``quantile_predictions`` has shape ``[..., K, Q]`` and ``uniforms`` has
    shape ``[..., M, K]`` with identical prefix dimensions. The result has
    shape ``[..., M, K]``. No interpolation of predicted values occurs.
    """

    values = torch.as_tensor(quantile_predictions)
    draws = torch.as_tensor(uniforms, device=values.device)
    levels = _validate_quantile_levels(torch.as_tensor(quantile_levels, device=values.device)).to(values.dtype)
    if values.ndim < 2 or draws.ndim < 2:
        raise ValueError("predictions [..., K, Q] and uniforms [..., M, K] require at least two dimensions")
    if values.shape[:-2] != draws.shape[:-2] or values.shape[-2] != draws.shape[-1]:
        raise ValueError("prediction and uniform prefix/entity dimensions do not match")
    if values.shape[-1] != levels.numel():
        raise ValueError("prediction quantile axis does not match quantile_levels")
    if not bool(torch.isfinite(draws).all()) or not bool(((draws >= 0) & (draws <= 1)).all()):
        raise ValueError("uniforms must be finite values in [0, 1]")

    boundaries = quantile_cell_boundaries(levels)
    indices = torch.bucketize(draws.contiguous(), boundaries[1:-1].contiguous(), right=True)
    sample_count = draws.shape[-2]
    expanded = values.unsqueeze(-3).expand(*values.shape[:-2], sample_count, *values.shape[-2:])
    projected = torch.gather(expanded, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)
    return (projected, indices) if return_indices else projected
