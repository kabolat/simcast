"""Same-lead Gaussian-copula sampling with fixed discrete marginals."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from simcast.sampling.quantile_projection import project_uniforms_to_quantiles


@dataclass(frozen=True, slots=True)
class ScenarioBatch:
    """Entity and aggregate scenarios for one or a batch of ``(i, tau)`` cases."""

    uniforms: torch.Tensor
    quantile_indices: torch.Tensor
    entity_samples: torch.Tensor
    aggregate_samples: torch.Tensor


def _as_batched_correlation(correlation: torch.Tensor) -> tuple[torch.Tensor, bool]:
    matrix = torch.as_tensor(correlation)
    if matrix.ndim not in {2, 3} or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("correlation must have shape [K, K] or [B, K, K]")
    return (matrix.unsqueeze(0), True) if matrix.ndim == 2 else (matrix, False)


def sample_gaussian_uniforms(
    correlation: torch.Tensor,
    num_samples: int,
    *,
    generator: torch.Generator | None = None,
    base_normals: torch.Tensor | None = None,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Sample uniforms from one or more Gaussian copulas via Cholesky."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if jitter < 0:
        raise ValueError("jitter cannot be negative")
    matrix, squeeze = _as_batched_correlation(correlation)
    batch_size, entities, _ = matrix.shape
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    scale = torch.diagonal(matrix, dim1=-2, dim2=-1).clamp_min(torch.finfo(matrix.dtype).eps).rsqrt()
    matrix = matrix * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    if jitter:
        matrix = matrix + jitter * torch.eye(entities, dtype=matrix.dtype, device=matrix.device)
        scale = torch.diagonal(matrix, dim1=-2, dim2=-1).rsqrt()
        matrix = matrix * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    cholesky = torch.linalg.cholesky(matrix)

    expected_shape = (batch_size, num_samples, entities)
    if base_normals is None:
        normals = torch.randn(expected_shape, dtype=matrix.dtype, device=matrix.device, generator=generator)
    else:
        normals = torch.as_tensor(base_normals, dtype=matrix.dtype, device=matrix.device)
        if squeeze and normals.shape == expected_shape[1:]:
            normals = normals.unsqueeze(0)
        if normals.shape != expected_shape:
            raise ValueError(f"base_normals must have shape {expected_shape} (or omit batch for one matrix)")
    gaussian = torch.einsum("bij,bmj->bmi", cholesky, normals)
    uniforms: torch.Tensor = torch.special.ndtr(gaussian)
    return uniforms.squeeze(0) if squeeze else uniforms


def generate_scenarios(
    correlation: torch.Tensor,
    quantile_predictions: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    num_samples: int = 4096,
    generator: torch.Generator | None = None,
    base_normals: torch.Tensor | None = None,
    jitter: float = 1e-6,
) -> ScenarioBatch:
    """Generate spatial entity scenarios and their aggregate sums."""

    uniforms = sample_gaussian_uniforms(
        correlation,
        num_samples,
        generator=generator,
        base_normals=base_normals,
        jitter=jitter,
    )
    entity_samples, indices = project_uniforms_to_quantiles(
        uniforms,
        quantile_predictions,
        quantile_levels,
        return_indices=True,
    )
    return ScenarioBatch(
        uniforms=uniforms,
        quantile_indices=indices,
        entity_samples=entity_samples,
        aggregate_samples=entity_samples.sum(dim=-1),
    )
