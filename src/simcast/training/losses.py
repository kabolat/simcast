"""Numerically stable dependence-model objectives."""

from __future__ import annotations

from typing import Literal

import torch

Reduction = Literal["none", "mean", "sum"]


def stabilize_correlation(correlation: torch.Tensor, *, jitter: float = 1.0e-6) -> torch.Tensor:
    """Symmetrize, impose unit diagonal, add jitter, and renormalize."""

    matrix = torch.as_tensor(correlation)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("correlation must have shape [..., entity, entity]")
    if jitter < 0:
        raise ValueError("jitter must be non-negative")
    if not torch.is_floating_point(matrix):
        matrix = matrix.to(torch.float32)

    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    matrix = matrix + torch.diag_embed(torch.ones_like(diagonal) - diagonal)
    if jitter:
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        matrix = matrix + jitter * identity
    scale = matrix.diagonal(dim1=-2, dim2=-1).clamp_min(torch.finfo(matrix.dtype).eps).rsqrt()
    matrix = matrix * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    return matrix + torch.diag_embed(torch.ones_like(diagonal) - diagonal)


def gaussian_copula_pseudo_nll(
    z: torch.Tensor,
    correlation: torch.Tensor,
    *,
    jitter: float = 1.0e-6,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Gaussian-copula negative pseudo-log-likelihood up to constants.

    Computes

    .. math::

       \tfrac12\{\log|R| + z^\top(R^{-1}-I)z\}

    using Cholesky solves.  No matrix inverse is formed.
    """

    scores = torch.as_tensor(z)
    matrix = torch.as_tensor(correlation, device=scores.device)
    if not torch.is_floating_point(scores):
        scores = scores.to(torch.float32)
    matrix = matrix.to(dtype=scores.dtype)
    if scores.ndim < 1:
        raise ValueError("z must have shape [..., entity]")
    if matrix.ndim < 2 or matrix.shape[-2:] != (scores.shape[-1], scores.shape[-1]):
        raise ValueError("correlation entity dimensions must match z")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    try:
        scores, _ = torch.broadcast_tensors(scores, matrix[..., 0])  # type: ignore[no-untyped-call]
    except RuntimeError as error:
        raise ValueError("z and correlation batch dimensions are not broadcastable") from error
    matrix = matrix.expand(*scores.shape[:-1], *matrix.shape[-2:])
    matrix = stabilize_correlation(matrix, jitter=jitter)

    cholesky = torch.linalg.cholesky(matrix)
    solved = torch.cholesky_solve(scores.unsqueeze(-1), cholesky).squeeze(-1)
    log_determinant = 2.0 * torch.log(cholesky.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)
    quadratic_difference = (scores * solved).sum(dim=-1) - scores.square().sum(dim=-1)
    loss: torch.Tensor = 0.5 * (log_determinant + quadratic_difference)
    if reduction == "mean":
        return torch.as_tensor(loss.mean())
    if reduction == "sum":
        return torch.as_tensor(loss.sum())
    return torch.as_tensor(loss)


def gaussian_copula_nll(
    z: torch.Tensor,
    correlation: torch.Tensor,
    *,
    jitter: float = 1.0e-6,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Short alias for :func:`gaussian_copula_pseudo_nll`."""

    return gaussian_copula_pseudo_nll(z, correlation, jitter=jitter, reduction=reduction)
