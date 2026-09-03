"""Conditional low-rank Gaussian copula (method M2)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def covariance_to_correlation(covariance: torch.Tensor, *, jitter: float = 0.0) -> torch.Tensor:
    """Symmetrize and normalize covariance matrices to unit-diagonal correlations."""

    matrix = torch.as_tensor(covariance)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("covariance must have shape [..., entity, entity]")
    if not torch.is_floating_point(matrix):
        matrix = matrix.to(torch.float32)
    if jitter < 0:
        raise ValueError("jitter must be non-negative")
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    if jitter:
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        matrix = matrix + jitter * identity
    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    if not bool(torch.all(diagonal > 0)):
        raise ValueError("covariance diagonal must be positive")
    inverse_scale = diagonal.rsqrt()
    correlation = matrix * inverse_scale.unsqueeze(-1) * inverse_scale.unsqueeze(-2)
    correlation = 0.5 * (correlation + correlation.transpose(-1, -2))
    # Avoid in-place diagonal writes so autograd keeps a clean graph.
    return correlation + torch.diag_embed(torch.ones_like(diagonal) - correlation.diagonal(dim1=-2, dim2=-1))


def low_rank_correlation(
    loadings: torch.Tensor,
    sigma: torch.Tensor,
    *,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Construct ``R`` from ``Sigma = Lambda Lambda^T + diag(sigma^2)``."""

    factors = torch.as_tensor(loadings)
    noise = torch.as_tensor(sigma, device=factors.device, dtype=factors.dtype)
    if factors.ndim < 2:
        raise ValueError("loadings must have shape [..., entity, rank]")
    if noise.shape != factors.shape[:-1]:
        raise ValueError("sigma must have shape [..., entity] matching loadings")
    covariance = factors @ factors.transpose(-1, -2) + torch.diag_embed(noise.square())
    return covariance_to_correlation(covariance, jitter=jitter)


class ConditionalLowRankGaussianCopula(nn.Module):
    """Predict a spatial correlation matrix with one shared entity-wise MLP.

    The model represents each Gaussianized PIT score as
    ``z_k = lambda_k^T xi + sigma_k epsilon_k``.  The latent ``xi`` captures
    unresolved common uncertainty; ``epsilon_k`` is entity-specific noise.
    Parameter count is independent of the number and ordering of entities.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        latent_rank: int = 4,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        layer_norm: bool = True,
        sigma_floor: float = 1.0e-3,
        jitter: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if latent_rank <= 0:
            raise ValueError("latent_rank must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if sigma_floor <= 0:
            raise ValueError("sigma_floor must be positive")
        if jitter < 0:
            raise ValueError("jitter must be non-negative")

        self.input_norm: nn.Module = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, latent_rank + 1))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.latent_rank = latent_rank
        self.sigma_floor = sigma_floor
        self.jitter = jitter
        self.model_kwargs: dict[str, object] = {
            "input_dim": input_dim,
            "latent_rank": latent_rank,
            "hidden_dims": tuple(hidden_dims),
            "dropout": dropout,
            "layer_norm": layer_norm,
            "sigma_floor": sigma_floor,
            "jitter": jitter,
        }

    def factor_parameters(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return entity-wise factor loadings and strictly positive noise scales."""

        values = torch.as_tensor(features)
        if values.ndim < 2 or values.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [..., entity, {self.input_dim}]")
        raw = self.network(self.input_norm(values))
        loadings = raw[..., : self.latent_rank]
        sigma = F.softplus(raw[..., self.latent_rank]) + self.sigma_floor
        return loadings, sigma

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return conditional correlations shaped ``[..., K, K]``."""

        loadings, sigma = self.factor_parameters(features)
        return low_rank_correlation(loadings, sigma, jitter=self.jitter)

    def correlation_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Named alias matching the dependence-model inference vocabulary."""

        return self.forward(features)
