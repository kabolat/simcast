"""Permutation-equivariant set-aware conditional low-rank copula (M3)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from simcast.dependence.conditional_low_rank import low_rank_correlation


class SetAwareLowRankGaussianCopula(nn.Module):
    """Inspect the current entity set before producing shared factor loadings.

    No positional or physical entity-ID embedding is used, so reordering the
    entities reorders the output correlation as ``P R P^T``. A padding mask is
    accepted solely as a possible software extension outside the full-group
    protocol, which always supplies the complete static group.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        latent_rank: int = 4,
        dropout: float = 0.1,
        sigma_floor: float = 1e-3,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(input_dim, model_dim, num_layers, num_heads, latent_rank) <= 0:
            raise ValueError("model dimensions, layer count, head count, and rank must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0 <= dropout < 1 or sigma_floor <= 0 or jitter < 0:
            raise ValueError("invalid dropout, sigma floor, or jitter")
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(model_dim, latent_rank + 1)
        self.input_dim = input_dim
        self.latent_rank = latent_rank
        self.sigma_floor = sigma_floor
        self.jitter = jitter
        self.model_kwargs: dict[str, object] = {
            "input_dim": input_dim,
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "latent_rank": latent_rank,
            "dropout": dropout,
            "sigma_floor": sigma_floor,
            "jitter": jitter,
        }

    def factor_parameters(
        self,
        features: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return set-conditioned loadings and uniqueness scales."""

        values = torch.as_tensor(features)
        squeeze = values.ndim == 2
        if squeeze:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [batch, entity, {self.input_dim}] or [entity, feature]")
        mask: torch.Tensor | None = None
        if padding_mask is not None:
            mask = torch.as_tensor(padding_mask, dtype=torch.bool, device=values.device)
            if squeeze and mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != values.shape[:2]:
                raise ValueError("padding_mask must have shape [batch, entity]")
            if bool(mask.all(dim=-1).any()):
                raise ValueError("each set must contain at least one unmasked entity")
        contextual = self.set_encoder(self.input_projection(values), src_key_padding_mask=mask)
        raw = self.head(contextual)
        loadings = raw[..., : self.latent_rank]
        sigma = F.softplus(raw[..., self.latent_rank]) + self.sigma_floor
        if mask is not None:
            loadings = loadings.masked_fill(mask.unsqueeze(-1), 0.0)
            sigma = torch.where(mask, torch.ones_like(sigma), sigma)
        if squeeze:
            return loadings.squeeze(0), sigma.squeeze(0)
        return loadings, sigma

    def forward(
        self,
        features: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loadings, sigma = self.factor_parameters(features, padding_mask=padding_mask)
        return low_rank_correlation(loadings, sigma, jitter=self.jitter)

    def correlation_matrix(
        self,
        features: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Named dependence-inference alias."""

        return self.forward(features, padding_mask=padding_mask)
