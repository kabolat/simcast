"""Conditional RBF-kernel Gaussian copula (bounded method M4)."""

from __future__ import annotations

from collections.abc import Sequence
from math import log

import torch
from torch import nn
from torch.nn import functional as F

from simcast.dependence.conditional_low_rank import covariance_to_correlation


class ConditionalKernelGaussianCopula(nn.Module):
    """Map entity features to a context-conditioned PSD RBF correlation."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 128),
        embedding_dim: int = 16,
        dropout: float = 0.1,
        initial_length_scale: float = 1.0,
        nugget: float = 1e-3,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(input_dim, embedding_dim) <= 0 or not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("input, hidden, and embedding dimensions must be positive")
        if not 0 <= dropout < 1 or initial_length_scale <= 0 or nugget < 0 or jitter < 0:
            raise ValueError("invalid dropout, length scale, nugget, or jitter")
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.GELU(), nn.Dropout(dropout)))
            previous = width
        layers.append(nn.Linear(previous, embedding_dim))
        self.embedding_network = nn.Sequential(*layers)
        inverse_softplus = log(torch.expm1(torch.tensor(initial_length_scale)).item())
        self.raw_length_scale = nn.Parameter(torch.tensor(inverse_softplus, dtype=torch.float32))
        self.input_dim = input_dim
        self.nugget = nugget
        self.jitter = jitter
        self.model_kwargs: dict[str, object] = {
            "input_dim": input_dim,
            "hidden_dims": tuple(hidden_dims),
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "initial_length_scale": initial_length_scale,
            "nugget": nugget,
            "jitter": jitter,
        }

    @property
    def length_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_length_scale).clamp_min(1e-6)

    def entity_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(features)
        if values.ndim < 2 or values.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [..., entity, {self.input_dim}]")
        return torch.as_tensor(self.embedding_network(values))

    def forward(
        self,
        features: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.entity_embeddings(features)
        squared_distance = torch.cdist(embeddings, embeddings).square()
        kernel = torch.exp(-squared_distance / (2.0 * self.length_scale.square()))
        if self.nugget:
            kernel = kernel + self.nugget * torch.eye(kernel.shape[-1], dtype=kernel.dtype, device=kernel.device)
        correlation = covariance_to_correlation(kernel, jitter=self.jitter)
        if padding_mask is not None:
            mask = torch.as_tensor(padding_mask, dtype=torch.bool, device=correlation.device)
            if mask.shape != correlation.shape[:-1]:
                raise ValueError("padding_mask must have shape [..., entity]")
            valid_pairs = (~mask).unsqueeze(-1) & (~mask).unsqueeze(-2)
            correlation = correlation * valid_pairs + torch.diag_embed(mask.to(correlation.dtype))
        return correlation

    def correlation_matrix(
        self,
        features: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(features, padding_mask=padding_mask)
