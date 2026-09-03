"""Common interface and sampling for spatial dependence models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Self, cast

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def validate_training_data(
    train_z: ArrayLike,
    entity_ids: Sequence[str],
    valid_mask: ArrayLike | None = None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Validate ``[origin, entity, lead]`` scores and complete-vector mask."""

    if isinstance(train_z, torch.Tensor):
        values = train_z.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        values = np.asarray(train_z, dtype=np.float64)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("train_z must have non-empty shape [origin, entity, lead]")

    ids = tuple(entity_ids)
    if len(ids) != values.shape[1]:
        raise ValueError("entity_ids must match the entity axis of train_z")
    if any(not isinstance(entity_id, str) or not entity_id for entity_id in ids):
        raise ValueError("entity_ids must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("entity_ids must be unique")

    complete = np.isfinite(values).all(axis=1)
    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            supplied = valid_mask.detach().to(device="cpu").numpy()
        else:
            supplied = np.asarray(valid_mask)
        if supplied.shape != complete.shape or supplied.dtype != np.bool_:
            raise ValueError("valid_mask must be boolean with shape [origin, lead]")
        complete &= supplied
    return values, ids, complete


def entity_indices(stored: Sequence[str], requested: Sequence[str] | None) -> tuple[int, ...]:
    """Map a requested entity order onto a fitted entity order."""

    selected = tuple(stored if requested is None else requested)
    if not selected or any(not isinstance(entity_id, str) or not entity_id for entity_id in selected):
        raise ValueError("requested entity_ids must be non-empty strings")
    if len(selected) != len(set(selected)):
        raise ValueError("requested entity_ids must be unique")
    positions = {entity_id: index for index, entity_id in enumerate(stored)}
    unknown = [entity_id for entity_id in selected if entity_id not in positions]
    if unknown:
        raise ValueError(f"unknown entity_ids: {unknown}")
    return tuple(positions[entity_id] for entity_id in selected)


class BaseDependenceModel(ABC):
    """Interface for same-lead cross-entity copula models.

    Public lead values are one based, matching mathematical ``tau in [H]``.
    Implementations that estimate parameters must
    receive an already selected training slice; this interface deliberately has
    no access to validation or test labels.
    """

    @abstractmethod
    def fit(
        self,
        train_z: ArrayLike,
        entity_ids: Sequence[str],
        valid_mask: ArrayLike | None = None,
        **training_data: object,
    ) -> Self:
        """Fit from training-only Gaussianized PIT vectors."""

    @abstractmethod
    def correlation_matrix(
        self,
        lead: int,
        *,
        entity_ids: Sequence[str] | None = None,
        **condition: object,
    ) -> torch.Tensor:
        """Return a correlation matrix in the requested entity order."""

    def sample_uniforms(
        self,
        num_samples: int,
        lead: int,
        *,
        entity_ids: Sequence[str] | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        **condition: object,
    ) -> torch.Tensor:
        """Draw Gaussian-copula uniforms shaped ``[sample, entity]``."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        correlation = self.correlation_matrix(lead, entity_ids=entity_ids, **condition)
        if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
            raise ValueError("correlation_matrix must return shape [entity, entity]")
        target_device = correlation.device if device is None else torch.device(device)
        correlation = correlation.to(device=target_device, dtype=dtype)
        factor = torch.linalg.cholesky(correlation)
        noise = torch.randn(
            (num_samples, correlation.shape[0]),
            generator=generator,
            device=target_device,
            dtype=dtype,
        )
        return cast(torch.Tensor, torch.special.ndtr(noise @ factor.mT))

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist the fitted model."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        """Restore a model saved by :meth:`save`."""
