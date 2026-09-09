"""Common interface and sampling for spatial dependence models."""

from __future__ import annotations

from collections.abc import Sequence

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
