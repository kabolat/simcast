"""Portable conditional-model checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from simcast.dependence.conditional_low_rank import ConditionalLowRankGaussianCopula
from simcast.fm.feature_builder import FeatureBuilder


@dataclass(frozen=True, slots=True)
class LoadedConditionalModel:
    method: str
    model: nn.Module
    feature_builder: FeatureBuilder
    entity_ids: tuple[str, ...]
    payload: dict[str, Any]


def load_conditional_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedConditionalModel:
    """Restore a supported conditional adapter and its train-only feature statistics."""

    payload: dict[str, Any] = torch.load(Path(path), map_location=device, weights_only=False)
    method = str(payload.get("method"))
    kwargs = payload.get("model_kwargs")
    builder_state = payload.get("feature_builder")
    entity_ids = payload.get("entity_ids")
    state = payload.get("model_state_dict")
    if not isinstance(kwargs, dict) or not isinstance(builder_state, dict) or not isinstance(state, dict):
        raise ValueError("conditional checkpoint is missing model or feature state")
    if not isinstance(entity_ids, (list, tuple)) or not all(isinstance(value, str) for value in entity_ids):
        raise ValueError("conditional checkpoint has invalid entity IDs")
    if method == "conditional_low_rank":
        model: nn.Module = ConditionalLowRankGaussianCopula(**kwargs)
    else:
        raise ValueError(f"unsupported conditional checkpoint method: {method}")
    model.load_state_dict(state)
    model.to(device).eval()
    return LoadedConditionalModel(
        method=method,
        model=model,
        feature_builder=FeatureBuilder.from_state_dict(builder_state),
        entity_ids=tuple(entity_ids),
        payload=payload,
    )
