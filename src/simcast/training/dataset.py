"""Flatten valid origin/lead PIT vectors for conditional copula training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class DependenceSample:
    """One complete spatial pseudo-observation at a single ``(i, tau)``."""

    features: torch.Tensor
    z: torch.Tensor
    origin_index: int
    lead: int


class DependenceDataset(Dataset[DependenceSample]):
    """Valid cases from features ``[N,K,H,F]`` and PIT scores ``[N,K,H]``."""

    def __init__(self, features: torch.Tensor, pit_z: torch.Tensor) -> None:
        feature_values = torch.as_tensor(features, dtype=torch.float32)
        scores = torch.as_tensor(pit_z, dtype=torch.float32)
        if feature_values.ndim != 4 or scores.ndim != 3 or feature_values.shape[:3] != scores.shape:
            raise ValueError("expected features [origin, entity, lead, feature] and pit_z [origin, entity, lead]")
        if feature_values.shape[-1] == 0 or feature_values.shape[1] < 2:
            raise ValueError("dependence data require at least two entities and one feature")
        feature_valid = torch.isfinite(feature_values).all(dim=(1, 3))
        score_valid = torch.isfinite(scores).all(dim=1)
        self.valid_cases = (feature_valid & score_valid).nonzero(as_tuple=False)
        if self.valid_cases.shape[0] == 0:
            raise ValueError("no complete origin/lead vectors are available")
        self.features = feature_values
        self.pit_z = scores

    def __len__(self) -> int:
        return self.valid_cases.shape[0]

    def __getitem__(self, index: int) -> DependenceSample:
        origin_index, lead_index = (int(value) for value in self.valid_cases[index])
        return DependenceSample(
            features=self.features[origin_index, :, lead_index],
            z=self.pit_z[origin_index, :, lead_index],
            origin_index=origin_index,
            lead=lead_index + 1,
        )


@dataclass(frozen=True, slots=True)
class DependenceBatch:
    """A batch of dependence samples sharing one entity dimension."""

    features: torch.Tensor
    z: torch.Tensor
    origin_indices: torch.Tensor
    leads: torch.Tensor


class DependenceCollator:
    """Collate complete static groups."""

    def __call__(self, samples: list[DependenceSample]) -> DependenceBatch:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        num_entities = samples[0].z.numel()
        if any(sample.z.numel() != num_entities for sample in samples):
            raise ValueError("all samples in a batch must share the physical group")
        feature_rows: list[torch.Tensor] = []
        score_rows: list[torch.Tensor] = []
        for sample in samples:
            feature_rows.append(sample.features)
            score_rows.append(sample.z)
        return DependenceBatch(
            features=torch.stack(feature_rows),
            z=torch.stack(score_rows),
            origin_indices=torch.tensor([sample.origin_index for sample in samples]),
            leads=torch.tensor([sample.lead for sample in samples]),
        )
