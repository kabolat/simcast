"""Flatten valid origin/lead PIT vectors for conditional copula training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class DependenceSample:
    """One compact active-entity pseudo-observation at a single ``(i, tau)``."""

    features: torch.Tensor
    z: torch.Tensor
    active_entity_indices: torch.Tensor
    origin_index: int
    lead: int

    @property
    def active_count(self) -> int:
        return self.z.numel()


class DependenceDataset(Dataset[DependenceSample]):
    """Compact valid cases from features ``[N,K,H,F]`` and PIT scores ``[N,K,H]``."""

    def __init__(
        self,
        features: torch.Tensor,
        pit_z: torch.Tensor,
        entity_valid: torch.Tensor | None = None,
        *,
        min_entities_per_case: int = 2,
    ) -> None:
        feature_values = torch.as_tensor(features, dtype=torch.float32)
        scores = torch.as_tensor(pit_z, dtype=torch.float32)
        if feature_values.ndim != 4 or scores.ndim != 3 or feature_values.shape[:3] != scores.shape:
            raise ValueError("expected features [origin, entity, lead, feature] and pit_z [origin, entity, lead]")
        if feature_values.shape[-1] == 0 or feature_values.shape[1] < min_entities_per_case:
            raise ValueError("dependence data require the minimum number of entities and one feature")
        if min_entities_per_case < 1:
            raise ValueError("min_entities_per_case must be positive")
        finite_features = torch.isfinite(feature_values).all(dim=-1)
        finite_scores = torch.isfinite(scores)
        if entity_valid is None:
            complete_case = finite_features.all(dim=1) & finite_scores.all(dim=1)
            active = (finite_features & finite_scores) & complete_case[:, None, :]
        else:
            supplied = torch.as_tensor(entity_valid, device=feature_values.device)
            if supplied.shape != scores.shape:
                raise ValueError("entity_valid must have shape [origin, entity, lead]")
            active = supplied.to(torch.bool) & finite_features & finite_scores
        self.valid_cases = (active.sum(dim=1) >= min_entities_per_case).nonzero(as_tuple=False)
        if self.valid_cases.shape[0] == 0:
            raise ValueError("no origin/lead cases meet min_entities_per_case with finite features and scores")
        self.features = feature_values
        self.pit_z = scores
        self.entity_valid = active
        self.min_entities_per_case = min_entities_per_case

    def __len__(self) -> int:
        return self.valid_cases.shape[0]

    def __getitem__(self, index: int) -> DependenceSample:
        origin_index, lead_index = (int(value) for value in self.valid_cases[index])
        active_entity_indices = self.entity_valid[origin_index, :, lead_index].nonzero(as_tuple=False).flatten()
        return DependenceSample(
            features=self.features[origin_index, :, lead_index].index_select(0, active_entity_indices),
            z=self.pit_z[origin_index, :, lead_index].index_select(0, active_entity_indices),
            active_entity_indices=active_entity_indices,
            origin_index=origin_index,
            lead=lead_index + 1,
        )


@dataclass(frozen=True, slots=True)
class DependenceBatch:
    """A padded batch of compact dependence samples."""

    features: torch.Tensor
    z: torch.Tensor
    active_mask: torch.Tensor
    canonical_entity_indices: torch.Tensor
    origin_indices: torch.Tensor
    leads: torch.Tensor

    @property
    def padding_mask(self) -> torch.Tensor:
        return ~self.active_mask

    @property
    def active_counts(self) -> torch.Tensor:
        return self.active_mask.sum(dim=-1)


class DependenceCollator:
    """Pad compact samples to the largest active set in each mini-batch."""

    def __call__(self, samples: list[DependenceSample]) -> DependenceBatch:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        max_entities = max(sample.active_count for sample in samples)
        feature_dim = samples[0].features.shape[-1]
        features = samples[0].features.new_zeros((len(samples), max_entities, feature_dim))
        scores = samples[0].z.new_zeros((len(samples), max_entities))
        active_mask = torch.zeros((len(samples), max_entities), dtype=torch.bool)
        canonical_indices = torch.full((len(samples), max_entities), -1, dtype=torch.long)
        for row, sample in enumerate(samples):
            if sample.features.shape != (sample.active_count, feature_dim):
                raise ValueError("sample features must have shape [active_entity, feature]")
            if sample.active_count == 0 or not bool(torch.isfinite(sample.features).all()):
                raise ValueError("samples must contain finite active features")
            if not bool(torch.isfinite(sample.z).all()):
                raise ValueError("samples must contain finite active scores")
            count = sample.active_count
            features[row, :count] = sample.features
            scores[row, :count] = sample.z
            active_mask[row, :count] = True
            canonical_indices[row, :count] = sample.active_entity_indices
        return DependenceBatch(
            features=features,
            z=scores,
            active_mask=active_mask,
            canonical_entity_indices=canonical_indices,
            origin_indices=torch.tensor([sample.origin_index for sample in samples]),
            leads=torch.tensor([sample.lead for sample in samples]),
        )
