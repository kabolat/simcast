"""Small, shared data structures for the Simcast data protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def canonical_entity_id(group_name: str, name: str) -> str:
    """Return the dataset-wide identifier for an entity."""

    if not group_name or not name:
        raise ValueError("group_name and name must be non-empty")
    return f"{group_name}::{name}"


@dataclass(frozen=True, slots=True)
class EntityMetadata:
    """Metadata for one physical Liander target."""

    name: str
    group_name: str
    latitude: float
    longitude: float
    description: str = ""
    benchmark_start: pd.Timestamp | None = None
    benchmark_end: pd.Timestamp | None = None
    train_start: pd.Timestamp | None = None
    upper_limit: float | None = None
    lower_limit: float | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def entity_id(self) -> str:
        """Canonical ``group_name::name`` identifier."""

        return canonical_entity_id(self.group_name, self.name)

    @property
    def canonical_id(self) -> str:
        """Alias useful at serialization boundaries."""

        return self.entity_id


@dataclass(slots=True)
class EntityGroup:
    """One static, ordered physical entity group."""

    group_id: str
    entity_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: list[EntityMetadata] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        if not self.entity_ids:
            raise ValueError("an entity group must contain at least one entity")
        if len(self.entity_ids) != len(set(self.entity_ids)):
            raise ValueError("entity_ids must be unique")
        if self.entities and self.entity_ids != [entity.entity_id for entity in self.entities]:
            raise ValueError("entity_ids must match entities and preserve their order")

    def __len__(self) -> int:
        return len(self.entity_ids)
