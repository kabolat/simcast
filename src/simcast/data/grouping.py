"""Build the initial homogeneous Liander entity group from metadata."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]

from simcast.types import EntityGroup, EntityMetadata

LOGGER = logging.getLogger(__name__)
ALLOWED_ENTITY_TYPES = frozenset(
    {"transformer", "solar_park", "wind_park", "mv_feeder", "station_installation"}
)


def _utc_timestamp(value: Any, field_name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc


def _optional_float(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    return None if value is None else float(value)


def load_target_metadata(path: str | Path) -> list[EntityMetadata]:
    """Parse ``liander2024_targets.yaml`` while preserving its entity order."""

    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{source} must contain a YAML list")

    entities: list[EntityMetadata] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"target entry {index} must be a mapping")
        try:
            name = str(item["name"])
            group_name = str(item["group_name"])
            latitude = float(item["latitude"])
            longitude = float(item["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid target entry {index}") from exc

        known = {
            "name",
            "group_name",
            "latitude",
            "longitude",
            "description",
            "benchmark_start",
            "benchmark_end",
            "train_start",
            "upper_limit",
            "lower_limit",
        }
        entities.append(
            EntityMetadata(
                name=name,
                group_name=group_name,
                latitude=latitude,
                longitude=longitude,
                description=str(item.get("description", "")),
                benchmark_start=_utc_timestamp(item.get("benchmark_start"), "benchmark_start"),
                benchmark_end=_utc_timestamp(item.get("benchmark_end"), "benchmark_end"),
                train_start=_utc_timestamp(item.get("train_start"), "train_start"),
                upper_limit=_optional_float(item, "upper_limit"),
                lower_limit=_optional_float(item, "lower_limit"),
                extra={key: value for key, value in item.items() if key not in known},
            )
        )

    ids = [entity.entity_id for entity in entities]
    if len(ids) != len(set(ids)):
        duplicates = sorted({entity_id for entity_id in ids if ids.count(entity_id) > 1})
        raise ValueError(f"duplicate canonical entity IDs: {duplicates}")
    return entities


def build_entity_group(
    targets_yaml: str | Path,
    entity_type: str = "transformer",
    *,
    group_id: str | None = None,
) -> EntityGroup:
    """Build the proof-of-concept's one homogeneous static group."""

    if entity_type not in ALLOWED_ENTITY_TYPES:
        allowed = ", ".join(sorted(ALLOWED_ENTITY_TYPES))
        raise ValueError(f"unsupported entity_type {entity_type!r}; expected one of: {allowed}")

    selected = [entity for entity in load_target_metadata(targets_yaml) if entity.group_name == entity_type]
    if not selected:
        raise ValueError(f"no entities with group_name={entity_type!r}")
    group = EntityGroup(
        group_id=group_id or entity_type,
        entity_ids=[entity.entity_id for entity in selected],
        metadata={"entity_type": entity_type, "entity_count": len(selected)},
        entities=selected,
    )
    LOGGER.info(
        "Selected Liander entity type %s with K_1=%d: %s",
        entity_type,
        len(group),
        ", ".join(entity.name for entity in selected),
    )
    return group


def build_entity_groups(targets_yaml: str | Path, entity_type: str = "transformer") -> list[EntityGroup]:
    """Return the single group as a list, leaving room for later group builders."""

    return [build_entity_group(targets_yaml, entity_type)]
