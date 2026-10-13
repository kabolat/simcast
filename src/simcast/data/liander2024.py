"""Liander2024 Parquet and metadata access."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from simcast.config import DataConfig, ResolvedExperimentConfig
from simcast.data.adapter import DatasetAdapter, EntityFrames, FrameDatasetAdapter
from simcast.data.grouping import ALLOWED_ENTITY_TYPES, build_entity_group, load_target_metadata
from simcast.data.timestamps import normalize_timestamp_frame
from simcast.types import EntityGroup, EntityMetadata

LOGGER = logging.getLogger(__name__)
DATASET_ID = "OpenSTEF/liander2024-energy-forecasting-benchmark"
DATASET_REVISION = "dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044"
COMPONENTS = frozenset({"load_measurements", "weather_measurements", "weather_forecasts_versioned"})


def read_liander_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Liander Parquet file with one consistent timestamp schema."""

    return normalize_timestamp_frame(pd.read_parquet(Path(path)))


def entity_file(
    data_root: str | Path,
    component: str,
    entity_type: str,
    entity_name: str,
) -> Path:
    """Resolve one entity component file and fail with a useful path."""

    if component not in COMPONENTS:
        raise ValueError(f"unsupported Liander component: {component!r}")
    path = Path(data_root) / component / entity_type / f"{entity_name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_entity_frame(
    data_root: str | Path,
    component: str,
    entity: EntityMetadata,
) -> pd.DataFrame:
    """Load one component for one metadata-defined entity."""

    return read_liander_parquet(entity_file(data_root, component, entity.group_name, entity.name))


ADAPTER_NAME = "liander2024"
TARGETS_FILE = "liander2024_targets.yaml"


def allow_patterns(data: DataConfig, *, entity_ids: Sequence[str] = ()) -> list[str]:
    """Minimum snapshot paths needed by the configured entity type."""

    del entity_ids
    entity_type = data.entity_type
    return [
        TARGETS_FILE,
        f"load_measurements/{entity_type}/*.parquet",
        f"weather_measurements/{entity_type}/*.parquet",
        f"weather_forecasts_versioned/{entity_type}/*.parquet",
    ]


def validate_data_config(data: DataConfig) -> None:
    if data.entity_type not in ALLOWED_ENTITY_TYPES:
        allowed = ", ".join(sorted(ALLOWED_ENTITY_TYPES))
        raise ValueError(f"unsupported entity_type {data.entity_type!r}; expected one of: {allowed}")
    if data.options:
        raise ValueError("the liander2024 adapter accepts no data.options")


def default_cache_name(data: DataConfig) -> str:
    return f"liander2024_{data.entity_type}"


def load_adapter(config: ResolvedExperimentConfig) -> DatasetAdapter:
    """Load the configured Liander entity group with load, measured weather, and weather vintages."""

    root = Path(config.data.local_dir).expanduser().resolve()
    group = build_entity_group(root / TARGETS_FILE, config.data.entity_type)
    frames = [
        EntityFrames(
            entity=entity,
            target=load_entity_frame(root, "load_measurements", entity),
            past_weather=load_entity_frame(root, "weather_measurements", entity),
            future_weather=load_entity_frame(root, "weather_forecasts_versioned", entity),
        )
        for entity in group.entities
    ]
    provenance = {
        "adapter": ADAPTER_NAME,
        "source": {"repository": config.data.dataset_id, "revision": config.data.revision},
        "local_dir": str(root),
    }
    return FrameDatasetAdapter(config, group, frames, provenance, past_weather_source="measured")


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Coverage and missingness for one static target group."""

    entity_type: str
    entity_count: int
    row_count: int
    missing_count: int
    missing_percent: float
    start: pd.Timestamp
    end: pd.Timestamp

def summarize_load_measurements(data_root: str | Path, group: EntityGroup) -> DatasetStats:
    """Calculate transparent group-level target coverage and missingness."""

    if not group.entities:
        raise ValueError("group.entities is required to locate measurement files")
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    row_count = 0
    missing_count = 0
    for entity in group.entities:
        frame = load_entity_frame(data_root, "load_measurements", entity)
        if "load" not in frame.columns:
            raise ValueError(f"load column missing for {entity.entity_id}")
        if frame.empty:
            continue
        starts.append(frame.index.min())
        ends.append(frame.index.max())
        row_count += len(frame)
        missing_count += int(frame["load"].isna().sum())
    if not starts:
        raise ValueError("all selected load measurement files are empty")
    stats = DatasetStats(
        entity_type=str(group.metadata.get("entity_type", group.group_id)),
        entity_count=len(group),
        row_count=row_count,
        missing_count=missing_count,
        missing_percent=100.0 * missing_count / row_count,
        start=min(starts),
        end=max(ends),
    )
    LOGGER.info(
        "Liander %s coverage %s to %s; %d entities, %d rows, %.3f%% missing targets",
        stats.entity_type,
        stats.start,
        stats.end,
        stats.entity_count,
        stats.row_count,
        stats.missing_percent,
    )
    return stats


__all__ = [
    "ADAPTER_NAME",
    "allow_patterns",
    "default_cache_name",
    "load_adapter",
    "validate_data_config",
    "COMPONENTS",
    "DATASET_ID",
    "DATASET_REVISION",
    "DatasetStats",
    "entity_file",
    "load_entity_frame",
    "load_target_metadata",
    "normalize_timestamp_frame",
    "read_liander_parquet",
    "summarize_load_measurements",
]
