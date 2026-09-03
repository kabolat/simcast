"""Liander2024 Parquet and metadata access."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from simcast.data.grouping import load_target_metadata
from simcast.types import EntityGroup, EntityMetadata

LOGGER = logging.getLogger(__name__)
DATASET_ID = "OpenSTEF/liander2024-energy-forecasting-benchmark"
DATASET_REVISION = "dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044"
COMPONENTS = frozenset({"load_measurements", "weather_measurements", "weather_forecasts_versioned"})


def _as_utc(values: Any, name: str) -> pd.DatetimeIndex:
    try:
        converted = pd.to_datetime(values, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} timestamps") from exc
    return pd.DatetimeIndex(converted, name=name)


def normalize_timestamp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Liander's column/index variants to a UTC timestamp index."""

    normalized = frame.copy()
    normalized = normalized.drop(columns=["__index_level_0__"], errors="ignore")
    if "timestamp" in normalized.columns:
        timestamps = _as_utc(normalized.pop("timestamp"), "timestamp")
    elif isinstance(normalized.index, pd.DatetimeIndex):
        timestamps = _as_utc(normalized.index, "timestamp")
    else:
        raise ValueError("frame must have a timestamp column or DatetimeIndex")
    normalized.index = timestamps
    if "available_at" in normalized.columns:
        normalized["available_at"] = pd.to_datetime(normalized["available_at"], utc=True, errors="raise")
    return normalized.sort_index(kind="stable")


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
