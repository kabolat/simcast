"""Forecast-origin availability rules for targets and weather."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from simcast.data.liander2024 import normalize_timestamp_frame

TWO_DAY_DELAY_ENTITY_TYPES = frozenset({"solar_park", "wind_park"})


def as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Normalize one timestamp to UTC."""

    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _target_available_at(frame: pd.DataFrame, entity_type: str | None) -> pd.Series:
    if "available_at" in frame.columns:
        return pd.to_datetime(frame["available_at"], utc=True, errors="raise")
    delay = pd.Timedelta(days=2) if entity_type in TWO_DAY_DELAY_ENTITY_TYPES else pd.Timedelta(0)
    return pd.Series(frame.index + delay, index=frame.index, name="available_at")


def available_target_mask(
    frame: pd.DataFrame,
    origin_timestamp: str | pd.Timestamp,
    *,
    entity_type: str | None = None,
) -> pd.Series:
    """Identify target rows knowable at a forecast origin.

    A timestamp after the origin is never considered an input, even if a bad
    source record claims an earlier ``available_at`` value.
    """

    normalized = normalize_timestamp_frame(frame)
    origin = as_utc_timestamp(origin_timestamp)
    return pd.Series(
        (normalized.index <= origin) & (_target_available_at(normalized, entity_type) <= origin),
        index=normalized.index,
        name="target_available",
    )


def mask_unavailable_targets(
    frame: pd.DataFrame,
    origin_timestamp: str | pd.Timestamp,
    *,
    entity_type: str | None = None,
    target_column: str = "load",
) -> pd.DataFrame:
    """Replace target values unavailable at the origin with ``NaN``."""

    normalized = normalize_timestamp_frame(frame)
    if target_column not in normalized.columns:
        raise ValueError(f"target column {target_column!r} is missing")
    mask = available_target_mask(normalized, origin_timestamp, entity_type=entity_type)
    normalized.loc[~mask, target_column] = float("nan")
    return normalized


def select_available_past_targets(
    frame: pd.DataFrame,
    origin_timestamp: str | pd.Timestamp,
    *,
    entity_type: str | None = None,
    target_column: str = "load",
) -> pd.Series:
    """Return history through the origin, masking delayed observations."""

    origin = as_utc_timestamp(origin_timestamp)
    masked = mask_unavailable_targets(
        frame,
        origin,
        entity_type=entity_type,
        target_column=target_column,
    )
    return masked.loc[masked.index <= origin, target_column].rename(target_column)


def select_latest_weather_forecast(
    frame: pd.DataFrame,
    origin_timestamp: str | pd.Timestamp,
    target_timestamps: Sequence[object] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Select the newest weather vintage available by the forecast origin."""

    normalized = normalize_timestamp_frame(frame)
    if "available_at" not in normalized.columns:
        raise ValueError("versioned weather requires an available_at column")
    origin = as_utc_timestamp(origin_timestamp)
    eligible = normalized.loc[normalized["available_at"] <= origin].sort_values(
        "available_at", kind="stable"
    )
    selected = eligible.loc[~eligible.index.duplicated(keep="last")].sort_index(kind="stable")
    if target_timestamps is not None:
        requested = pd.DatetimeIndex(
            pd.to_datetime(cast(Any, target_timestamps), utc=True),
            name="timestamp",
        )
        selected = selected.reindex(requested)
    return selected


def select_measured_weather(
    frame: pd.DataFrame,
    origin_timestamp: str | pd.Timestamp,
    target_timestamps: Sequence[object] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Select historical measurements, whose timestamp is their availability."""

    normalized = normalize_timestamp_frame(frame)
    origin = as_utc_timestamp(origin_timestamp)
    selected = normalized.loc[normalized.index <= origin]
    if target_timestamps is not None:
        requested = pd.DatetimeIndex(
            pd.to_datetime(cast(Any, target_timestamps), utc=True),
            name="timestamp",
        )
        if bool((requested > origin).any()):
            raise ValueError("measured weather cannot supply timestamps after the origin")
        selected = selected.reindex(requested)
    return selected


# Concise aliases for callers that use singular terminology.
mask_unavailable_target = mask_unavailable_targets
latest_available_weather = select_latest_weather_forecast
