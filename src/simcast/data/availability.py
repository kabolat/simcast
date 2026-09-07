"""Forecast-origin availability rules for targets and weather."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from simcast.data.liander2024 import normalize_timestamp_frame


def as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Normalize one timestamp to UTC."""

    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def available_target_mask(
    frame: pd.DataFrame,
    origin_timestamp: str | pd.Timestamp,
) -> pd.Series:
    """Identify target rows measured no later than a forecast origin."""

    normalized = normalize_timestamp_frame(frame)
    origin = as_utc_timestamp(origin_timestamp)
    return pd.Series(
        normalized.index <= origin,
        index=normalized.index,
        name="target_available",
    )


def mask_unavailable_targets(
    frame: pd.DataFrame,
    origin_timestamp: str | pd.Timestamp,
    *,
    target_column: str = "load",
) -> pd.DataFrame:
    """Replace target values measured after the origin with ``NaN``."""

    normalized = normalize_timestamp_frame(frame)
    if target_column not in normalized.columns:
        raise ValueError(f"target column {target_column!r} is missing")
    mask = available_target_mask(normalized, origin_timestamp)
    normalized.loc[~mask, target_column] = float("nan")
    return normalized


def select_available_past_targets(
    frame: pd.DataFrame,
    origin_timestamp: str | pd.Timestamp,
    *,
    target_column: str = "load",
) -> pd.Series:
    """Return target history through the origin."""

    origin = as_utc_timestamp(origin_timestamp)
    masked = mask_unavailable_targets(
        frame,
        origin,
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
    eligible = normalized.loc[normalized["available_at"] <= origin].sort_values("available_at", kind="stable")
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
