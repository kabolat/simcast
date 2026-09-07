"""Weather selection and deterministic calendar covariates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from simcast.data.availability import (
    as_utc_timestamp,
    select_latest_weather_forecast,
    select_measured_weather,
)
from simcast.data.liander2024 import normalize_timestamp_frame


def _utc_index(timestamps: Sequence[object] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.to_datetime(cast(Any, timestamps), utc=True, errors="raise"),
        name="timestamp",
    )


def calendar_features(
    timestamps: Sequence[object] | pd.DatetimeIndex,
    *,
    include_hour: bool = True,
    include_day_of_week: bool = True,
    include_is_weekend: bool = True,
) -> pd.DataFrame:
    """Encode UTC calendar position with cyclic pairs and a weekend indicator."""

    index = _utc_index(timestamps)
    values: dict[str, np.ndarray] = {}
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0 + index.second.to_numpy() / 3600.0
    if include_hour:
        angle = 2.0 * np.pi * hour / 24.0
        values["hour_sin"] = np.sin(angle)
        values["hour_cos"] = np.cos(angle)
    if include_day_of_week:
        angle = 2.0 * np.pi * index.dayofweek.to_numpy() / 7.0
        values["day_of_week_sin"] = np.sin(angle)
        values["day_of_week_cos"] = np.cos(angle)
    if include_is_weekend:
        values["is_weekend"] = (index.dayofweek.to_numpy() >= 5).astype(np.float32)
    return pd.DataFrame(values, index=index, dtype="float32")


def select_weather_features(frame: pd.DataFrame, weather_features: Sequence[str]) -> pd.DataFrame:
    """Keep configured weather columns in their configured order."""

    normalized = normalize_timestamp_frame(frame)
    missing = [name for name in weather_features if name not in normalized.columns]
    if missing:
        raise ValueError(f"weather columns are missing: {missing}")
    return normalized.loc[:, list(weather_features)].astype("float32")


def _with_calendar(
    weather: pd.DataFrame,
    *,
    calendar: bool,
    include_hour: bool,
    include_day_of_week: bool,
    include_is_weekend: bool,
) -> pd.DataFrame:
    if not calendar:
        return weather
    return weather.join(
        calendar_features(
            pd.DatetimeIndex(weather.index),
            include_hour=include_hour,
            include_day_of_week=include_day_of_week,
            include_is_weekend=include_is_weekend,
        )
    )


def build_past_covariates(
    measurements: pd.DataFrame,
    timestamps: Sequence[object] | pd.DatetimeIndex,
    weather_features: Sequence[str],
    *,
    origin_timestamp: str | pd.Timestamp | None = None,
    calendar: bool = True,
    include_hour: bool = True,
    include_day_of_week: bool = True,
    include_is_weekend: bool = True,
) -> pd.DataFrame:
    """Build aligned past covariates from measured weather."""

    requested = _utc_index(timestamps)
    origin = as_utc_timestamp(origin_timestamp) if origin_timestamp is not None else requested.max()
    measured = select_measured_weather(measurements, origin, requested)
    weather = select_weather_features(measured, weather_features)
    return _with_calendar(
        weather,
        calendar=calendar,
        include_hour=include_hour,
        include_day_of_week=include_day_of_week,
        include_is_weekend=include_is_weekend,
    )


def build_future_covariates(
    forecasts: pd.DataFrame,
    origin_timestamp: str | pd.Timestamp,
    timestamps: Sequence[object] | pd.DatetimeIndex,
    weather_features: Sequence[str],
    *,
    measurements: pd.DataFrame | None = None,
    future_weather_source: Literal["vintage", "oracle"] = "vintage",
    calendar: bool = True,
    include_hour: bool = True,
    include_day_of_week: bool = True,
    include_is_weekend: bool = True,
) -> pd.DataFrame:
    """Build future covariates from weather vintages or realized measurements."""

    origin = as_utc_timestamp(origin_timestamp)
    requested = _utc_index(timestamps)
    if bool((requested <= origin).any()):
        raise ValueError("future covariate timestamps must be strictly after the origin")
    if future_weather_source == "vintage":
        weather_frame = select_latest_weather_forecast(forecasts, origin, requested)
    elif future_weather_source == "oracle":
        if measurements is None:
            raise ValueError("oracle future weather requires weather measurements")
        weather_frame = normalize_timestamp_frame(measurements).reindex(requested)
    else:
        raise ValueError("future_weather_source must be 'vintage' or 'oracle'")
    weather = select_weather_features(weather_frame, weather_features)
    return _with_calendar(
        weather,
        calendar=calendar,
        include_hour=include_hour,
        include_day_of_week=include_day_of_week,
        include_is_weekend=include_is_weekend,
    )


def missingness_percent(frame: pd.DataFrame) -> dict[str, float]:
    """Return per-column missing percentages for run metadata/logging."""

    if frame.empty:
        return {column: 0.0 for column in frame.columns}
    return {str(column): float(value) for column, value in (100.0 * frame.isna().mean()).items()}
