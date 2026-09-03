"""Weather selection and deterministic cyclic calendar covariates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

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
    include_day_of_year: bool = True,
) -> pd.DataFrame:
    """Encode calendar position with continuous sine/cosine pairs."""

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
    if include_day_of_year:
        day_fraction = hour / 24.0
        days_in_year = np.where(index.is_leap_year, 366.0, 365.0)
        angle = 2.0 * np.pi * (index.dayofyear.to_numpy() - 1.0 + day_fraction) / days_in_year
        values["day_of_year_sin"] = np.sin(angle)
        values["day_of_year_cos"] = np.cos(angle)
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
    include_day_of_year: bool,
) -> pd.DataFrame:
    if not calendar:
        return weather
    return weather.join(
        calendar_features(
            pd.DatetimeIndex(weather.index),
            include_hour=include_hour,
            include_day_of_week=include_day_of_week,
            include_day_of_year=include_day_of_year,
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
    include_day_of_year: bool = True,
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
        include_day_of_year=include_day_of_year,
    )


def build_future_covariates(
    forecasts: pd.DataFrame,
    origin_timestamp: str | pd.Timestamp,
    timestamps: Sequence[object] | pd.DatetimeIndex,
    weather_features: Sequence[str],
    *,
    calendar: bool = True,
    include_hour: bool = True,
    include_day_of_week: bool = True,
    include_day_of_year: bool = True,
) -> pd.DataFrame:
    """Build aligned future covariates from the newest eligible weather vintage."""

    origin = as_utc_timestamp(origin_timestamp)
    requested = _utc_index(timestamps)
    if bool((requested <= origin).any()):
        raise ValueError("future covariate timestamps must be strictly after the origin")
    forecast = select_latest_weather_forecast(forecasts, origin, requested)
    weather = select_weather_features(forecast, weather_features)
    return _with_calendar(
        weather,
        calendar=calendar,
        include_hour=include_hour,
        include_day_of_week=include_day_of_week,
        include_day_of_year=include_day_of_year,
    )


def missingness_percent(frame: pd.DataFrame) -> dict[str, float]:
    """Return per-column missing percentages for run metadata/logging."""

    if frame.empty:
        return {column: 0.0 for column in frame.columns}
    return {str(column): float(value) for column, value in (100.0 * frame.isna().mean()).items()}
