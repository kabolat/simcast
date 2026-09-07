from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simcast.data.covariates import (
    build_future_covariates,
    build_past_covariates,
    calendar_features,
    missingness_percent,
)


def test_calendar_features_are_cyclic_and_include_weekend_indicator() -> None:
    timestamps = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01 00:00", tz="UTC"), pd.Timestamp("2024-01-01 06:00", tz="UTC")]
    )
    features = calendar_features(timestamps)
    assert features.loc[timestamps[0], "hour_sin"] == pytest.approx(0.0, abs=1e-7)
    assert features.loc[timestamps[0], "hour_cos"] == pytest.approx(1.0)
    assert features.loc[timestamps[1], "hour_sin"] == pytest.approx(1.0)
    assert set(features) == {
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
    }


def test_past_and_future_covariates_use_distinct_weather_sources_without_leakage() -> None:
    origin = pd.Timestamp("2024-01-01 00:15", tz="UTC")
    past_times = pd.date_range(origin - pd.Timedelta(minutes=15), origin, freq="15min")
    future_times = pd.date_range(origin + pd.Timedelta(minutes=15), periods=2, freq="15min")
    measurements = pd.DataFrame(
        {
            "timestamp": past_times.append(pd.DatetimeIndex([future_times[0]])),
            "temperature_2m": [1.0, 2.0, 999.0],
        }
    )
    forecasts = pd.DataFrame(
        {
            "timestamp": [future_times[0], future_times[0], future_times[1]],
            "available_at": [origin, origin + pd.Timedelta(minutes=1), origin],
            "temperature_2m": [3.0, 999.0, 4.0],
        }
    )

    past = build_past_covariates(
        measurements,
        past_times,
        ["temperature_2m"],
        origin_timestamp=origin,
        calendar=False,
    )
    future = build_future_covariates(forecasts, origin, future_times, ["temperature_2m"], calendar=False)
    np.testing.assert_array_equal(past["temperature_2m"], [1.0, 2.0])
    np.testing.assert_array_equal(future["temperature_2m"], [3.0, 4.0])


def test_oracle_future_weather_uses_realized_measurements() -> None:
    origin = pd.Timestamp("2024-01-01 00:15", tz="UTC")
    future_times = pd.date_range(origin + pd.Timedelta(minutes=15), periods=2, freq="15min")
    measurements = pd.DataFrame(
        {"timestamp": future_times, "temperature_2m": [7.0, 8.0]}
    )
    forecasts = pd.DataFrame(
        {
            "timestamp": future_times,
            "available_at": [origin, origin],
            "temperature_2m": [3.0, 4.0],
        }
    )

    future = build_future_covariates(
        forecasts,
        origin,
        future_times,
        ["temperature_2m"],
        measurements=measurements,
        future_weather_source="oracle",
        calendar=False,
    )
    np.testing.assert_array_equal(future["temperature_2m"], [7.0, 8.0])


def test_missingness_is_explicit() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, np.nan]})
    assert missingness_percent(frame) == {"a": 50.0, "b": 100.0}
