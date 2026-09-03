from __future__ import annotations

import pandas as pd

from simcast.data.availability import (
    mask_unavailable_targets,
    select_available_past_targets,
    select_latest_weather_forecast,
)


def test_two_day_target_delay_and_future_truth_are_masked() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "load": [1.0, 2.0, 3.0, 4.0]})
    origin = pd.Timestamp("2024-01-03", tz="UTC")

    solar = mask_unavailable_targets(frame, origin, entity_type="solar_park")
    assert solar["load"].tolist()[:1] == [1.0]
    assert solar["load"].iloc[1:].isna().all()

    transformer = select_available_past_targets(frame, origin, entity_type="transformer")
    assert transformer.tolist() == [1.0, 2.0, 3.0]
    assert timestamps[-1] not in transformer.index


def test_available_at_is_authoritative_and_future_target_never_enters_input() -> None:
    origin = pd.Timestamp("2024-01-02", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [origin - pd.Timedelta(minutes=15), origin, origin + pd.Timedelta(minutes=15)],
            "load": [1.0, 2.0, 999.0],
            "available_at": [origin + pd.Timedelta(minutes=1), origin, origin - pd.Timedelta(days=1)],
        }
    )
    masked = mask_unavailable_targets(frame, origin)
    assert pd.isna(masked["load"].iloc[0])
    assert masked["load"].iloc[1] == 2.0
    assert pd.isna(masked["load"].iloc[2])


def test_latest_weather_vintage_uses_origin_equality_and_excludes_later_revision() -> None:
    origin = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    target = origin + pd.Timedelta(minutes=15)
    other_target = origin + pd.Timedelta(minutes=30)
    forecasts = pd.DataFrame(
        {
            "timestamp": [target, target, target, other_target],
            "available_at": [
                origin - pd.Timedelta(hours=1),
                origin,
                origin + pd.Timedelta(seconds=1),
                origin - pd.Timedelta(hours=2),
            ],
            "temperature_2m": [10.0, 20.0, 999.0, 30.0],
        }
    )

    selected = select_latest_weather_forecast(forecasts, origin, [target, other_target])
    assert selected.loc[target, "temperature_2m"] == 20.0
    assert selected.loc[target, "available_at"] == origin
    assert selected.loc[other_target, "temperature_2m"] == 30.0
    assert bool((selected["available_at"] <= origin).all())

