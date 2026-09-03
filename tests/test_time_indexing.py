from datetime import time

import pandas as pd
import pytest

from simcast.data.windows import (
    ForecastWindow,
    build_aligned_group_windows,
    generate_origins,
    make_forecast_window,
)


def test_window_endpoints_lengths_and_one_based_leads() -> None:
    origin = pd.Timestamp("2025-01-02 23:45", tz="UTC")

    window = make_forecast_window(
        origin,
        lookback_steps=4,
        horizon_steps=3,
        frequency_minutes=15,
        entity_ids=("transformer::a", "transformer::b"),
    )

    assert window.past_timestamps.tolist() == [
        pd.Timestamp("2025-01-02 23:00", tz="UTC"),
        pd.Timestamp("2025-01-02 23:15", tz="UTC"),
        pd.Timestamp("2025-01-02 23:30", tz="UTC"),
        origin,
    ]
    assert window.future_timestamps.tolist() == [
        pd.Timestamp("2025-01-03 00:00", tz="UTC"),
        pd.Timestamp("2025-01-03 00:15", tz="UTC"),
        pd.Timestamp("2025-01-03 00:30", tz="UTC"),
    ]
    assert window.lookback_steps == 4
    assert window.horizon_steps == 3
    assert window.timestamp_at_lead(1) == origin + pd.Timedelta(minutes=15)
    assert window.timestamp_at_lead(3) == origin + pd.Timedelta(minutes=45)
    with pytest.raises(ValueError, match="lead must be"):
        window.timestamp_at_lead(0)
    with pytest.raises(ValueError, match="lead must be"):
        window.timestamp_at_lead(4)


def test_origins_have_default_daily_stride_and_2345_phase() -> None:
    origins = generate_origins(
        pd.Timestamp("2025-01-01 08:00", tz="UTC"),
        pd.Timestamp("2025-01-03 23:45", tz="UTC"),
    )

    assert origins.tolist() == [
        pd.Timestamp("2025-01-01 23:45", tz="UTC"),
        pd.Timestamp("2025-01-02 23:45", tz="UTC"),
        pd.Timestamp("2025-01-03 23:45", tz="UTC"),
    ]


def test_origins_respect_custom_anchor_and_stride_phase() -> None:
    origins = generate_origins(
        "2025-01-01 05:00+00:00",
        "2025-01-03 06:15+00:00",
        frequency_minutes=15,
        origin_stride_steps=96,
        origin_time=time(6, 15),
    )

    assert origins.tolist() == [
        pd.Timestamp("2025-01-01 06:15", tz="UTC"),
        pd.Timestamp("2025-01-02 06:15", tz="UTC"),
        pd.Timestamp("2025-01-03 06:15", tz="UTC"),
    ]


def test_aligned_builder_keeps_static_group_only_when_every_entity_is_eligible() -> None:
    rejected_origin = pd.Timestamp("2025-01-01 00:15", tz="UTC")
    calls: list[tuple[str, pd.Timestamp]] = []

    def eligible(entity_id: str, window: ForecastWindow) -> bool:
        calls.append((entity_id, window.origin_timestamp))
        return not (entity_id == "transformer::b" and window.origin_timestamp == rejected_origin)

    windows = build_aligned_group_windows(
        "2025-01-01 00:00+00:00",
        "2025-01-01 01:00+00:00",
        entity_ids=("transformer::a", "transformer::b"),
        lookback_steps=2,
        horizon_steps=2,
        frequency_minutes=15,
        origin_stride_steps=1,
        origin_time="00:00",
        is_entity_eligible=eligible,
    )

    assert [window.origin_timestamp for window in windows] == [
        pd.Timestamp("2025-01-01 00:30", tz="UTC")
    ]
    assert all(window.entity_ids == ("transformer::a", "transformer::b") for window in windows)
    assert ("transformer::a", rejected_origin) in calls
    assert ("transformer::b", rejected_origin) in calls


def test_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        make_forecast_window(
            "2025-01-01 23:45",
            lookback_steps=2,
            horizon_steps=2,
        )
