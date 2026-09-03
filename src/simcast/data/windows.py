"""Leakage-safe forecast windows and chronological origin splits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, time
from itertools import pairwise
from math import floor

import pandas as pd

TimestampInput = str | datetime | pd.Timestamp


@dataclass(frozen=True, slots=True)
class ForecastWindow:
    """One complete, aligned group window around a forecast origin."""

    origin_timestamp: pd.Timestamp
    entity_ids: tuple[str, ...]
    past_timestamps: pd.DatetimeIndex
    future_timestamps: pd.DatetimeIndex

    @property
    def lookback_steps(self) -> int:
        return len(self.past_timestamps)

    @property
    def horizon_steps(self) -> int:
        return len(self.future_timestamps)

    def timestamp_at_lead(self, lead: int) -> pd.Timestamp:
        """Map a one-based forecast lead to its physical timestamp."""

        if not 1 <= lead <= self.horizon_steps:
            raise ValueError(f"lead must be in [1, {self.horizon_steps}], got {lead}")
        return pd.Timestamp(self.future_timestamps[lead - 1])


EntityWindowEligibility = Callable[[str, ForecastWindow], bool]


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Chronological origins after removing leakage at partition boundaries."""

    train: tuple[pd.Timestamp, ...]
    validation: tuple[pd.Timestamp, ...]
    test: tuple[pd.Timestamp, ...]
    purged_train: tuple[pd.Timestamp, ...] = ()
    purged_validation: tuple[pd.Timestamp, ...] = ()

    @property
    def tune(self) -> tuple[pd.Timestamp, ...]:
        return self.train + self.validation


def make_forecast_window(
    origin_timestamp: TimestampInput,
    *,
    lookback_steps: int,
    horizon_steps: int,
    frequency_minutes: int = 15,
    entity_ids: Sequence[str] = (),
) -> ForecastWindow:
    """Build inclusive ``[t-L+1, t]`` and ``[t+1, t+H]`` timestamp windows."""

    _require_positive_int(lookback_steps, "lookback_steps")
    _require_positive_int(horizon_steps, "horizon_steps")
    step = _frequency(frequency_minutes)
    origin = _to_utc(origin_timestamp, "origin_timestamp")
    entities = tuple(entity_ids)
    if len(set(entities)) != len(entities):
        raise ValueError("entity_ids must be unique")

    past = pd.date_range(
        end=origin,
        periods=lookback_steps,
        freq=step,
        name="timestamp",
    )
    future = pd.date_range(
        start=origin + step,
        periods=horizon_steps,
        freq=step,
        name="timestamp",
    )
    return ForecastWindow(origin, entities, past, future)


def generate_origins(
    start_timestamp: TimestampInput,
    end_timestamp: TimestampInput,
    *,
    frequency_minutes: int = 15,
    origin_stride_steps: int = 96,
    origin_time: str | time = "23:45",
) -> pd.DatetimeIndex:
    """Generate inclusive UTC origins on a stable, anchor-relative phase."""

    step = _frequency(frequency_minutes)
    _require_positive_int(origin_stride_steps, "origin_stride_steps")
    start = _to_utc(start_timestamp, "start_timestamp")
    end = _to_utc(end_timestamp, "end_timestamp")
    if start > end:
        return pd.DatetimeIndex([], tz="UTC", name="origin_timestamp")

    anchor = _parse_origin_time(origin_time)
    epoch_anchor = pd.Timestamp("1970-01-01", tz="UTC") + pd.Timedelta(
        hours=anchor.hour,
        minutes=anchor.minute,
    )
    stride = step * origin_stride_steps
    offset_ns = start.value - epoch_anchor.value
    stride_ns = stride.value
    stride_count = -(-offset_ns // stride_ns)
    first = epoch_anchor + stride_count * stride
    return pd.date_range(
        start=first,
        end=end,
        freq=stride,
        name="origin_timestamp",
    )


def build_aligned_group_windows(
    coverage_start: TimestampInput,
    coverage_end: TimestampInput,
    *,
    entity_ids: Sequence[str],
    lookback_steps: int,
    horizon_steps: int,
    frequency_minutes: int = 15,
    origin_stride_steps: int = 96,
    origin_time: str | time = "23:45",
    is_entity_eligible: EntityWindowEligibility | None = None,
) -> tuple[ForecastWindow, ...]:
    """Build only origins for which every member of a static group is eligible.

    ``coverage_start`` and ``coverage_end`` are inclusive bounds of a complete
    fixed-frequency timeline. The callback handles entity-specific gaps and
    covariate availability without changing the physical group definition.
    """

    _require_positive_int(lookback_steps, "lookback_steps")
    _require_positive_int(horizon_steps, "horizon_steps")
    step = _frequency(frequency_minutes)
    start = _to_utc(coverage_start, "coverage_start")
    end = _to_utc(coverage_end, "coverage_end")
    if start > end:
        raise ValueError("coverage_start must not be after coverage_end")

    entities = tuple(entity_ids)
    if not entities:
        raise ValueError("entity_ids must contain at least one entity")
    if len(set(entities)) != len(entities):
        raise ValueError("entity_ids must be unique")

    first_origin = start + (lookback_steps - 1) * step
    last_origin = end - horizon_steps * step
    origins = generate_origins(
        first_origin,
        last_origin,
        frequency_minutes=frequency_minutes,
        origin_stride_steps=origin_stride_steps,
        origin_time=origin_time,
    )

    windows: list[ForecastWindow] = []
    for origin in origins:
        window = make_forecast_window(
            origin,
            lookback_steps=lookback_steps,
            horizon_steps=horizon_steps,
            frequency_minutes=frequency_minutes,
            entity_ids=entities,
        )
        if is_entity_eligible is None or all(
            is_entity_eligible(entity_id, window) for entity_id in entities
        ):
            windows.append(window)
    return tuple(windows)


def chronological_split(
    origins: Sequence[TimestampInput] | pd.DatetimeIndex,
    *,
    horizon_steps: int,
    frequency_minutes: int = 15,
    tune_fraction: float = 0.80,
    validation_fraction_within_tune: float = 0.20,
) -> ChronologicalSplit:
    """Floor-split ordered origins, then purge overlap from earlier partitions."""

    _require_positive_int(horizon_steps, "horizon_steps")
    step = _frequency(frequency_minutes)
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("tune_fraction must be strictly between 0 and 1")
    if not 0.0 <= validation_fraction_within_tune < 1.0:
        raise ValueError("validation_fraction_within_tune must be in [0, 1)")

    ordered = tuple(_to_utc(origin, "origin") for origin in origins)
    if any(left >= right for left, right in pairwise(ordered)):
        raise ValueError("origins must be strictly increasing and unique")

    tune_count = floor(len(ordered) * tune_fraction)
    validation_count = floor(tune_count * validation_fraction_within_tune)
    train_count = tune_count - validation_count

    raw_train = ordered[:train_count]
    raw_validation = ordered[train_count:tune_count]
    test = ordered[tune_count:]

    next_after_train = raw_validation if raw_validation else test
    train, purged_train = _purge_overlapping_origins(
        raw_train,
        next_after_train,
        step=step,
        horizon_steps=horizon_steps,
    )
    validation, purged_validation = _purge_overlapping_origins(
        raw_validation,
        test,
        step=step,
        horizon_steps=horizon_steps,
    )
    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        purged_train=purged_train,
        purged_validation=purged_validation,
    )


def _purge_overlapping_origins(
    earlier: tuple[pd.Timestamp, ...],
    later: tuple[pd.Timestamp, ...],
    *,
    step: pd.Timedelta,
    horizon_steps: int,
) -> tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...]]:
    if not earlier or not later:
        return earlier, ()

    first_later_target = later[0] + step
    keep_count = len(earlier)
    while keep_count and earlier[keep_count - 1] + horizon_steps * step >= first_later_target:
        keep_count -= 1
    return earlier[:keep_count], earlier[keep_count:]


def _frequency(frequency_minutes: int) -> pd.Timedelta:
    _require_positive_int(frequency_minutes, "frequency_minutes")
    return pd.Timedelta(minutes=frequency_minutes)


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _to_utc(value: TimestampInput, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must not be missing")
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _parse_origin_time(value: str | time) -> time:
    try:
        parsed = time.fromisoformat(value) if isinstance(value, str) else value
    except ValueError as error:
        raise ValueError("origin_time must use HH:MM format") from error
    if parsed.tzinfo is not None:
        raise ValueError("origin_time is interpreted in UTC and must not include a timezone")
    if parsed.second or parsed.microsecond:
        raise ValueError("origin_time must have minute precision")
    return parsed
