"""Dataset-independent UTC timestamp normalization."""

from __future__ import annotations

from typing import Any

import pandas as pd


def as_utc_index(values: Any, name: str) -> pd.DatetimeIndex:
    try:
        converted = pd.to_datetime(values, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} timestamps") from exc
    return pd.DatetimeIndex(converted, name=name)


def normalize_timestamp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column/index timestamp variants to a sorted UTC ``DatetimeIndex``."""

    normalized = frame.copy()
    normalized = normalized.drop(columns=["__index_level_0__"], errors="ignore")
    if "timestamp" in normalized.columns:
        timestamps = as_utc_index(normalized.pop("timestamp"), "timestamp")
    elif isinstance(normalized.index, pd.DatetimeIndex):
        timestamps = as_utc_index(normalized.index, "timestamp")
    else:
        raise ValueError("frame must have a timestamp column or DatetimeIndex")
    normalized.index = timestamps
    if "available_at" in normalized.columns:
        normalized["available_at"] = pd.to_datetime(normalized["available_at"], utc=True, errors="raise")
    return normalized.sort_index(kind="stable")
