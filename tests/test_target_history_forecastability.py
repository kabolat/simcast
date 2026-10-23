"""Forecastability requires at least one finite target observation in the lookback window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _staggered_fixture import (
    START,
    STEP,
    RecordingForecaster,
    inject_adapter,
    staggered_adapter,
    staggered_config,
)

import simcast.cli.build_cache as cache_cli
from simcast.data.adapter import InsufficientTargetHistory
from simcast.data.windows import make_forecast_window
from simcast.fm.cache import load_pit_library

PERIODS = 48
LIFETIMES = ((0, 47), (8, 35), (20, 47))
# Entity 0's lookbacks ending at 27, 29, and 31 lie wholly inside this gap.
GAPS = {0: ((24, 31),)}
CANDIDATES = list(range(3, 46, 2))


def _window(origin_index: int):
    return make_forecast_window(START + origin_index * STEP, lookback_steps=4, horizon_steps=2)


def _origin_indices(values: np.ndarray) -> list[int]:
    return [int((pd.Timestamp(value, tz="UTC") - START) / STEP) for value in values]


def test_variable_mode_marks_empty_history_unforecastable_and_keeps_partial(tmp_path: Path) -> None:
    config = staggered_config(tmp_path, cardinality_mode="variable", entity_count=3)
    adapter = staggered_adapter(config, LIFETIMES, periods=PERIODS, weather_follows_lifetime=False)

    before_b = adapter.prepare_window(_window(7))
    assert before_b.forecastable_entity.tolist() == [True, False, False]
    assert before_b.empty_history is not None and before_b.empty_history.tolist() == [False, True, True]
    assert before_b.entity_indices.tolist() == [0] and len(before_b.inputs) == 1
    assert before_b.history_missing.tolist() == [0, 4, 4]

    partial_b = adapter.prepare_window(_window(9))
    assert partial_b.forecastable_entity.tolist() == [True, True, False]
    partial_history = np.asarray(partial_b.inputs[1]["target"])
    assert np.isnan(partial_history[:2]).all() and np.isfinite(partial_history[2:]).all()


def test_fixed_mode_rejects_an_origin_with_any_empty_required_history(tmp_path: Path) -> None:
    config = staggered_config(tmp_path, cardinality_mode="fixed", entity_count=3)
    adapter = staggered_adapter(config, ((0, 47),) * 3, periods=PERIODS, gaps=GAPS)

    with pytest.raises(InsufficientTargetHistory, match="transformer::e0"):
        adapter.prepare_window(_window(27))
    partial = adapter.prepare_window(_window(25))
    assert partial.forecastable_entity.all()
    assert partial.history_missing.tolist() == [2, 0, 0]


def test_variable_cache_records_empty_history_and_calls_chronos_on_usable_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    config = staggered_config(tmp_path, cardinality_mode="variable", entity_count=3)
    adapter = staggered_adapter(config, LIFETIMES, periods=PERIODS, gaps=GAPS, weather_follows_lifetime=False)
    inject_adapter(monkeypatch, adapter)
    forecaster = RecordingForecaster()

    destination = cache_cli.build_cache_from_config(config, forecaster=forecaster, output_dir=tmp_path / "v")

    library = load_pit_library(destination)
    missingness = library.metadata["missingness"]
    entity_ids = adapter.group.entity_ids
    empty_a = {27, 29, 31}
    empty_b = {o for o in CANDIDATES if o - 3 > 35 or o < 8}
    empty_c = {o for o in CANDIDATES if o < 20}
    assert missingness["history_rule"] == "at_least_one_finite_lookback_target"
    assert missingness["empty_history_rejected_origin_count"] == 0
    assert missingness["empty_history_non_forecastable_count_by_entity"] == dict(
        zip(entity_ids, [len(empty_a), len(empty_b), len(empty_c)], strict=True)
    )
    assert missingness["empty_history_non_forecastable_count"] == len(empty_a) + len(empty_b) + len(empty_c)

    origins = _origin_indices(library.dataset["origin_timestamp"].values)
    forecastable = library.dataset["forecastable_entity"].values
    for row, origin in zip(forecastable, origins, strict=True):
        assert row.tolist() == [origin not in empty_a, origin not in empty_b, origin not in empty_c]

    assert len(forecaster.calls) == len(CANDIDATES)
    assert all(count >= 1 for counts in forecaster.finite_history_counts for count in counts)
    assert any(count < 4 for counts in forecaster.finite_history_counts for count in counts)
    assert sum(len(call) for call in forecaster.calls) == 3 * len(CANDIDATES) - (
        len(empty_a) + len(empty_b) + len(empty_c)
    )


def test_fixed_cache_counts_origins_rejected_for_empty_history(monkeypatch, tmp_path: Path) -> None:
    config = staggered_config(tmp_path, cardinality_mode="fixed", entity_count=3)
    adapter = staggered_adapter(config, ((0, 47),) * 3, periods=PERIODS, gaps=GAPS)
    inject_adapter(monkeypatch, adapter)
    forecaster = RecordingForecaster()

    destination = cache_cli.build_cache_from_config(config, forecaster=forecaster, output_dir=tmp_path / "f")

    library = load_pit_library(destination)
    origins = library.metadata["origins"]
    assert origins["candidate_count"] == len(CANDIDATES)
    assert origins["eligible_count"] == len(CANDIDATES) - 3
    assert library.metadata["missingness"]["empty_history_rejected_origin_count"] == 3
    assert library.metadata["missingness"]["empty_history_non_forecastable_count"] == 0
    assert all("no finite historical target" in item["reason"] for item in origins["skipped"])
    assert not {27, 29, 31} & set(_origin_indices(library.dataset["origin_timestamp"].values))
    assert library.dataset["forecastable_entity"].values.all()
    assert all(len(call) == 3 for call in forecaster.calls)
