"""Candidate coverage and forecastability for entities with staggered observation lifetimes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _staggered_fixture import (
    START,
    STEP,
    RecordingForecaster,
    expected_forecastable,
    inject_adapter,
    staggered_adapter,
    staggered_config,
    timestamp,
)

import simcast.cli.build_cache as cache_cli
from simcast.fm.cache import load_pit_library

PERIODS = 48
# "Jan-Dec", "Mar-Oct", and "Jun-Dec" on a compressed 48-step timeline.
LIFETIMES = ((0, 47), (8, 35), (20, 47))


def _origin_indices(values: np.ndarray) -> list[int]:
    return [int((pd.Timestamp(value, tz="UTC") - START) / STEP) for value in values]


def _build(monkeypatch, tmp_path: Path, mode: str) -> tuple[Path, RecordingForecaster, list[str]]:
    config = staggered_config(tmp_path, cardinality_mode=mode, entity_count=len(LIFETIMES))
    adapter = staggered_adapter(config, LIFETIMES, periods=PERIODS)
    inject_adapter(monkeypatch, adapter)
    forecaster = RecordingForecaster()
    destination = cache_cli.build_cache_from_config(config, forecaster=forecaster, output_dir=tmp_path / mode)
    return destination, forecaster, adapter.group.entity_ids


@pytest.mark.parametrize(("mode", "expected"), [("fixed", (20, 35)), ("variable", (0, 47))])
def test_coverage_is_intersection_for_fixed_and_envelope_for_variable(tmp_path: Path, mode, expected) -> None:
    config = staggered_config(tmp_path, cardinality_mode=mode, entity_count=len(LIFETIMES))
    adapter = staggered_adapter(config, LIFETIMES, periods=PERIODS)

    assert adapter.coverage() == (timestamp(expected[0]), timestamp(expected[1]))


def test_fixed_mode_uses_only_the_common_interval(monkeypatch, tmp_path: Path) -> None:
    destination, forecaster, entity_ids = _build(monkeypatch, tmp_path, "fixed")
    library = load_pit_library(destination)

    assert library.metadata["origins"]["candidate_count"] == 6
    origins = _origin_indices(library.dataset["origin_timestamp"].values)
    assert min(origins) >= 20 + 3 and max(origins) <= 35 - 2
    assert library.dataset["forecastable_entity"].values.all()
    assert all(call == entity_ids for call in forecaster.calls)
    forecast_protocol = library.metadata["forecast_protocol"]
    assert (forecast_protocol["coverage_start"], forecast_protocol["coverage_end"]) == (
        timestamp(20).isoformat(),
        timestamp(35).isoformat(),
    )


def test_variable_mode_spans_the_envelope_and_entities_enter_and_leave(monkeypatch, tmp_path: Path) -> None:
    destination, forecaster, entity_ids = _build(monkeypatch, tmp_path, "variable")
    library = load_pit_library(destination)

    assert library.metadata["origins"]["candidate_count"] == 22
    assert library.metadata["origins"]["eligible_count"] == 22
    origins = _origin_indices(library.dataset["origin_timestamp"].values)
    assert min(origins) < 8 and max(origins) > 35
    assert library.dataset["entity_id"].values.tolist() == entity_ids
    assert library.metadata["entity_group"]["entity_ids"] == entity_ids

    forecastable = library.dataset["forecastable_entity"].values
    expected = np.asarray([[expected_forecastable(origin, life) for life in LIFETIMES] for origin in origins])
    np.testing.assert_array_equal(forecastable, expected)
    # B (index 1) is forecastable from origin 11 through 33; C (index 2) from origin 23 onward.
    entered_b = origins[int(np.argmax(forecastable[:, 1]))]
    left_b = origins[len(origins) - 1 - int(np.argmax(forecastable[::-1, 1]))]
    entered_c = origins[int(np.argmax(forecastable[:, 2]))]
    assert (entered_b, left_b, entered_c) == (11, 33, 23)
    assert forecastable[:, 0].all() and forecastable[-1, 2] and not forecastable[-1, 1]

    for call in forecaster.calls:
        assert call == [entity for entity in entity_ids if entity in call]
    quantiles = library.dataset["quantile_prediction"].values
    assert np.isnan(quantiles[~forecastable]).all()
    assert np.isfinite(quantiles[forecastable]).all()
