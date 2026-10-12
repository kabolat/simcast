from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from simcast.cli.build_cache import build_cache_from_config
from simcast.config import (
    IndependentMethodConfig,
    load_base_config,
    load_composite_config,
    load_method_config,
    resolve_run_config,
)
from simcast.data.adapter import IneligibleWindow
from simcast.data.gefcom2014 import (
    NWP_FILE,
    NWP_VARIABLES,
    TARGETS_FILE,
    ZONES,
    load_adapter,
    nwp_available_at,
    read_nwp,
    read_targets,
)
from simcast.data.windows import build_aligned_group_windows, chronological_split
from simcast.experiments import update_method
from simcast.fm.cache import load_pit_library


def _write_gefcom(root: Path, days: int = 15) -> None:
    (root / "data").mkdir(parents=True)
    index = pd.date_range("2012-01-01 01:00", periods=days * 24, freq="h", name="TIMESTAMP")
    target_values = {
        zone: (0.1 + zone_index * 0.01 + (np.arange(len(index)) % 24) * 0.001).astype(np.float32)
        for zone_index, zone in enumerate(ZONES)
    }
    targets = pd.DataFrame(target_values, index=index)
    targets["knowledge_time"] = pd.Timestamp("2012-01-01")
    targets.to_parquet(root / TARGETS_FILE)

    nwp_index = index
    issue_times = pd.DatetimeIndex([pd.Timestamp(year=stamp.year, month=stamp.month, day=1) for stamp in nwp_index])
    values = {
        f"{zone}_{variable}": np.arange(len(nwp_index), dtype=np.float32) + zone_index + variable_index
        for zone_index, zone in enumerate(ZONES)
        for variable_index, variable in enumerate(NWP_VARIABLES)
    }
    nwp = pd.DataFrame(values, index=pd.MultiIndex.from_arrays(
        [issue_times, nwp_index], names=["issue_time", "valid_time"]
    ))
    nwp["knowledge_time"] = issue_times
    nwp.to_parquet(root / NWP_FILE)


class _Forecast:
    def __init__(self, entity_ids: list[str], inputs, prediction_length: int) -> None:
        medians = torch.tensor([float(np.asarray(item["target"])[-1]) for item in inputs])
        medians = medians[:, None].expand(-1, prediction_length)
        self.entity_ids = entity_ids
        self.quantile_levels = torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        self.quantile_predictions = torch.stack(
            (medians - 0.03, medians - 0.02, medians - 0.01, medians, medians + 0.01, medians + 0.02, medians + 0.03),
            dim=-1,
        )
        self.forecast_embeddings = torch.ones((len(entity_ids), 1, 4))
        self.output_patch_size = 2


class _Forecaster:
    def predict(self, entity_ids, inputs, prediction_length, *, batch_size=256, context_length=None):
        del batch_size, context_length
        return _Forecast(list(entity_ids), inputs, prediction_length)


def _config(root: Path, *, small_windows: bool = False):
    repo = Path(__file__).parents[1]
    overrides = [f"data.local_dir={root}"]
    if small_windows:
        overrides.extend(
            ["forecast.lookback_steps=4", "forecast.horizon_steps=2", "forecast.origin_stride_steps=24"]
        )
    base = load_base_config(repo / "configs/bases/gefcom2014/wind.yaml", overrides=overrides)
    return resolve_run_config(base, IndependentMethodConfig(kind="method", id="m0", family="independent"))


def test_gefcom_parquet_parsing_and_fixed_entity_order(tmp_path: Path) -> None:
    _write_gefcom(tmp_path)
    targets = read_targets(tmp_path / TARGETS_FILE)
    nwp = read_nwp(tmp_path / NWP_FILE)
    adapter = load_adapter(_config(tmp_path, small_windows=True))

    assert targets.index.tz is not None
    assert targets.columns.tolist() == list(ZONES)
    assert nwp.index.tz is not None
    assert "issue_time" not in nwp.columns
    assert adapter.group.entity_ids == [f"gefcom_wind::{zone}" for zone in ZONES]
    assert all(entity.latitude is None and entity.longitude is None for entity in adapter.group.entities)


def test_nwp_run_availability_is_point_in_time() -> None:
    valid = pd.date_range("2012-01-01 01:00", periods=48, freq="h", tz="UTC")
    availability = nwp_available_at(valid, run_start_hour_utc=0, run_length_hours=24)
    assert availability.loc[valid[0]] == pd.Timestamp("2012-01-01 00:00", tz="UTC")
    assert availability.loc[valid[23]] == pd.Timestamp("2012-01-01 00:00", tz="UTC")
    assert availability.loc[valid[24]] == pd.Timestamp("2012-01-02 00:00", tz="UTC")
    assert availability.loc[valid[47]] == pd.Timestamp("2012-01-02 00:00", tz="UTC")


def test_daily_windows_have_168_hour_history_and_24_hour_horizon(tmp_path: Path) -> None:
    _write_gefcom(tmp_path)
    config = _config(tmp_path)
    adapter = load_adapter(config)
    start, end = adapter.coverage()
    forecast = config.forecast
    windows = build_aligned_group_windows(
        start,
        end,
        entity_ids=adapter.group.entity_ids,
        lookback_steps=forecast.lookback_steps,
        horizon_steps=forecast.horizon_steps,
        frequency_minutes=forecast.frequency_minutes,
        origin_stride_steps=forecast.origin_stride_steps,
        origin_time=forecast.origin_time,
    )
    assert len(windows) == 8
    assert all(window.lookback_steps == 168 and window.horizon_steps == 24 for window in windows)
    prepared = adapter.prepare_window(windows[0])
    assert prepared.true_y.shape == (10, 24)
    assert prepared.history_missing.shape == (10,)
    assert len(prepared.inputs) == 10
    assert set(prepared.inputs[0]["future_covariates"]) == set(NWP_VARIABLES) | {"hour_sin", "hour_cos"}


def test_earlier_origin_cannot_use_next_stitched_nwp_run(tmp_path: Path) -> None:
    _write_gefcom(tmp_path)
    adapter = load_adapter(_config(tmp_path, small_windows=True))
    origin = pd.Timestamp("2012-01-01 23:00", tz="UTC")
    window = next(
        window
        for window in build_aligned_group_windows(
            pd.Timestamp("2012-01-01 01:00", tz="UTC"),
            pd.Timestamp("2012-01-16 00:00", tz="UTC"),
            entity_ids=adapter.group.entity_ids,
            lookback_steps=4,
            horizon_steps=2,
            frequency_minutes=60,
            origin_stride_steps=1,
            origin_time="23:00",
        )
        if window.origin_timestamp == origin
    )
    with pytest.raises(IneligibleWindow, match="covariates unavailable"):
        adapter.prepare_window(window)


def test_hourly_window_split_is_chronological_and_purged() -> None:
    origins = pd.date_range("2012-01-01", periods=240, freq="h", tz="UTC")
    split = chronological_split(
        origins,
        horizon_steps=24,
        frequency_minutes=60,
        tune_fraction=0.8,
        validation_fraction_within_tune=0.2,
    )
    assert split.train[-1] < split.validation[0] < split.test[0]
    assert split.purged_train
    assert split.purged_validation


def test_gefcom_synthetic_cache_build_without_chronos(tmp_path: Path) -> None:
    _write_gefcom(tmp_path)
    config = _config(tmp_path, small_windows=True)
    destination = build_cache_from_config(config, forecaster=_Forecaster(), output_dir=tmp_path / "cache")
    library = load_pit_library(destination)

    assert library.dataset.sizes["entity"] == 10
    assert library.dataset.sizes["lead"] == 2
    assert "latitude" not in library.dataset and "longitude" not in library.dataset
    assert library.metadata["dataset"]["adapter"] == "gefcom2014_wind"
    assert library.metadata["dataset"]["provenance"]["nwp_availability"]["inferred"] is True
    assert library.metadata["origins"]["eligible_count"] > 0
    split = library.dataset["split"].values
    truth = library.dataset["true_y"].values
    assert np.isfinite(truth[split != "test"]).all()
    assert np.isnan(truth[split == "test"]).all()


def test_gefcom_download_patterns_are_minimal() -> None:
    from simcast.data.gefcom2014 import allow_patterns

    config = _config(Path("/unused")).data
    assert allow_patterns(config) == [TARGETS_FILE, NWP_FILE]


def test_gefcom_smoke_composite_disables_missing_locations() -> None:
    repo = Path(__file__).parents[1]
    composite_path = repo / "configs/venues/lab/gefcom_wind_smoke.yaml"
    composite = load_composite_config(composite_path)
    entry = next(method for method in composite.methods if method.id == "m2")
    method = load_method_config(composite_path.parent / entry.method)
    resolved = update_method(method, entry.overrides)
    assert resolved.features.use_location is False
    assert resolved.optimization.epochs == 2
