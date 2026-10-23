from __future__ import annotations

import io
import json
import stat
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import simcast.data.heapo as heapo
from simcast.config import ResolvedExperimentConfig
from simcast.data.heapo import (
    SMALL_FILES,
    archive_members,
    household_number,
    read_households,
    read_smart_meter,
    read_target,
    safe_extract_selected,
    target_summary,
    verify_archive,
)
from simcast.data.registry import adapter_spec

HEADER = (
    "Household_ID;AffectsTimePoint;Group;Timestamp;"
    "kWh_received_Total;kWh_received_HeatPump;kWh_received_Other\n"
)


def _meter(rows: str) -> io.BytesIO:
    return io.BytesIO((HEADER + rows).encode())


def _config(root: Path, *, ids: tuple[int, ...] = (101, 202, 303, 404, 505), group: str = "control"):
    return ResolvedExperimentConfig.model_validate({
        "data": {
            "adapter": "heapo", "dataset_id": heapo.DATASET_ID, "revision": "v1", "local_dir": root,
            "entity_type": "household", "target_column": heapo.TARGET_COLUMN,
            "options": {"group": group, "weather_id": "8jB"},
        },
        "protocol": {
            "entity_count": len(ids), "ordered_entity_ids": [f"household::{hh}" for hh in ids],
            "cardinality_mode": "variable", "min_entities_per_case": min(2, len(ids)),
        },
        "forecast": {
            "lookback_steps": 4, "horizon_steps": 2, "origin_stride_steps": 2, "origin_time": "00:45",
        },
        "covariates": {"weather": ["Temperature_avg_hourly"], "future_weather_source": "oracle"},
        "split": {"tune_fraction": 0.6, "validation_fraction_within_tune": 0.25},
        "chronos": {"device": "cpu"},
        "pit": {"dependence_transform": "training_frequency"},
    })


def _source(root: Path) -> None:
    for name in ["meta_data", "smart_meter_data/15min", "weather_data/hourly"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Household_ID": [101, 202, 303, 404, 505, 606, 707],
        "Group": ["control"] * 5 + ["treatment", "control"],
        "Weather_ID": ["8jB"] * 6 + ["Hg"],
        "SmartMeterData_Available_15min": [True] * 7,
    }).to_csv(root / "meta_data/households.csv", sep=";", index=False)
    pd.DataFrame({
        "Household_ID": [101, 202, 303, 606],
        "Survey_Building_LivingArea": [150.0, 200.0, np.nan, 120.0],
        "Survey_HeatPump_Installation_Type": ["air-source", "ground-source", None, "air-source"],
    }).to_csv(root / "meta_data/meta_data.csv", sep=";", index=False)
    pd.DataFrame({
        "VariableName": ["Household_ID", "Survey_Building_LivingArea", "Survey_HeatPump_Installation_Type"],
        "Description": ["identifier", "living area in square meters", "heat pump type"],
    }).to_csv(root / "meta_data/meta_data_variables.csv", sep=";", index=False)
    times = pd.date_range("2024-01-01", periods=160, freq="15min", tz="UTC")
    lifetimes = ((0, 159), (80, 159), (0, 110), (0, 159), (60, 100), (0, 159), (0, 159))
    rng = np.random.default_rng(17)
    common = rng.normal(size=160)
    for hh, (first, last) in zip([101, 202, 303, 404, 505, 606, 707], lifetimes, strict=True):
        values = 10 + common + 0.3 * rng.normal(size=160)
        data = pd.DataFrame({
            "Household_ID": hh, "AffectsTimePoint": "unknown", "Group": "treatment" if hh == 606 else "control",
            "Timestamp": times[first:last + 1], "kWh_received_Total": values[first:last + 1],
            "kWh_received_HeatPump": np.nan, "kWh_received_Other": np.nan,
        })
        if hh == 404:
            data.loc[data["Timestamp"].between(times[30], times[37]), "kWh_received_Total"] = np.nan
            data.loc[data["Timestamp"].between(times[120], times[127]), "kWh_received_Total"] = np.nan
        data.to_csv(root / f"smart_meter_data/15min/{hh}.csv", sep=";", index=False)
    hourly = pd.date_range("2023-12-31", periods=6 * 24, freq="h", tz="UTC")
    for station in ["8jB", "Hg"]:
        pd.DataFrame({
            "Weather_ID": station, "Timestamp": hourly, "Temperature_avg_hourly": np.arange(len(hourly)) / 10,
        }).to_csv(root / f"weather_data/hourly/{station}.csv", sep=";", index=False)


def test_adapter_registry_native_energy_and_real_lifetimes(tmp_path: Path) -> None:
    _source(tmp_path)
    config = _config(tmp_path, ids=(303, 101, 202))
    adapter = adapter_spec("heapo").load(config)
    assert isinstance(adapter, heapo.FrameDatasetAdapter)
    assert adapter.group.entity_ids == ["household::303", "household::101", "household::202"]
    assert adapter.frames[2].target.index.min() == pd.Timestamp("2024-01-01T20:15Z")
    assert len(adapter.frames[2].target) == 80
    assert all(entity.latitude is None and entity.longitude is None for entity in adapter.group.entities)
    assert adapter.provenance["license"] == "CC-BY-4.0"
    assert adapter.provenance["units"] == "kWh"
    raw = pd.read_csv(tmp_path / "smart_meter_data/15min/202.csv", sep=";")
    np.testing.assert_array_equal(adapter.frames[2].target[heapo.TARGET_COLUMN], raw[heapo.TARGET_COLUMN])


def test_metadata_is_retained_for_diagnostics_not_forecaster_inputs(tmp_path: Path) -> None:
    from simcast.data.windows import make_forecast_window

    _source(tmp_path)
    adapter = heapo.load_adapter(_config(tmp_path))
    survey = adapter.provenance["survey_metadata"]
    assert survey["households_with_survey"] == 3
    assert survey["households_without_survey"] == 2
    assert survey["missing_count_by_field"]["Survey_Building_LivingArea"] == 1
    assert survey["households"][2]["Survey_Building_LivingArea"] is None
    window = make_forecast_window("2024-01-01T23:45Z", lookback_steps=4, horizon_steps=2)
    prepared = adapter.prepare_window(window)
    for model_input in prepared.inputs:
        assert set(model_input) == {"target", "past_covariates", "future_covariates"}
        assert all(not key.startswith("Survey_") for key in model_input["past_covariates"])


def test_tracked_heapo_protocol_is_native_variable_and_oracle() -> None:
    from simcast.config import load_base_config

    base = load_base_config(Path(__file__).parents[1] / "configs/bases/heapo/control_station_8jB.yaml")
    assert base.protocol.cardinality_mode == "variable" and base.protocol.min_entities_per_case == 2
    assert base.protocol.entity_count == 128
    assert len(set(base.protocol.ordered_entity_ids)) == 128
    forecast = base.forecast
    assert (forecast.lookback_steps, forecast.horizon_steps, forecast.origin_stride_steps) == (672, 96, 96)
    assert base.forecast.origin_time.isoformat() == "23:45:00"
    assert base.split.purge_overlapping_horizons
    assert base.covariates.future_weather_source == "oracle"
    assert base.covariates.weather == ["Temperature_avg_hourly"]
    assert base.pit.mode == "discretized" and base.pit.monotone_repair == "isotonic"
    assert base.pit.dependence_transform == "training_frequency" and base.pit.eps == 1e-7


def test_tracked_smoke_expands_m0_through_m3_without_location() -> None:
    from simcast.cli.run_composite import expand_methods
    from simcast.config import load_composite_config, load_evaluation_config, resolve_run_config

    root = Path(__file__).parents[1]
    path = root / "configs/venues/lab/heapo_smoke.yaml"
    composite = load_composite_config(path)
    fits = expand_methods(path, composite)
    assert [fit.method_id for fit in fits] == ["m0", "m1", "m2", "m3"]
    for fit in fits:
        runtime = resolve_run_config(fit.base, fit.method)
        if runtime.features is not None:
            assert runtime.features.use_location is False
    evaluation = load_evaluation_config(root / "configs/evaluations/heapo_smoke.yaml")
    assert evaluation.evaluation.cross_entity_statistic == "sum"
    assert {"energy_score", "variogram_score", "test_pseudo_nll"} <= set(evaluation.metrics)


def test_adapter_enforces_group_and_weather_mapping(tmp_path: Path) -> None:
    _source(tmp_path)
    with pytest.raises(ValueError, match="control/treatment"):
        heapo.load_adapter(_config(tmp_path, ids=(101, 606)))
    assert heapo.load_adapter(_config(tmp_path, ids=(606,), group="treatment")).group.entity_ids == ["household::606"]
    assert len(heapo.load_adapter(_config(tmp_path, ids=(101, 606), group="all")).group) == 2
    with pytest.raises(ValueError, match="Weather_ID"):
        heapo.load_adapter(_config(tmp_path, ids=(101, 707)))


def test_selection_uses_stable_hash_not_coverage_and_filters_groups(tmp_path: Path) -> None:
    _source(tmp_path)
    households = read_households(tmp_path / "meta_data/households.csv")
    chosen = heapo.select_households(households, "8jB", maximum=3)
    assert chosen == heapo.select_households(households.iloc[::-1], "8jB", maximum=3)
    assert chosen == sorted(chosen) and len(chosen) == 3
    assert set(chosen) <= {101, 202, 303, 404, 505}
    assert heapo.select_households(households, "8jB", group="treatment") == [606]
    assert heapo.select_households(households, "8jB", group="all") == [101, 202, 303, 404, 505, 606]
    households["lifetime_days"] = np.arange(len(households))[::-1]
    assert heapo.select_households(households, "8jB", maximum=3) == chosen
    assert heapo.select_households(households, "8jB", maximum=3, seed=13) != chosen


def test_hourly_weather_uses_bounded_hold_not_future_interpolation() -> None:
    hours = pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T02:00Z", "2024-01-01T03:00Z"])
    weather = pd.DataFrame({"Temperature_avg_hourly": [1.0, 9.0, np.nan]}, index=hours)
    grid = pd.date_range("2024-01-01", periods=17, freq="15min", tz="UTC")
    aligned = heapo.align_hourly_weather(weather, grid)
    np.testing.assert_array_equal(aligned.iloc[:4, 0], [1.0] * 4)
    assert aligned.iloc[4:8, 0].isna().all()
    np.testing.assert_array_equal(aligned.iloc[8:12, 0], [9.0] * 4)
    assert aligned.iloc[12:, 0].isna().all()


def test_adapter_uses_oracle_weather_and_excludes_weather_gaps(tmp_path: Path) -> None:
    from simcast.config import CovariatesConfig
    from simcast.data.windows import make_forecast_window

    _source(tmp_path)
    config = _config(tmp_path, ids=(101, 303))
    adapter = heapo.load_adapter(config)
    assert isinstance(adapter, heapo.FrameDatasetAdapter)
    window = make_forecast_window("2024-01-01T05:45Z", lookback_steps=4, horizon_steps=2)
    prepared = adapter.prepare_window(window)
    assert prepared.forecastable_entity.all()
    future = prepared.inputs[0]["future_covariates"]["Temperature_avg_hourly"]
    past = prepared.inputs[0]["past_covariates"]["Temperature_avg_hourly"]
    assert future[0] > past[-1]
    source = tmp_path / "weather_data/hourly/8jB.csv"
    weather = pd.read_csv(source, sep=";")
    weather = weather.loc[~weather["Timestamp"].str.startswith("2024-01-01 06:00")]
    weather.to_csv(source, sep=";", index=False)
    assert not heapo.load_adapter(config).prepare_window(window).forecastable_entity.any()
    non_oracle = config.model_copy(update={"covariates": CovariatesConfig(future_weather_source="vintage")})
    with pytest.raises(ValueError, match="oracle"):
        heapo.load_adapter(non_oracle)


def test_source_diagnostics_match_generic_adapter_forecastability(tmp_path: Path) -> None:
    from simcast.data.heapo_diagnostics import _availability, source_diagnostics

    _source(tmp_path)
    adapter = heapo.load_adapter(_config(tmp_path))
    assert isinstance(adapter, heapo.FrameDatasetAdapter)
    windows, forecast, empty, active = _availability(adapter)
    for index, window in enumerate(windows):
        prepared = adapter.prepare_window(window)
        np.testing.assert_array_equal(forecast[index], prepared.forecastable_entity)
        np.testing.assert_array_equal(empty[index], prepared.empty_history)
        np.testing.assert_array_equal(active[index], forecast[index, :, None] & np.isfinite(prepared.true_y))
    report = source_diagnostics(adapter)
    assert report["canonical_K_max"] == 5
    training = report["potential_training_diagnostics"]
    assert training["households_without_training_observations"] == ["household::202"]
    assert training["training_frequency_fallback"]["fallback_count"] == 2
    assert training["zero_overlap_pairs_by_lead"] == [4, 4]
    assert report["source_label_active_K"]["overall"]["min"] >= 2
    assert report["source_label_active_K"]["overall"]["max"] == 5


def test_standard_download_workflow_verifies_archive_and_extracts_selected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    import yaml

    from simcast.cli.download_data import download_data

    root = tmp_path / "data"
    (root / "archives").mkdir(parents=True)
    archive = root / "archives/heapo_data.zip"
    _archive(archive)
    monkeypatch.setattr(heapo, "ARCHIVE_SIZE", archive.stat().st_size)
    monkeypatch.setattr(heapo, "ARCHIVE_MD5", hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest())
    runtime = _config(root, ids=(1,))
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({
        "kind": "base", "id": "heapo_test", "data": runtime.data.model_dump(mode="json"),
        "protocol": runtime.protocol.model_dump(mode="json"),
    }))
    assert download_data(base) == root
    assert (root / "smart_meter_data/15min/1.csv").is_file()
    assert (root / "weather_data/hourly/8jB.csv").is_file()
    assert not (root / "smart_meter_data/15min/2.csv").exists()


def test_download_checksum_failure_does_not_promote_corrupt_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_curl(command, *, check):
        assert check and "--continue-at" in command
        (tmp_path / "archives/heapo_data.zip.part").write_bytes(b"abd")

    monkeypatch.setattr(heapo.subprocess, "run", fake_curl)
    monkeypatch.setattr(heapo, "ARCHIVE_SIZE", 3)
    monkeypatch.setattr(heapo, "ARCHIVE_MD5", "900150983cd24fb0d6963f7d28e17f72")
    with pytest.raises(ValueError, match="size/MD5"):
        heapo._download_archive(tmp_path)
    assert not (tmp_path / "archives/heapo_data.zip").exists()
    assert not (tmp_path / "archives/heapo_data.zip.part").exists()


def test_control_filter_rejects_smart_meter_group_disagreement(tmp_path: Path) -> None:
    _source(tmp_path)
    path = tmp_path / "smart_meter_data/15min/101.csv"
    path.write_text(path.read_text().replace(";control;", ";treatment;"))
    with pytest.raises(ValueError, match="disagrees"):
        heapo.load_adapter(_config(tmp_path))


def test_utc_offsets_are_normalized_without_treating_ids_as_coordinates() -> None:
    target = heapo.read_target(_meter(
        "101;unknown;control;2024-03-31 03:00:00+02:00;1.0;;\n"
        "101;unknown;control;2024-03-31 03:15:00+02:00;;;\n"
    ), 101)
    assert target.index[0] == pd.Timestamp("2024-03-31T01:15Z")
    assert target.index[1] == pd.Timestamp("2024-03-31T01:30Z")
    assert pd.isna(target.iloc[1, 0])


def test_location_features_are_explicitly_rejected(tmp_path: Path) -> None:
    _source(tmp_path)
    payload = _config(tmp_path).model_dump(mode="json")
    payload["dependence"] = {"method": "conditional_low_rank", "model": {"latent_rank": 2}}
    payload["features"] = {"use_location": True}
    payload["training"] = {"epochs": 1, "patience": 1}
    with pytest.raises(ValueError, match="use_location: false"):
        heapo.load_adapter(ResolvedExperimentConfig.model_validate(payload))


def test_missing_day_blocks_are_not_filled_or_padded() -> None:
    times = pd.date_range("2024-01-01", periods=4 * 96, freq="15min", tz="UTC")
    rows = "".join(f"101;unknown;control;{time};1.0;;\n" for time in times if time.day != 2)
    target = heapo.read_target(_meter(rows), 101)
    assert len(target) == 3 * 96
    assert target.index.min() == pd.Timestamp("2024-01-01T00:15Z")
    assert target.index.max() == pd.Timestamp("2024-01-05T00:00Z")
    missing_day = pd.date_range("2024-01-02T00:15Z", periods=96, freq="15min")
    assert target.reindex(missing_day)[heapo.TARGET_COLUMN].isna().all()


def test_irregular_household_cohort_cache_fit_and_evaluate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from _staggered_fixture import RecordingForecaster

    import simcast.cli.build_cache as cache_cli
    import simcast.cli.train_dependence as train_cli
    from simcast.cli.evaluate import evaluate_from_config
    from simcast.data.heapo_diagnostics import cache_diagnostics, fitted_m1_diagnostics
    from simcast.fm.cache import load_pit_library
    from simcast.training.dataset import DependenceCollator

    _source(tmp_path)
    config = _config(tmp_path)
    forecaster = RecordingForecaster()
    cache = cache_cli.build_cache_from_config(config, forecaster=forecaster, output_dir=tmp_path / "cache")
    sealed = load_pit_library(cache)
    full = load_pit_library(cache, access="evaluation")
    ids = config.protocol.ordered_entity_ids
    assert full.dataset["entity_id"].values.tolist() == ids
    split = full.dataset["split"].values
    assert np.isnan(sealed.dataset["entity_valid"].values[split == "test"]).all()
    forecastable = full.dataset["forecastable_entity"].values.astype(bool)
    assert not forecastable[0, 1] and forecastable[-1, 1]
    assert forecastable[0, 2] and not forecastable[-1, 2]
    assert not forecastable[0, 4] and not forecastable[-1, 4] and forecastable[:, 4].any()
    assert min(min(counts) for counts in forecaster.finite_history_counts) == 1
    for call in forecaster.calls:
        assert call == [entity for entity in ids if entity in call]
    assert sum(map(len, forecaster.calls)) == int(forecastable.sum())

    batches = []

    class RecordingCollator(DependenceCollator):
        def __call__(self, samples):
            batch = super().__call__(samples)
            batches.append(batch)
            return batch

    monkeypatch.setattr(train_cli, "DependenceCollator", RecordingCollator)
    models = {
        "independent": None,
        "static_gaussian": {},
        "conditional_low_rank": {"hidden_dims": [8], "latent_rank": 2, "dropout": 0.0},
        "set_aware_low_rank": {
            "model_dim": 8, "num_layers": 1, "num_heads": 2, "latent_rank": 2, "dropout": 0.0,
        },
    }
    runs = {}
    for method, model in models.items():
        payload = config.model_dump(mode="json")
        payload["dependence"] = {"method": method, "model": model}
        if method in ("conditional_low_rank", "set_aware_low_rank"):
            payload["features"] = {"use_location": False}
            payload["training"] = {"batch_size": 8, "epochs": 2, "patience": 1}
        runtime = ResolvedExperimentConfig.model_validate(payload)
        runs[method] = train_cli.train_from_config(runtime, cache_dir=cache, output_dir=tmp_path / method)
        with np.load(runs[method] / "pit_training_frequency_map.npz") as mapping:
            expected = np.zeros((5, 2), dtype=bool)
            expected[1] = True
            np.testing.assert_array_equal(mapping["fallback_mask"], expected)
            assert int(mapping["fallback_count"]) == 2
    assert batches and max(batch.features.shape[1] for batch in batches) == 4
    assert set(torch.cat([batch.active_counts for batch in batches]).tolist()) == {2, 3, 4}
    m1 = fitted_m1_diagnostics(runs["static_gaussian"], ids)
    assert m1["estimator_mode"] == "pairwise_complete"
    assert m1["independent_pairs_after_psd_repair_by_lead"] == [4, 4]
    assert (np.asarray(m1["pairwise_overlap_counts"])[:, 1] == 0).all()
    before_test = cache_diagnostics(cache, config)
    assert before_test["active_K"]["by_split"]["test"]["count"] == 0
    assert before_test["pit_training_frequency_fallback"]["fallback_count"] == 2

    payload = config.model_dump(mode="json")
    payload["sampling"] = {"num_samples": 32}
    payload["evaluation"] = {
        "quantile_levels": [0.1, 0.5, 0.9], "interval_levels": [0.8],
        "joint_score_num_samples": 8, "scenario_batch_size": 4, "cross_entity_statistic": "sum",
    }
    output = evaluate_from_config(
        ResolvedExperimentConfig.model_validate(payload), methods=tuple(models),
        method_runs=runs, cache_dir=cache, output_dir=tmp_path / "evaluation",
    )
    test = full.test_data()
    expected_active = (
        test["forecastable_entity"].values.astype(bool)[:, :, None] & test["entity_valid"].values.astype(bool)
    )
    for method in models:
        with np.load(output / f"{method}_aggregate_predictions.npz") as saved:
            np.testing.assert_array_equal(saved["active_entity_mask"], expected_active)
            np.testing.assert_array_equal(saved["valid"], expected_active.sum(axis=1) >= 2)
            assert np.isfinite(saved["pseudo_nll"][saved["valid"]]).all()
    cases = pd.read_parquet(output / "per_origin_lead_metrics.parquet")
    row = cases.loc[(cases["method"] == "independent") & cases["valid"]].iloc[0]
    test_index = list(np.flatnonzero(split == "test")).index(row["origin_index"])
    lead = int(row["lead"]) - 1
    truth = test["true_y"].values[test_index, expected_active[test_index, :, lead], lead].sum()
    assert row["observed_aggregate"] == pytest.approx(truth)
    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    assert manifest["cardinality_mode"] == "variable" and manifest["group"]["entity_count"] == 5
    assert manifest["common_random_numbers"]
    assert manifest["active_entity_count"]["min"] == 2 and manifest["active_entity_count"]["max"] == 5
    final = cache_diagnostics(cache, config, include_test_labels=True)
    assert final["active_K"]["by_split"]["test"]["count"] > 0
    assert final["m1_training_overlap"]["zero_overlap_pairs_by_lead"] == [4, 4]
    assert final["pit_training_frequency_fallback"] == before_test["pit_training_frequency_fallback"]


def test_actual_v1_smart_meter_schema_and_missingness() -> None:
    frame = read_smart_meter(_meter(
        "8790869;unknown;control;2022-05-07 00:00:00+00:00;0.0;;\n"
        "8790869;unknown;control;2022-05-07 00:15:00+00:00;;;\n"
        "8790869;unknown;control;2022-05-08 00:00:00+00:00;-0.1;;\n"
    ), 8790869)
    assert str(frame.index.tz) == "UTC"
    assert len(frame) == 3
    assert frame["kWh_received_Total"].iloc[-1] == -0.1
    summary = target_summary(frame)
    assert summary["finite_quarter_hours"] == 2
    assert summary["missing_values"] == 1
    assert summary["absent_quarter_hours"] == 94


def test_completed_interval_values_are_not_available_at_left_boundary() -> None:
    from simcast.data.availability import select_available_past_targets

    target = read_target(_meter(
        "8790869;unknown;control;2022-05-07 23:45:00+00:00;1.75;;\n"
        "8790869;unknown;control;2022-05-08 00:00:00+00:00;2.25;;\n"
    ), 8790869)
    assert target.index.tolist() == [pd.Timestamp("2022-05-08T00:00Z"), pd.Timestamp("2022-05-08T00:15Z")]
    available = select_available_past_targets(
        target, pd.Timestamp("2022-05-08T00:00Z"), target_column="kWh_received_Total",
    )
    assert available.iloc[0] == 1.75
    assert pd.Timestamp("2022-05-08T00:15Z") not in available.index
    assert target["kWh_received_Total"].tolist() == [1.75, 2.25]


def test_completed_interval_daily_origin_has_first_forecast_ending_at_midnight() -> None:
    from simcast.data.windows import make_forecast_window

    window = make_forecast_window("2022-05-07T23:45Z", lookback_steps=672, horizon_steps=96)
    assert window.past_timestamps[-1] == pd.Timestamp("2022-05-07T23:45Z")
    assert window.future_timestamps[0] == pd.Timestamp("2022-05-08T00:00Z")
    assert window.future_timestamps[-1] == pd.Timestamp("2022-05-08T23:45Z")


@pytest.mark.parametrize("identifier", ["household::0", "household::01", "household::1.0", "../1", "1"])
def test_household_ids_reject_ambiguous_or_unsafe_forms(identifier: str) -> None:
    with pytest.raises(ValueError, match="invalid HEAPO entity ID"):
        household_number(identifier)


def test_household_ids_and_raw_validation() -> None:
    assert household_number("household::8790869") == 8790869
    row = "8790869;unknown;control;2022-05-07 00:00:00+00:00;0.0;;\n"
    with pytest.raises(ValueError, match="duplicate"):
        read_smart_meter(_meter(row * 2), 8790869)
    with pytest.raises(ValueError, match="ID mismatch"):
        read_smart_meter(_meter(row), 1)
    with pytest.raises(ValueError, match="missing columns"):
        read_smart_meter(io.BytesIO(b"Household_ID;Timestamp\n1;x\n"), 1)


def test_actual_household_group_encoding_and_weather_mapping() -> None:
    rows = (
        "Household_ID;Group;Weather_ID;SmartMeterData_Available_15min\n"
        "661091;treatment;MqO;True\n120701;control;Hg;True\n24611;control;wDD;False\n"
    )
    frame = read_households(io.BytesIO(rows.encode()))
    assert frame["Household_ID"].tolist() == [24611, 120701, 661091]
    assert frame.set_index("Household_ID").loc[661091, "Weather_ID"] == "MqO"
    with pytest.raises(ValueError, match="Group"):
        read_households(io.BytesIO(rows.replace("control", "Control").encode()))


def test_archive_checksum_rejects_wrong_release(tmp_path: Path) -> None:
    path = tmp_path / "heapo_data.zip"
    path.write_bytes(b"not the official archive")
    with pytest.raises(ValueError, match="size/MD5"):
        verify_archive(path)


def test_archive_checksum_checks_content_not_just_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "heapo_data.zip"
    path.write_bytes(b"abc")
    monkeypatch.setattr(heapo, "ARCHIVE_SIZE", 3)
    monkeypatch.setattr(heapo, "ARCHIVE_MD5", "900150983cd24fb0d6963f7d28e17f72")
    verify_archive(path)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="size/MD5"):
        verify_archive(path)


@pytest.mark.parametrize(("old", "new", "error"), [
    ("control", "Control", "Group"),
    ("00:00:00", "00:01:00", "off-grid"),
    (";0.0;;", ";inf;;", "infinite"),
])
def test_invalid_source_values_fail_explicitly(old: str, new: str, error: str) -> None:
    row = "8790869;unknown;control;2022-05-07 00:00:00+00:00;0.0;;\n"
    with pytest.raises(ValueError, match=error):
        read_smart_meter(_meter(row.replace(old, new)), 8790869)


@pytest.mark.parametrize("name", [
    "../escape.csv", "/tmp/escape.csv", "heapo_data/../escape.csv", "heapo_data\\escape.csv",
])
def test_archive_inspection_rejects_traversal(tmp_path: Path, name: str) -> None:
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, b"unsafe")
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="unsafe"):
        archive_members(archive)


def test_archive_layout_and_symlinks(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in SMALL_FILES:
            archive.writestr(f"heapo_data/{name}", "column\n")
    with zipfile.ZipFile(path) as archive:
        assert set(archive_members(archive)) == set(SMALL_FILES)
    with zipfile.ZipFile(path, "a") as archive:
        info = zipfile.ZipInfo("heapo_data/link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "/etc/passwd")
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="unsafe"):
        archive_members(archive)


def _archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in SMALL_FILES:
            data = "column\n"
            if name == "meta_data/households.csv":
                data = (
                    "Household_ID;Group;Weather_ID;SmartMeterData_Available_15min\n"
                    "1;control;8jB;True\n2;treatment;Hg;True\n"
                )
            archive.writestr(f"heapo_data/{name}", data)
        for name in [
            "smart_meter_data/15min/1.csv", "smart_meter_data/15min/2.csv",
            "weather_data/hourly/8jB.csv", "weather_data/hourly/Hg.csv",
            "smart_meter_data/daily/1.csv", "reports/protocols.csv",
        ]:
            archive.writestr(f"heapo_data/{name}", "unused\n")


def test_selective_extraction_keeps_only_requested_households_and_weather(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    _archive(path)
    root = tmp_path / "selected"
    paths = safe_extract_selected(path, root, [1])
    assert {str(path.relative_to(root)) for path in paths} == {
        *SMALL_FILES, "smart_meter_data/15min/1.csv", "weather_data/hourly/8jB.csv",
    }
    assert not (root / "smart_meter_data/15min/2.csv").exists()
    assert not (root / "weather_data/hourly/Hg.csv").exists()
    assert not (root / "reports").exists()
    with pytest.raises(ValueError, match="unknown"):
        safe_extract_selected(path, root, [3])


def test_extraction_rejects_existing_symlink_parent(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    _archive(path)
    root = tmp_path / "selected"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "meta_data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract_selected(path, root, [1])
    assert not list(outside.iterdir())
