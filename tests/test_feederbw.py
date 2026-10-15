from __future__ import annotations

import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

import simcast.cli.build_cache as cache_cli
from simcast.config import (
    CovariatesConfig,
    IndependentMethodConfig,
    base_fingerprint,
    load_base_config,
    load_download_config,
    resolve_run_config,
)
from simcast.data.adapter import IneligibleWindow
from simcast.data.feederbw import (
    METADATA_DIAGNOSTIC_FIELDS,
    WEATHER_FEATURES,
    _align_hourly_weather,
    allow_patterns,
    feeders_for_substation,
    read_metadata,
    resample_active_power,
    safe_extract_selected,
)
from simcast.data.registry import adapter_spec
from simcast.data.windows import make_forecast_window
from simcast.fm.cache import load_pit_library

ROOT = Path(__file__).parents[1]
BASE_CONFIG = ROOT / "configs/bases/feederbw/substation_52.yaml"
ALL_FEEDERS_BASE = ROOT / "configs/bases/feederbw/all_feeders.yaml"
FULL_DOWNLOAD_CONFIG = ROOT / "configs/downloads/feederbw_all.yaml"
FEEDERS = [60, 106, 109, 165, 170]


@dataclass
class _MockForecast:
    entity_ids: list[str]
    quantile_levels: torch.Tensor
    quantile_predictions: torch.Tensor
    forecast_embeddings: torch.Tensor
    output_patch_size: int


class _MockForecaster:
    def predict(
        self,
        entity_ids: list[str],
        inputs: list[dict[str, object]],
        prediction_length: int,
        *,
        batch_size: int = 256,
        context_length: int | None = None,
    ) -> _MockForecast:
        del batch_size, context_length
        center = torch.tensor([float(np.asarray(item["target"])[-1]) for item in inputs])
        center = center[:, None].expand(-1, prediction_length)
        predictions = torch.stack(
            (center - 1.5, center - 1, center - 0.5, center, center + 0.5, center + 1, center + 1.5),
            dim=-1,
        )
        return _MockForecast(
            entity_ids=list(entity_ids),
            quantile_levels=torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]),
            quantile_predictions=predictions,
            forecast_embeddings=torch.ones((len(entity_ids), 1, 4)),
            output_patch_size=2,
        )


def _base_config(local_dir: Path):
    base = load_base_config(BASE_CONFIG)
    return base.model_copy(update={"data": base.data.model_copy(update={"local_dir": str(local_dir)})})


def _write_probe_data(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "measurements").mkdir()
    metadata_rows: list[dict[str, Any]] = []
    for feeder in FEEDERS:
        metadata_rows.append(
            {
                "feeder": feeder,
                "substation": 52,
                "date": "2024-01-01",
                **{field: float(feeder) for field in METADATA_DIAGNOSTIC_FIELDS},
            }
        )
    metadata_rows.append(
        {
            "feeder": FEEDERS[0],
            "substation": 52,
            "date": "2024-02-01",
            **{field: float(feeder + 1) for field in METADATA_DIAGNOSTIC_FIELDS},
        }
    )
    pd.DataFrame(metadata_rows).to_csv(root / "feeder_metadata.csv", index=False)

    target_times = pd.date_range("2024-01-01", periods=21 * 24 * 60, freq="min", tz="UTC")
    weather_times = pd.date_range("2023-12-31", periods=22 * 24, freq="h", tz="UTC")
    weather_rows: list[pd.DataFrame] = []
    for feeder in FEEDERS:
        pd.DataFrame(
            {
                "feeder": feeder,
                "timestamp_UTC": target_times,
                "active_power_kW": np.arange(len(target_times), dtype=np.float32) % 60 + feeder / 10,
            }
        ).to_parquet(root / "measurements" / f"feeder_{feeder:03d}.parquet", index=False)
        weather_rows.append(
            pd.DataFrame(
                {
                    "feeder": feeder,
                    "timestamp_UTC": weather_times,
                    **{
                        feature: np.full(len(weather_times), feeder + index, dtype=np.float32)
                        for index, feature in enumerate(WEATHER_FEATURES)
                    },
                }
            )
        )
    pd.concat(weather_rows, ignore_index=True).to_parquet(root / "weather_data.parquet", index=False)


def _write_all_feeders_probe_data(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "measurements").mkdir()
    target_times = pd.date_range("2024-01-01", periods=15, freq="min", tz="UTC")
    weather_time = pd.Timestamp("2024-01-01T00:00:00Z")
    metadata_rows: list[dict[str, Any]] = []
    weather_rows: list[pd.DataFrame] = []
    for feeder in range(1, 201):
        substation = (feeder - 1) % 121 + 1
        metadata_rows.append(
            {
                "feeder": feeder,
                "substation": substation,
                "date": "2024-01-01",
                **{field: float(feeder) for field in METADATA_DIAGNOSTIC_FIELDS},
            }
        )
        pd.DataFrame(
            {
                "feeder": feeder,
                "timestamp_UTC": target_times,
                "active_power_kW": np.full(len(target_times), feeder / 10, dtype=np.float32),
            }
        ).to_parquet(root / "measurements" / f"feeder_{feeder:03d}.parquet", index=False)
        weather_rows.append(
            pd.DataFrame(
                {
                    "feeder": [feeder],
                    "timestamp_UTC": [weather_time],
                    **{feature: [float(feeder)] for feature in WEATHER_FEATURES},
                }
            )
        )
    pd.DataFrame(metadata_rows).to_csv(root / "feeder_metadata.csv", index=False)
    pd.concat(weather_rows, ignore_index=True).to_parquet(root / "weather_data.parquet", index=False)


def test_v11_metadata_schema_and_deterministic_substation_group(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "feeder": [60, 106, 109, 165, 170, 60],
            "substation": [52, 52, 52, 52, 52, 52],
            "date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01", "2024-02-01"],
            **{field: [1, 2, 3, 4, 5, 6] for field in METADATA_DIAGNOSTIC_FIELDS},
        }
    ).to_csv(path, index=False)
    metadata = read_metadata(path)
    assert feeders_for_substation(metadata, 52) == FEEDERS
    assert isinstance(metadata["date"].dtype, pd.DatetimeTZDtype)
    assert str(metadata["date"].dtype.tz) == "UTC"

    wrong_schema = pd.read_csv(path).rename(columns={"substation": "substation_id"})
    wrong_path = tmp_path / "old-schema.csv"
    wrong_schema.to_csv(wrong_path, index=False)
    with pytest.raises(ValueError, match="substation"):
        read_metadata(wrong_path)


def test_all_feeders_base_loads_complete_group_across_substations(tmp_path: Path) -> None:
    _write_all_feeders_probe_data(tmp_path)
    base = load_base_config(ALL_FEEDERS_BASE)
    base = base.model_copy(update={"data": base.data.model_copy(update={"local_dir": str(tmp_path)})})
    run = resolve_run_config(base, IndependentMethodConfig(kind="method", id="m0", family="independent"))
    adapter = adapter_spec("feederbw").load(run)

    assert base.protocol.entity_count == 200
    assert adapter.group.group_id == "feederbw_all_feeders"
    assert adapter.group.entity_ids == [f"feederbw::feeder_{feeder:03d}" for feeder in range(1, 201)]
    assert adapter.group.metadata["selection_scope"] == "all_feeders"
    assert adapter.group.metadata["substation_id"] is None
    assert adapter.group.metadata["substation_ids"] == list(range(1, 122))
    assert adapter.group.entities[0].extra["substation"] == 1
    assert adapter.group.entities[121].extra["substation"] == 1
    assert adapter.diagnostics()["entity_count"] == 200
    assert adapter.diagnostics()["selected_substation_id"] is None
    assert adapter.provenance["selection_scope"] == "all_feeders"
    assert adapter.provenance["selected_substation_ids"] == list(range(1, 122))
    assert adapter_spec("feederbw").default_cache_name(base.data) == "feederbw_all_feeders"


def test_metadata_dates_use_german_local_midnight_before_utc_conversion(tmp_path: Path) -> None:
    path = tmp_path / "metadata-dates.csv"
    pd.DataFrame(
        {
            "feeder": [60, 60],
            "substation": [52, 52],
            "date": ["2024-01-01", "2024-07-01"],
            **{field: [1.0, 2.0] for field in METADATA_DIAGNOSTIC_FIELDS},
        }
    ).to_csv(path, index=False)

    metadata = read_metadata(path)

    assert metadata["date"].tolist() == [
        pd.Timestamp("2023-12-31T23:00:00Z"),
        pd.Timestamp("2024-06-30T22:00:00Z"),
    ]


def test_download_patterns_include_only_selected_measurement_batches() -> None:
    base = load_base_config(BASE_CONFIG)
    assert allow_patterns(base.data, entity_ids=base.protocol.ordered_entity_ids) == [
        "feeder_metadata.csv",
        "weather_data.parquet",
        "feeder_measurement_data_041-080.zip",
        "feeder_measurement_data_081-120.zip",
        "feeder_measurement_data_161-200.zip",
    ]


def test_dataset_wide_download_profile_selects_all_measurement_batches() -> None:
    config = load_download_config(FULL_DOWNLOAD_CONFIG)
    patterns = allow_patterns(config.data)
    assert patterns == [
        "feeder_metadata.csv",
        "weather_data.parquet",
        "feeder_measurement_data_001-040.zip",
        "feeder_measurement_data_041-080.zip",
        "feeder_measurement_data_081-120.zip",
        "feeder_measurement_data_121-160.zip",
        "feeder_measurement_data_161-200.zip",
    ]


def test_generic_downloader_uses_feederbw_selective_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from simcast.cli.download_data import download_data as download_config
    from simcast.data import feederbw

    downloaded: list[str] = []
    extracted: list[tuple[str, list[int]]] = []

    def fake_download_asset(name: str, destination: Path) -> Path:
        downloaded.append(name)
        return destination

    def fake_extract_selected(archive: Path, feeder_ids: list[int], destination: Path) -> list[Path]:
        extracted.append((archive.name, feeder_ids))
        return []

    monkeypatch.setattr(feederbw, "_download_asset", fake_download_asset)
    monkeypatch.setattr(feederbw, "safe_extract_selected", fake_extract_selected)
    result = download_config(
        BASE_CONFIG,
        overrides=[f"data.local_dir={tmp_path / 'download'}"],
        snapshot_download_fn=lambda **_: pytest.fail("FeederBW must not use Hugging Face downloads"),
    )

    assert result == tmp_path / "download"
    assert downloaded == [
        "feeder_metadata.csv",
        "weather_data.parquet",
        "feeder_measurement_data_041-080.zip",
        "feeder_measurement_data_081-120.zip",
        "feeder_measurement_data_161-200.zip",
    ]
    assert extracted == [
        ("feeder_measurement_data_041-080.zip", [60]),
        ("feeder_measurement_data_081-120.zip", [106, 109]),
        ("feeder_measurement_data_161-200.zip", [165, 170]),
    ]


def test_full_download_extracts_every_feeder_in_all_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from simcast.cli.download_data import download_data as download_config
    from simcast.data import feederbw

    downloaded: list[str] = []
    extracted: list[tuple[str, list[int]]] = []

    def fake_download_asset(name: str, destination: Path) -> Path:
        downloaded.append(name)
        return destination

    def fake_extract_selected(archive: Path, feeder_ids: list[int], destination: Path) -> list[Path]:
        extracted.append((archive.name, feeder_ids))
        return []

    monkeypatch.setattr(feederbw, "_download_asset", fake_download_asset)
    monkeypatch.setattr(feederbw, "safe_extract_selected", fake_extract_selected)
    download_config(
        FULL_DOWNLOAD_CONFIG,
        overrides=[f"data.local_dir={tmp_path / 'full-download'}"],
        snapshot_download_fn=lambda **_: pytest.fail("FeederBW must not use Hugging Face downloads"),
    )

    expected_batches = [
        "feeder_measurement_data_001-040.zip",
        "feeder_measurement_data_041-080.zip",
        "feeder_measurement_data_081-120.zip",
        "feeder_measurement_data_121-160.zip",
        "feeder_measurement_data_161-200.zip",
    ]
    assert downloaded == ["feeder_metadata.csv", "weather_data.parquet", *expected_batches]
    assert [name for name, _ in extracted] == expected_batches
    assert [feeder for _, batch in extracted for feeder in batch] == list(range(1, 201))


def test_zip_extraction_rejects_traversal_and_symlinks(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../outside.txt", "no")
        archive.writestr("feeder_060.parquet", b"data")
    with pytest.raises(ValueError, match="unsafe path"):
        safe_extract_selected(traversal, [60], tmp_path / "out")

    symlink = tmp_path / "symlink.zip"
    link_info = zipfile.ZipInfo("feeder_060.parquet")
    link_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(link_info, "target.parquet")
    with pytest.raises(ValueError, match="symlink"):
        safe_extract_selected(symlink, [60], tmp_path / "out")


def test_resampling_uses_utc_means_and_minimum_sample_threshold(tmp_path: Path) -> None:
    times = [*pd.date_range("2024-01-01T00:00:00Z", periods=12, freq="min"),
             *pd.date_range("2024-01-01T00:15:00Z", periods=15, freq="min")]
    path = tmp_path / "feeder_060.parquet"
    pd.DataFrame(
        {"feeder": 60, "timestamp_UTC": times, "active_power_kW": np.arange(27, dtype=np.float32)}
    ).to_parquet(path, index=False)

    strict, stats = resample_active_power(path, feeder_id=60, minimum_samples_per_interval=15)
    permissive, _ = resample_active_power(path, feeder_id=60, minimum_samples_per_interval=12)
    assert strict.index.tz is not None
    assert strict.index.tolist() == [pd.Timestamp("2024-01-01T00:15:00Z"), pd.Timestamp("2024-01-01T00:30:00Z")]
    assert pd.isna(strict.iloc[0, 0])
    assert strict.iloc[1, 0] == pytest.approx(19)
    assert permissive.iloc[0, 0] == pytest.approx(5.5)
    assert stats["missing_source_timestamps"] == 3
    assert stats["valid_intervals"] == 1


def test_resampling_labels_completed_intervals_at_the_right_boundary(tmp_path: Path) -> None:
    times = pd.date_range("2024-01-01T23:30:00Z", periods=45, freq="min")
    path = tmp_path / "feeder_060.parquet"
    pd.DataFrame(
        {"feeder": 60, "timestamp_UTC": times, "active_power_kW": np.arange(45, dtype=np.float32)}
    ).to_parquet(path, index=False)

    target, _ = resample_active_power(path, feeder_id=60)

    assert target.index.tolist() == [
        pd.Timestamp("2024-01-01T23:45:00Z"),
        pd.Timestamp("2024-01-02T00:00:00Z"),
        pd.Timestamp("2024-01-02T00:15:00Z"),
    ]
    assert target.loc[pd.Timestamp("2024-01-01T23:45:00Z"), "active_power_kW"] == pytest.approx(7)
    assert target.loc[pd.Timestamp("2024-01-02T00:00:00Z"), "active_power_kW"] == pytest.approx(22)
    assert target.loc[pd.Timestamp("2024-01-02T00:15:00Z"), "active_power_kW"] == pytest.approx(37)


def test_hourly_weather_forward_fill_is_bounded_by_one_source_interval() -> None:
    source_index = pd.DatetimeIndex(
        ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T03:00:00Z"],
        name="timestamp",
    )
    weather = pd.DataFrame({"temperature": [10.0, 11.0, 13.0]}, index=source_index)
    target_index = pd.date_range(source_index[0], source_index[-1], freq="15min", name="timestamp")

    aligned = _align_hourly_weather(weather, target_index)

    assert aligned.loc[pd.Timestamp("2024-01-01T00:15:00Z"), "temperature"] == pytest.approx(10)
    assert aligned.loc[pd.Timestamp("2024-01-01T00:45:00Z"), "temperature"] == pytest.approx(10)
    assert aligned.loc[pd.Timestamp("2024-01-01T01:00:00Z"), "temperature"] == pytest.approx(11)
    assert aligned.loc[pd.Timestamp("2024-01-01T02:00:00Z"), "temperature"] == pytest.approx(11)
    assert pd.isna(aligned.loc[pd.Timestamp("2024-01-01T02:15:00Z"), "temperature"])
    assert pd.isna(aligned.loc[pd.Timestamp("2024-01-01T02:45:00Z"), "temperature"])
    assert aligned.loc[pd.Timestamp("2024-01-01T03:00:00Z"), "temperature"] == pytest.approx(13)


@pytest.mark.parametrize(
    "origin",
    ["2024-01-09T23:45:00Z", "2024-01-10T23:45:00Z"],
    ids=["oracle-future-gap", "history-gap"],
)
def test_missing_hourly_weather_makes_affected_history_or_oracle_window_ineligible(
    tmp_path: Path, origin: str
) -> None:
    _write_probe_data(tmp_path)
    weather_path = tmp_path / "weather_data.parquet"
    weather = pd.read_parquet(weather_path)
    missing_row = (weather["feeder"] == FEEDERS[0]) & (
        pd.to_datetime(weather["timestamp_UTC"], utc=True) == pd.Timestamp("2024-01-10T00:00:00Z")
    )
    weather.loc[~missing_row].to_parquet(weather_path, index=False)

    base = _base_config(tmp_path)
    run = resolve_run_config(base, IndependentMethodConfig(kind="method", id="m0", family="independent"))
    adapter = adapter_spec("feederbw").load(run)
    window = make_forecast_window(
        origin,
        lookback_steps=base.forecast.lookback_steps,
        horizon_steps=base.forecast.horizon_steps,
        frequency_minutes=base.forecast.frequency_minutes,
        entity_ids=base.protocol.ordered_entity_ids,
    )

    with pytest.raises(IneligibleWindow, match="covariates unavailable"):
        adapter.prepare_window(window)


def test_adapter_oracle_uses_realized_future_weather(tmp_path: Path) -> None:
    _write_probe_data(tmp_path)
    base = _base_config(tmp_path)
    run = resolve_run_config(base, IndependentMethodConfig(kind="method", id="m0", family="independent"))
    adapter = adapter_spec("feederbw").load(run)

    diagnostics = adapter.diagnostics()
    assert diagnostics["entity_ids"] == base.protocol.ordered_entity_ids
    assert diagnostics["common_complete_interval_fraction"] == pytest.approx(1.0)
    assert diagnostics["metadata"]["latest_record_per_feeder"][0]["date"].startswith("2024-01-31T23:00:00")
    assert diagnostics["weather"]["future_weather_supplied"] is True
    assert diagnostics["weather"]["future_weather_source"] == "oracle"
    assert diagnostics["weather"]["maximum_age_minutes"] == 60
    assert "oracle experiment" in diagnostics["weather"]["future_weather_usage"]
    assert "not operational" in diagnostics["weather"]["future_weather_usage"]
    assert adapter.provenance["weather"] == diagnostics["weather"]
    assert diagnostics["target"]["timestamp_semantics"] == (
        "right boundary; target at t represents [t - 15 minutes, t)"
    )
    assert adapter.provenance["measurement"]["interval_label"] == "right boundary"
    assert adapter.provenance["measurement"]["interval_convention"] == (
        "target at t represents measurements in [t - 15 minutes, t)"
    )
    window = make_forecast_window(
        "2024-01-12T23:45:00Z",
        lookback_steps=base.forecast.lookback_steps,
        horizon_steps=base.forecast.horizon_steps,
        frequency_minutes=base.forecast.frequency_minutes,
        entity_ids=base.protocol.ordered_entity_ids,
    )
    assert window.past_timestamps[-1] == pd.Timestamp("2024-01-12T23:45:00Z")
    assert window.future_timestamps[0] == pd.Timestamp("2024-01-13T00:00:00Z")
    assert window.future_timestamps[-1] == pd.Timestamp("2024-01-13T23:45:00Z")
    assert len(window.future_timestamps) == 96
    assert window.future_timestamps[-1] - window.future_timestamps[0] + pd.Timedelta(minutes=15) == pd.Timedelta(
        hours=24
    )
    prepared = adapter.prepare_window(window)
    assert prepared.true_y.shape == (5, 96)
    for entity_index, (feeder_id, model_input) in enumerate(zip(FEEDERS, prepared.inputs, strict=True)):
        assert set(base.covariates.weather).issubset(model_input["past_covariates"])
        assert model_input["target"][-1] == pytest.approx(37 + feeder_id / 10)
        assert prepared.true_y[entity_index, 0] == pytest.approx(52 + feeder_id / 10)
        np.testing.assert_array_equal(
            model_input["future_covariates"]["air_temperature_C"],
            np.full(96, feeder_id + 2, dtype=np.float32),
        )
        assert all(np.isfinite(values).all() for values in model_input["future_covariates"].values())


def test_adapter_rejects_vintage_without_archived_forecasts(tmp_path: Path) -> None:
    _write_probe_data(tmp_path)
    base = _base_config(tmp_path)
    covariates = base.covariates.model_copy(update={"future_weather_source": "vintage"})
    run = resolve_run_config(
        base.model_copy(update={"covariates": covariates}),
        IndependentMethodConfig(kind="method", id="m0", family="independent"),
    )
    with pytest.raises(ValueError, match="no archived future forecasts"):
        adapter_spec("feederbw").load(run)


def test_synthetic_feederbw_cache_builds_without_chronos(tmp_path: Path) -> None:
    _write_probe_data(tmp_path / "source")
    base = _base_config(tmp_path / "source")
    run = resolve_run_config(base, IndependentMethodConfig(kind="method", id="m0", family="independent"))
    destination = cache_cli.build_cache_from_config(
        run,
        forecaster=_MockForecaster(),
        output_dir=tmp_path / "cache",
    )

    library = load_pit_library(destination)
    assert library.dataset.sizes["entity"] == 5
    assert library.dataset.sizes["origin"] > 0
    assert library.metadata["dataset"]["provenance"]["source"]["version"] == "v1.1"
    assert library.metadata["covariates"]["future_weather_source"] == "oracle"
    provenance = library.metadata["dataset"]["provenance"]
    assert provenance["measurement"]["interval_label"] == "right boundary"
    assert provenance["measurement"]["interval_convention"] == (
        "target at t represents measurements in [t - 15 minutes, t)"
    )
    assert "oracle experiment" in provenance["weather"]["future_weather_usage"]
    assert (destination / "marginal_diagnostics.json").is_file()


def test_weather_changes_the_marginal_fingerprint(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    changed_covariates = base.covariates.model_copy(update={"weather": ["air_temperature_C"]})
    changed = base.model_copy(update={"covariates": changed_covariates})
    assert base_fingerprint(base) != base_fingerprint(changed)


def test_past_weather_is_not_a_config_field() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CovariatesConfig.model_validate({"past_weather": ["air_temperature_C"]})
