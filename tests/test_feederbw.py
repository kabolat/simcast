from __future__ import annotations

import stat
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

import simcast.cli.build_cache as cache_cli
from simcast.config import IndependentMethodConfig, base_fingerprint, load_base_config, resolve_run_config
from simcast.data.feederbw import (
    METADATA_DIAGNOSTIC_FIELDS,
    WEATHER_FEATURES,
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
                "active_power_kW": np.full(len(target_times), feeder / 10, dtype=np.float32),
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


def test_download_patterns_include_only_selected_measurement_batches() -> None:
    base = load_base_config(BASE_CONFIG)
    assert allow_patterns(base.data, entity_ids=base.protocol.ordered_entity_ids) == [
        "feeder_metadata.csv",
        "weather_data.parquet",
        "feeder_measurement_data_041-080.zip",
        "feeder_measurement_data_081-120.zip",
        "feeder_measurement_data_161-200.zip",
    ]


def test_download_patterns_can_include_all_measurement_batches() -> None:
    base = load_base_config(BASE_CONFIG)
    patterns = allow_patterns(base.data, entity_ids=base.protocol.ordered_entity_ids, all_feeders=True)
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
    assert pd.isna(strict.iloc[0, 0])
    assert strict.iloc[1, 0] == pytest.approx(19)
    assert permissive.iloc[0, 0] == pytest.approx(5.5)
    assert stats["missing_source_timestamps"] == 3
    assert stats["valid_intervals"] == 1


def test_adapter_builds_group_and_never_passes_future_weather(tmp_path: Path) -> None:
    _write_probe_data(tmp_path)
    base = _base_config(tmp_path)
    run = resolve_run_config(base, IndependentMethodConfig(kind="method", id="m0", family="independent"))
    adapter = adapter_spec("feederbw").load(run)

    diagnostics = adapter.diagnostics()
    assert diagnostics["entity_ids"] == base.protocol.ordered_entity_ids
    assert diagnostics["common_complete_interval_fraction"] == pytest.approx(1.0)
    assert diagnostics["metadata"]["latest_record_per_feeder"][0]["date"].startswith("2024-02-01")
    assert diagnostics["weather"]["future_weather_supplied"] is False

    adapter.frames = [
        replace(
            frame,
            future_weather=pd.DataFrame(
                {"air_temperature_C": np.full(len(frame.target), 9999, dtype=np.float32)},
                index=frame.target.index,
            ),
        )
        for frame in adapter.frames
    ]
    window = make_forecast_window(
        "2024-01-12T23:45:00Z",
        lookback_steps=base.forecast.lookback_steps,
        horizon_steps=base.forecast.horizon_steps,
        frequency_minutes=base.forecast.frequency_minutes,
        entity_ids=base.protocol.ordered_entity_ids,
    )
    prepared = adapter.prepare_window(window)
    assert prepared.true_y.shape == (5, 96)
    for model_input in prepared.inputs:
        assert set(base.covariates.past_weather).issubset(model_input["past_covariates"])
        assert "air_temperature_C" not in model_input["future_covariates"]
        assert all(np.isfinite(values).all() for values in model_input["future_covariates"].values())


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
    assert library.metadata["covariates"]["future_weather_source"] == "none"
    assert (destination / "marginal_diagnostics.json").is_file()


def test_past_weather_changes_the_marginal_fingerprint(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    changed_covariates = base.covariates.model_copy(update={"past_weather": ["air_temperature_C"]})
    changed = base.model_copy(update={"covariates": changed_covariates})
    assert base_fingerprint(base) != base_fingerprint(changed)
