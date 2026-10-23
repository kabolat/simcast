from __future__ import annotations

import io
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


def test_adapter_enforces_group_and_weather_mapping(tmp_path: Path) -> None:
    _source(tmp_path)
    with pytest.raises(ValueError, match="control/treatment"):
        heapo.load_adapter(_config(tmp_path, ids=(101, 606)))
    assert heapo.load_adapter(_config(tmp_path, ids=(606,), group="treatment")).group.entity_ids == ["household::606"]
    assert len(heapo.load_adapter(_config(tmp_path, ids=(101, 606), group="all")).group) == 2
    with pytest.raises(ValueError, match="Weather_ID"):
        heapo.load_adapter(_config(tmp_path, ids=(101, 707)))


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
