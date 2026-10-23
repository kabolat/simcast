"""HEAPO v1 household grid-received interval energy and station-cohort diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import IO, Any

import numpy as np
import pandas as pd

from simcast.config import (
    BaseExperimentConfig,
    DataConfig,
    DatasetDownloadConfig,
    DownloadDataConfig,
    ResolvedExperimentConfig,
)
from simcast.data.adapter import DatasetAdapter, EntityFrames, FrameDatasetAdapter
from simcast.types import EntityGroup, EntityMetadata

ADAPTER_NAME = "heapo"
DATASET_ID = "https://zenodo.org/records/15056919"
DATASET_REVISION = "v1"
DATASET_DOI = "10.5281/zenodo.15056919"
LICENSE = "CC-BY-4.0"
ARCHIVE_NAME = "heapo_data.zip"
ARCHIVE_SIZE = 458317017
ARCHIVE_MD5 = "2b92b0d52487e1a733c35ee59860450d"
ARCHIVE_PREFIX = "heapo_data/"
TARGET_COLUMN = "kWh_received_Total"
TARGET_LABEL_SHIFT = pd.Timedelta(minutes=15)
INTERVAL_CONVENTION = (
    "Raw interval boundaries are not established conclusively by v1 metadata, loader, or daily totals. "
    "Conservatively treat raw t as [t,t+15min); represent and expose the value at completed boundary t+15min."
)
SMART_METER_COLUMNS = (
    "Household_ID", "AffectsTimePoint", "Group", "Timestamp",
    TARGET_COLUMN, "kWh_received_HeatPump", "kWh_received_Other",
)
SMALL_FILES = (
    "meta_data/households.csv",
    "meta_data/meta_data.csv",
    "meta_data/meta_data_variables.csv",
    "smart_meter_data/overview/smart_meter_data_15min_overview.csv",
    "smart_meter_data/overview/smart_meter_data_daily_overview.csv",
    "smart_meter_data/overview/smart_meter_data_monthly_overview.csv",
    "smart_meter_data/overview/smart_meter_data_cumulative_counters_overview.csv",
    "weather_data/overview/weather_variables_availability.csv",
    "weather_data/overview/weather_variables.csv",
)


def read_households(source: str | Path | IO[bytes]) -> pd.DataFrame:
    frame = read_csv(source)
    required = {"Household_ID", "Group", "Weather_ID", "SmartMeterData_Available_15min"}
    if required - set(frame.columns) or frame.empty:
        raise ValueError("HEAPO households.csv is empty or missing identity/group/weather/availability columns")
    ids = pd.to_numeric(frame["Household_ID"], errors="raise")
    if ids.isna().any() or (ids % 1 != 0).any() or (ids <= 0).any() or ids.duplicated().any():
        raise ValueError("HEAPO Household_ID must contain unique positive integers")
    frame["Household_ID"] = ids.astype("int64")
    if not frame["Group"].isin(["control", "treatment"]).all():
        raise ValueError("HEAPO Group must be exactly 'control' or 'treatment'")
    if not frame["Weather_ID"].astype("string").str.fullmatch(r"[A-Za-z0-9]+").fillna(False).all():
        raise ValueError("HEAPO Weather_ID must be a nonempty alphanumeric station identifier")
    if not frame["SmartMeterData_Available_15min"].isin([True, False]).all():
        raise ValueError("HEAPO 15-minute availability must be boolean")
    return frame.sort_values("Household_ID", kind="stable").reset_index(drop=True)


def validate_data_config(data: DataConfig) -> None:
    if data.dataset_id != DATASET_ID or data.revision != DATASET_REVISION:
        raise ValueError("HEAPO requires Zenodo record 15056919, revision v1")
    if data.entity_type != "household" or data.target_column != TARGET_COLUMN:
        raise ValueError("HEAPO requires entity_type: household and target_column: kWh_received_Total")
    if set(data.options) != {"group", "weather_id"}:
        raise ValueError("HEAPO data.options must contain exactly group and weather_id")
    if data.options["group"] not in ("control", "treatment", "all"):
        raise ValueError("HEAPO group must be control, treatment, or all")
    station = data.options["weather_id"]
    if not isinstance(station, str) or re.fullmatch(r"[A-Za-z0-9]+", station) is None:
        raise ValueError("HEAPO weather_id must be an opaque alphanumeric station identifier")


def default_cache_name(data: DataConfig) -> str:
    validate_data_config(data)
    return f"heapo_{data.options['group']}_station_{data.options['weather_id']}"


def allow_patterns(data: DataConfig | DownloadDataConfig, *, entity_ids: Sequence[str] = ()) -> list[str]:
    del entity_ids
    if data.dataset_id != DATASET_ID or data.revision != DATASET_REVISION:
        raise ValueError("HEAPO requires Zenodo record 15056919, revision v1")
    return [ARCHIVE_NAME]


def household_number(entity_id: str) -> int:
    """Parse anonymized IDs without assigning ordinal or spatial meaning."""

    match = re.fullmatch(r"household::([1-9][0-9]*)", entity_id)
    if match is None:
        raise ValueError(f"invalid HEAPO entity ID {entity_id!r}; expected household::<positive integer>")
    return int(match.group(1))


def verify_archive(path: str | Path) -> None:
    archive = Path(path)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
    if archive.stat().st_size != ARCHIVE_SIZE or digest != ARCHIVE_MD5:
        raise ValueError("HEAPO archive failed official v1 size/MD5 verification")


def archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Validate all paths before any member is opened or extracted."""

    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if (
            path.is_absolute() or ".." in path.parts or "\\" in info.filename
            or not info.filename.startswith(ARCHIVE_PREFIX)
            or stat.S_ISLNK(info.external_attr >> 16)
        ):
            raise ValueError(f"unsafe HEAPO ZIP member: {info.filename!r}")
        if info.is_dir():
            continue
        relative = str(path.relative_to("heapo_data"))
        if relative in members or info.file_size > 100_000_000:
            raise ValueError(f"duplicate or oversized HEAPO ZIP member: {info.filename!r}")
        members[relative] = info
    missing = set(SMALL_FILES) - members.keys()
    if missing:
        raise ValueError(f"HEAPO v1 archive is missing required members: {sorted(missing)}")
    return members


def _download_archive(root: Path) -> Path:
    destination = root / "archives" / ARCHIVE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_archive(destination)
        return destination
    partial = destination.with_suffix(".zip.part")
    try:
        subprocess.run([
            "curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
            "--output", str(partial),
            f"https://zenodo.org/api/records/15056919/files/{ARCHIVE_NAME}/content",
        ], check=True)
    except FileNotFoundError as error:
        raise RuntimeError("HEAPO download requires curl on PATH") from error
    verify_archive(partial)
    partial.replace(destination)
    return destination


def safe_extract_selected(
    zip_path: str | Path,
    destination: str | Path,
    household_ids: Sequence[int],
) -> list[Path]:
    """Extract metadata, requested quarter-hour files, and only their station weather."""

    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(isinstance(value, bool) or value < 1 for value in household_ids):
        raise ValueError("HEAPO selected household IDs must be positive integers")
    with zipfile.ZipFile(zip_path) as archive:
        members = archive_members(archive)
        with archive.open(members["meta_data/households.csv"]) as stream:
            households = read_households(stream).set_index("Household_ID")
        unknown = set(household_ids) - set(households.index)
        if unknown:
            raise ValueError(f"unknown HEAPO households: {sorted(unknown)}")
        stations = households.loc[list(household_ids), "Weather_ID"].unique().tolist()
        selected = [
            *SMALL_FILES,
            *(f"smart_meter_data/15min/{identifier}.csv" for identifier in sorted(set(household_ids))),
            *(f"weather_data/hourly/{station}.csv" for station in sorted(stations)),
        ]
        missing = set(selected) - members.keys()
        if missing:
            raise ValueError(f"selected HEAPO archive members are missing: {sorted(missing)}")
        targets = [(name, (root / name).resolve()) for name in selected]
        for name, target in targets:
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe HEAPO extraction destination: {name}")
        for name, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            if partial.is_symlink():
                raise ValueError(f"refusing symlink extraction partial: {partial}")
            with archive.open(members[name]) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            if partial.stat().st_size != members[name].file_size:
                raise ValueError(f"truncated HEAPO archive member: {name}")
            partial.replace(target)
    return [target for _, target in targets]


def download_data(
    config: BaseExperimentConfig | DatasetDownloadConfig,
    *,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> Path:
    del snapshot_download_fn
    data = config.data
    if data.adapter != ADAPTER_NAME or data.dataset_id != DATASET_ID or data.revision != DATASET_REVISION:
        raise ValueError("HEAPO downloads require Zenodo record 15056919, revision v1")
    root = Path(data.local_dir).expanduser().resolve()
    archive = _download_archive(root)
    if isinstance(config, DatasetDownloadConfig):
        with zipfile.ZipFile(archive) as source:
            members = archive_members(source)
            household_ids = [
                int(PurePosixPath(name).stem) for name in members
                if re.fullmatch(r"smart_meter_data/15min/[1-9][0-9]*\.csv", name)
            ]
        if config.entity_count != len(household_ids):
            raise ValueError("HEAPO dataset-wide entity_count must match the 1407 quarter-hour files")
    else:
        household_ids = [household_number(identifier) for identifier in config.protocol.ordered_entity_ids]
        if not household_ids:
            raise ValueError("HEAPO selective download requires explicit protocol.ordered_entity_ids")
    safe_extract_selected(archive, root, household_ids)
    return root


def read_csv(source: str | Path | IO[bytes]) -> pd.DataFrame:
    return pd.read_csv(source, sep=";")


def read_smart_meter(source: str | Path | IO[bytes], household_id: int) -> pd.DataFrame:
    """Read raw UTC labels, leaving missing rows/values and target values untouched."""

    frame = read_csv(source)
    missing = set(SMART_METER_COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"HEAPO household {household_id}: empty data or missing columns {sorted(missing)}")
    if not frame["Household_ID"].eq(household_id).all():
        raise ValueError(f"HEAPO household ID mismatch for {household_id}")
    if not frame["Group"].isin(["control", "treatment"]).all() or frame["Group"].nunique() != 1:
        raise ValueError(f"HEAPO household {household_id} has invalid/inconsistent Group")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame.pop("Timestamp"), utc=True, errors="raise"))
    if timestamps.hasnans or timestamps.has_duplicates:
        raise ValueError(f"HEAPO household {household_id} has missing/duplicate UTC timestamps")
    if ((timestamps.minute % 15 != 0) | (timestamps.second != 0) | (timestamps.microsecond != 0)).any():
        raise ValueError(f"HEAPO household {household_id} has off-grid 15-minute timestamps")
    values = pd.to_numeric(frame[TARGET_COLUMN], errors="raise")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError(f"HEAPO household {household_id} has infinite target values")
    frame[TARGET_COLUMN] = values
    frame.index = timestamps.rename("timestamp")
    return frame.sort_index(kind="stable")


def target_summary(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame[TARGET_COLUMN]
    slots = int((frame.index.max() - frame.index.min()) / pd.Timedelta(minutes=15)) + 1
    finite = int(np.isfinite(target.to_numpy(dtype=float)).sum())
    return {
        "start": frame.index.min().isoformat(),
        "end": frame.index.max().isoformat(),
        "rows": len(frame),
        "finite_quarter_hours": finite,
        "equivalent_available_days": finite / 96,
        "days_with_observations": int(pd.DatetimeIndex(frame.index[target.notna()]).normalize().nunique()),
        "lifetime_days": slots / 96,
        "missing_values": int(target.isna().sum()),
        "absent_quarter_hours": slots - len(frame),
        "missing_fraction_within_lifetime": 1 - finite / slots,
    }


def read_target(source: str | Path | IO[bytes], household_id: int) -> pd.DataFrame:
    """Expose energy only at its conservatively assumed completed-interval boundary."""

    raw = read_smart_meter(source, household_id)
    target = raw.loc[:, [TARGET_COLUMN]].copy()
    target.index = pd.DatetimeIndex(target.index + TARGET_LABEL_SHIFT, name="timestamp")
    return target


def load_adapter(config: ResolvedExperimentConfig) -> DatasetAdapter:
    validate_data_config(config.data)
    if config.forecast.frequency_minutes != 15:
        raise ValueError("HEAPO requires native 15-minute resolution")
    if config.covariates.future_weather_source != "oracle":
        raise ValueError("HEAPO has observed weather only; require future_weather_source: oracle")
    if config.features is not None and config.features.use_location:
        raise ValueError("HEAPO has no household coordinates; set features.use_location: false")
    root = Path(config.data.local_dir).expanduser().resolve()
    households = read_households(root / "meta_data/households.csv").set_index("Household_ID")
    ids = [household_number(identifier) for identifier in config.protocol.ordered_entity_ids]
    if not ids or len(set(ids)) != len(ids) or config.protocol.entity_count != len(ids):
        raise ValueError("HEAPO requires explicit unique canonical household IDs and matching entity_count")
    unknown = set(ids) - set(households.index)
    if unknown:
        raise ValueError(f"unknown HEAPO households: {sorted(unknown)}")
    selected = households.loc[ids]
    station = str(config.data.options["weather_id"])
    group_name = config.data.options["group"]
    if not selected["Weather_ID"].eq(station).all():
        raise ValueError("selected HEAPO households do not all map to configured Weather_ID")
    if group_name != "all" and not selected["Group"].eq(group_name).all():
        raise ValueError("selected HEAPO households violate configured control/treatment filter")
    if not selected["SmartMeterData_Available_15min"].all():
        raise ValueError("selected HEAPO household has no 15-minute data")
    weather_raw = read_csv(root / f"weather_data/hourly/{station}.csv")
    if "Weather_ID" not in weather_raw or not weather_raw["Weather_ID"].eq(station).all():
        raise ValueError("HEAPO hourly weather station ID mismatch")
    weather_raw.index = pd.DatetimeIndex(pd.to_datetime(weather_raw.pop("Timestamp"), utc=True))
    if weather_raw.index.has_duplicates or weather_raw.index.hasnans:
        raise ValueError("HEAPO weather has missing or duplicate UTC timestamps")
    fields = config.covariates.weather
    if not fields or set(fields) - set(weather_raw.columns) or "Weather_ID" in fields:
        raise ValueError("HEAPO weather must select supplied hourly measurement columns")
    weather = weather_raw.loc[:, fields].apply(pd.to_numeric, errors="raise").sort_index()
    entities = [
        EntityMetadata(
            name=str(identifier), group_name="household",
            description=f"HEAPO {group_name} household, coarse weather-station cohort {station}",
            extra={"Household_ID": identifier, "Group": selected.loc[identifier, "Group"], "Weather_ID": station},
        )
        for identifier in ids
    ]
    frames = [
        EntityFrames(
            entity=entity,
            target=read_target(root / f"smart_meter_data/15min/{identifier}.csv", identifier),
            past_weather=weather,
            future_weather=pd.DataFrame(index=weather.index),
        )
        for identifier, entity in zip(ids, entities, strict=True)
    ]
    group = EntityGroup(
        group_id=default_cache_name(config.data), entity_ids=[entity.entity_id for entity in entities],
        entities=entities, metadata={"Weather_ID": station, "group": group_name, "topology": "not_electrical"},
    )
    provenance = {
        "adapter": ADAPTER_NAME, "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "dataset_doi": DATASET_DOI, "license": LICENSE, "archive_md5": ARCHIVE_MD5,
        "target": TARGET_COLUMN, "units": "kWh", "target_label_shift_minutes": 15,
        "interval_convention": INTERVAL_CONVENTION, "Weather_ID": station,
        "future_weather_source": "oracle", "coordinates": "not supplied",
        "grouping": "nearest weather station; coarse proximity cohort, not electrical topology",
        "households": json.loads(selected.reset_index().to_json(orient="records")),
    }
    return FrameDatasetAdapter(config, group, frames, provenance)


def daily_interval_check(quarter_hours: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    """Compare complete UTC-day sums under both boundary interpretations, without filling gaps."""

    daily_values = daily[TARGET_COLUMN].copy()
    daily_values.index = pd.DatetimeIndex(pd.to_datetime(daily["Timestamp"], utc=True)).normalize()
    result: dict[str, Any] = {}
    for convention, offset in (("left_labelled", 0), ("right_labelled", -15)):
        values = quarter_hours[TARGET_COLUMN].copy()
        values.index = pd.DatetimeIndex(values.index) + pd.Timedelta(minutes=offset)
        totals = values.resample("D").sum(min_count=96)
        compared = pd.concat([totals.rename("quarters"), daily_values.rename("daily")], axis=1, sort=True).dropna()
        difference = (compared["quarters"] - compared["daily"]).abs()
        result[convention] = {
            "complete_compared_days": len(compared),
            "mean_absolute_difference_kWh": float(difference.mean()) if len(compared) else None,
            "median_absolute_difference_kWh": float(difference.median()) if len(compared) else None,
            "fraction_within_0.02_kWh": float((difference <= 0.02).mean()) if len(compared) else None,
        }
    return result


def inspect_archive(path: str | Path, output: str | Path) -> dict[str, Any]:
    """Audit every 15-minute household in the verified archive without extracting it."""

    verify_archive(path)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        members = archive_members(archive)
        schemas: dict[str, list[str]] = {}
        for name in SMALL_FILES:
            with archive.open(members[name]) as stream:
                schemas[name] = read_csv(stream).columns.tolist()
        with archive.open(members["meta_data/households.csv"]) as stream:
            households = read_households(stream)
        for name in sorted(members):
            if name.startswith("weather_data/hourly/"):
                with archive.open(members[name]) as stream:
                    schemas[name] = read_csv(stream).columns.tolist()
        station_counts = []
        for station, cohort in households.groupby("Weather_ID", sort=True):
            control = cohort["Group"].eq("control")
            station_counts.append({
                "Weather_ID": str(station), "total": len(cohort),
                "control": int(control.sum()),
                "control_15min": int((control & cohort["SmartMeterData_Available_15min"]).sum()),
            })
        column_counts: Counter[str] = Counter()
        for name in sorted(members):
            if not re.fullmatch(r"smart_meter_data/15min/[1-9][0-9]*\.csv", name):
                continue
            household_id = int(PurePosixPath(name).stem)
            with archive.open(members[name]) as stream:
                frame = read_smart_meter(stream, household_id)
            column_counts.update(frame.columns)
            rows.append({"Household_ID": household_id, "Group": frame["Group"].iloc[0], **target_summary(frame)})
        summary = {
            "record": DATASET_ID, "revision": DATASET_REVISION, "license": LICENSE,
            "md5": ARCHIVE_MD5, "archive_bytes": Path(path).stat().st_size,
            "extracted_bytes": sum(member.file_size for member in members.values()),
            "member_count": len(members), "schemas": schemas,
            "layout": dict(Counter(str(PurePosixPath(name).parent) for name in members)),
            "smart_meter_columns": list(SMART_METER_COLUMNS),
            "column_household_counts": dict(column_counts),
            "households_15min": len(rows),
            "groups_15min": dict(Counter(row["Group"] for row in rows)),
            "groups_all": households["Group"].value_counts().to_dict(),
            "station_counts": station_counts,
            "duplicate_timestamps": 0,
            "targets": rows,
        }
    pd.DataFrame(rows).to_csv(destination / "household_target_lifetimes.csv", index=False)
    (destination / "source_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary
