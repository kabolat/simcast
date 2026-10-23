"""HEAPO v1 household grid-received interval energy and station-cohort diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import IO, Any

import numpy as np
import pandas as pd

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
SMART_METER_COLUMNS = (
    "Household_ID", "AffectsTimePoint", "Group", "Timestamp",
    TARGET_COLUMN, "kWh_received_HeatPump", "kWh_received_Other",
)
SMALL_FILES = (
    "meta_data/households.csv",
    "meta_data/meta_data.csv",
    "meta_data/meta_data_variables.csv",
    "smart_meter_data/overview/smart_meter_data_15min_overview.csv",
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
