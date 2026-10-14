"""FeederBW v1.1 data loading, group diagnostics, and selective Zenodo downloads."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import stat
import subprocess
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import typer

from simcast.config import (
    BaseExperimentConfig,
    DataConfig,
    DatasetDownloadConfig,
    DownloadDataConfig,
    IndependentMethodConfig,
    ResolvedExperimentConfig,
    resolve_run_config,
)
from simcast.data.adapter import DatasetAdapter, EntityFrames, FrameDatasetAdapter
from simcast.types import EntityGroup, EntityMetadata

LOGGER = logging.getLogger(__name__)

ADAPTER_NAME = "feederbw"
DATASET_ID = "https://zenodo.org/records/19550073"
DATASET_REVISION = "v1.1"
ZENODO_RECORD_ID = 19550073
ZENODO_DOI = "10.5281/zenodo.19550073"
CONCEPT_RECORD_ID = 17831176
V1_0_RECORD_ID = 17831177
LICENSE = "CC BY 4.0"
METADATA_FILE = "feeder_metadata.csv"
WEATHER_FILE = "weather_data.parquet"
MEASUREMENTS_DIR = "measurements"
ARCHIVES_DIR = "archives"
TARGET_COLUMN = "active_power_kW"
SOURCE_TIMESTAMP_COLUMN = "timestamp_UTC"
SOURCE_RESOLUTION_MINUTES = 1
TARGET_RESOLUTION_MINUTES = 15
DEFAULT_MIN_SAMPLES_PER_INTERVAL = 15
WEATHER_MAX_AGE = pd.Timedelta(hours=1)
BATCHES = ((1, 40), (41, 80), (81, 120), (121, 160), (161, 200))
WEATHER_FEATURES = (
    "direct_radiation_W/m2",
    "diffuse_radiation_W/m2",
    "air_temperature_C",
    "precipitation_kg/m2",
    "humidity_kg/kg",
    "max_wind_gust_m/s",
    "meridional_wind_m/s",
    "zonal_wind_m/s",
)
METADATA_DIAGNOSTIC_FIELDS = (
    "housing_units_count",
    "industry_and_commerce_count",
    "PV_systems_kW",
    "heat_pumps_kW",
    "EV_chargers_kW",
    "batteries_kW",
)
OPTION_KEYS = frozenset(
    {
        "substation_id",
        "source_resolution_minutes",
        "aggregation",
        "minimum_samples_per_interval",
        "metadata_usage",
    }
)


def _asset_info() -> dict[str, tuple[int, str]]:
    return {
        "feeder_metadata.csv": (177870, "d93ced868ce0c9600ec8bce6b0df196c"),
        "weather_data.parquet": (105578255, "6f4263241741e2b59722f024f17bbacf"),
        "feeder_measurement_data_001-040.zip": (1697455667, "9aeb811da458e9137e7318f918b74b30"),
        "feeder_measurement_data_041-080.zip": (1694329072, "c9d616dc746943a5f72e4175911170c2"),
        "feeder_measurement_data_081-120.zip": (1685824608, "6979a158742641310cbe1c480b59f1d2"),
        "feeder_measurement_data_121-160.zip": (1697631248, "773df953e799e3ce9b84463b001df3f8"),
        "feeder_measurement_data_161-200.zip": (1667802546, "c237f9a0ccb3cb3ff09bb59caaf529b7"),
    }


ASSET_INFO = _asset_info()


def validate_data_config(data: DataConfig) -> None:
    _validate_download_source(data)
    if data.entity_type != ADAPTER_NAME:
        raise ValueError(f"feederbw requires data.entity_type={ADAPTER_NAME!r}")
    if set(data.options) != OPTION_KEYS:
        raise ValueError(f"feederbw requires exactly these data.options: {sorted(OPTION_KEYS)}")
    options = data.options
    if options["source_resolution_minutes"] != SOURCE_RESOLUTION_MINUTES:
        raise ValueError("feederbw source_resolution_minutes must be 1")
    if options["aggregation"] != "mean":
        raise ValueError("feederbw aggregation must be 'mean'")
    threshold = options["minimum_samples_per_interval"]
    if not isinstance(threshold, int) or isinstance(threshold, bool) or not 12 <= threshold <= 15:
        raise ValueError("feederbw minimum_samples_per_interval must be an integer from 12 through 15")
    if options["metadata_usage"] != "grouping_and_diagnostics_only":
        raise ValueError("feederbw metadata_usage must be 'grouping_and_diagnostics_only'")
    substation_id = options["substation_id"]
    if not isinstance(substation_id, int) or isinstance(substation_id, bool) or substation_id < 1:
        raise ValueError("feederbw substation_id must be a positive integer")


def _validate_download_source(data: DataConfig | DownloadDataConfig) -> None:
    if data.adapter != ADAPTER_NAME:
        raise ValueError(f"FeederBW download requires adapter={ADAPTER_NAME!r}")
    if data.dataset_id != DATASET_ID or data.revision != DATASET_REVISION:
        raise ValueError(f"feederbw requires Zenodo record {ZENODO_RECORD_ID} ({DATASET_REVISION})")


def _feeder_number(entity_id: str) -> int:
    match = re.fullmatch(r"feederbw::feeder_(\d{3})", entity_id)
    if match is None:
        raise ValueError(f"invalid FeederBW entity ID {entity_id!r}; expected feederbw::feeder_NNN")
    feeder = int(match.group(1))
    if not 1 <= feeder <= 200:
        raise ValueError(f"FeederBW feeder ID out of range: {feeder}")
    return feeder


def _batch_name(feeder: int) -> str:
    start = ((feeder - 1) // 40) * 40 + 1
    end = start + 39
    return f"feeder_measurement_data_{start:03d}-{end:03d}.zip"


def allow_patterns(
    data: DataConfig | DownloadDataConfig,
    *,
    entity_ids: Sequence[str] = (),
) -> list[str]:
    """Return shared files and the measurement archives implied by the config type."""

    if isinstance(data, DownloadDataConfig):
        _validate_download_source(data)
        batches = sorted(name for name in ASSET_INFO if name.endswith(".zip"))
    else:
        validate_data_config(data)
        feeders = sorted({_feeder_number(entity_id) for entity_id in entity_ids})
        if not feeders:
            raise ValueError("FeederBW downloads require protocol.ordered_entity_ids")
        batches = sorted({_batch_name(feeder) for feeder in feeders})
    if not batches:
        raise ValueError("FeederBW downloads require protocol.ordered_entity_ids")
    return [METADATA_FILE, WEATHER_FILE, *batches]


def default_cache_name(data: DataConfig) -> str:
    validate_data_config(data)
    return f"feederbw_substation_{data.options['substation_id']}"


def _asset_url(name: str) -> str:
    if name not in ASSET_INFO:
        raise ValueError(f"unknown FeederBW v1.1 asset: {name}")
    return f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/{quote(name)}/content"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_asset(name: str, destination: Path) -> Path:
    """Resume a Zenodo download with curl, then verify the published size and MD5."""

    size, expected_md5 = ASSET_INFO[name]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size and _md5(destination) == expected_md5:
        return destination
    partial = destination.with_name(destination.name + ".part")
    if destination.exists():
        destination.unlink()
    command = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "3",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        _asset_url(name),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FeederBW download requires curl on PATH") from exc
    if partial.stat().st_size != size:
        raise ValueError(f"downloaded {name} has size {partial.stat().st_size}, expected {size}; rerun to resume")
    actual_md5 = _md5(partial)
    if actual_md5 != expected_md5:
        partial.unlink(missing_ok=True)
        raise ValueError(f"downloaded {name} failed Zenodo MD5 verification")
    partial.replace(destination)
    return destination


def safe_extract_selected(zip_path: str | Path, feeder_ids: Sequence[int], destination: str | Path) -> list[Path]:
    """Extract only requested top-level feeder Parquets without trusting ZIP paths."""

    archive_path = Path(zip_path)
    output_root = Path(destination).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = {f"feeder_{feeder:03d}.parquet" for feeder in feeder_ids}
    if not selected:
        return []
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        for info in infos:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts or "\\" in info.filename:
                raise ValueError(f"unsafe path in FeederBW ZIP: {info.filename!r}")
        for name in sorted(selected):
            matches = [info for info in infos if PurePosixPath(info.filename).name == name and not info.is_dir()]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one ZIP member for {name}, found {len(matches)}")
            info = matches[0]
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"refusing symlink ZIP member: {info.filename!r}")
            if info.file_size > 1_000_000_000:
                raise ValueError(f"unexpectedly large ZIP member: {info.filename!r}")
            target = (output_root / name).resolve()
            if target.parent != output_root:
                raise ValueError(f"unsafe extraction target for {info.filename!r}")
            partial = target.with_name(target.name + ".part")
            with archive.open(info) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            if partial.stat().st_size != info.file_size:
                partial.unlink(missing_ok=True)
                raise ValueError(f"truncated ZIP member: {info.filename!r}")
            partial.replace(target)
            extracted.append(target)
    return extracted


def download_data(
    config: BaseExperimentConfig | DatasetDownloadConfig,
    *,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> Path:
    """Download the configured group or the dataset-wide FeederBW profile."""

    del snapshot_download_fn
    data = config.data
    root = Path(data.local_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(config, DatasetDownloadConfig):
        if config.scope != "all" or config.entity_count != 200:
            raise ValueError("FeederBW full-download profile must declare scope: all and entity_count: 200")
        entity_ids: Sequence[str] = ()
    else:
        entity_ids = config.protocol.ordered_entity_ids
        if not entity_ids:
            raise ValueError("FeederBW base must declare ordered entity IDs before downloading")
    feeders = list(range(1, 201)) if isinstance(config, DatasetDownloadConfig) else [
        _feeder_number(entity_id) for entity_id in entity_ids
    ]
    patterns = allow_patterns(data, entity_ids=entity_ids)
    for name in patterns:
        if name.endswith(".zip"):
            target = _download_asset(name, root / ARCHIVES_DIR / name)
            batch_feeders = [feeder for feeder in feeders if _batch_name(feeder) == name]
            safe_extract_selected(target, batch_feeders, root / MEASUREMENTS_DIR)
        else:
            _download_asset(name, root / name)
    return root


def read_metadata(path: str | Path) -> pd.DataFrame:
    """Read the v1.1 CSV schema, including the actual ``substation`` column."""

    frame = pd.read_csv(Path(path))
    required = {"feeder", "substation", "date", *METADATA_DIAGNOSTIC_FIELDS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FeederBW v1.1 metadata is missing columns: {missing}")
    if frame[["feeder", "substation", "date"]].isna().any().any():
        raise ValueError("FeederBW metadata has missing feeder, substation, or date values")
    for column in ("feeder", "substation"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if bool((numeric % 1 != 0).any()):
            raise ValueError(f"FeederBW metadata {column} values must be integers")
        frame[column] = numeric.astype("int64")
    local_dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    frame["date"] = local_dates.tz_localize(
        "Europe/Berlin", ambiguous="raise", nonexistent="raise"
    ).tz_convert("UTC")
    if bool((~frame["feeder"].between(1, 200)).any()):
        raise ValueError("FeederBW metadata feeder IDs must be in [1, 200]")
    if frame.duplicated(["feeder", "date"]).any():
        raise ValueError("FeederBW metadata has duplicate feeder/date rows")
    if bool((frame.groupby("feeder")["substation"].nunique() != 1).any()):
        raise ValueError("a FeederBW feeder changes substation in the selected metadata")
    return frame.sort_values(["feeder", "date"], kind="stable").reset_index(drop=True)


def feeders_for_substation(metadata: pd.DataFrame, substation_id: int) -> list[int]:
    """Return a deterministic ascending feeder list for one stable substation group."""

    feeders = sorted(metadata.loc[metadata["substation"] == substation_id, "feeder"].unique().astype(int).tolist())
    if not feeders:
        raise ValueError(f"no FeederBW feeders belong to substation {substation_id}")
    return feeders


def resample_active_power(
    path: str | Path,
    *,
    feeder_id: int,
    minimum_samples_per_interval: int = DEFAULT_MIN_SAMPLES_PER_INTERVAL,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Mean 1-minute active power into completed, right-labelled 15-minute UTC bins."""

    frame = pd.read_parquet(Path(path), columns=["feeder", SOURCE_TIMESTAMP_COLUMN, TARGET_COLUMN])
    if frame.empty:
        raise ValueError(f"empty FeederBW measurement file: {path}")
    if not bool((pd.to_numeric(frame["feeder"], errors="raise") == feeder_id).all()):
        raise ValueError(f"measurement file {path} does not contain only feeder {feeder_id}")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame.pop(SOURCE_TIMESTAMP_COLUMN), utc=True, errors="raise"))
    if timestamps.has_duplicates:
        raise ValueError(f"FeederBW feeder {feeder_id} has duplicate UTC timestamps")
    if bool(((timestamps.second != 0) | (timestamps.microsecond != 0)).any()):
        raise ValueError(f"FeederBW feeder {feeder_id} has timestamps off the one-minute grid")
    values = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    minute_grid = pd.date_range(timestamps.min(), timestamps.max(), freq="min", name="timestamp")
    missing_source_timestamps = len(minute_grid.difference(timestamps))
    source = pd.DataFrame({TARGET_COLUMN: values}, index=timestamps.rename("timestamp")).sort_index(kind="stable")
    source_index = pd.DatetimeIndex(source.index)
    target_frequency = f"{TARGET_RESOLUTION_MINUTES}min"
    interval_offset = pd.Timedelta(minutes=TARGET_RESOLUTION_MINUTES)
    source["interval"] = source_index.floor(target_frequency) + interval_offset
    grouped = source.groupby("interval", sort=True)[TARGET_COLUMN].agg(["mean", "count"])
    target_index = pd.date_range(
        source_index.min().floor(target_frequency) + interval_offset,
        source_index.max().floor(target_frequency) + interval_offset,
        freq=target_frequency,
        name="timestamp",
    )
    grouped = grouped.reindex(target_index)
    interval_count = grouped["count"].fillna(0).astype("int16")
    target = grouped["mean"].where(interval_count >= minimum_samples_per_interval)
    result = pd.DataFrame({TARGET_COLUMN: target.astype("float32")}, index=target_index)
    valid_intervals = int(result[TARGET_COLUMN].notna().sum())
    stats: dict[str, int | float | str] = {
        "feeder": feeder_id,
        "source_rows": len(source),
        "missing_source_timestamps": int(missing_source_timestamps),
        "missing_source_values": int(np.isnan(values).sum()),
        "intervals": len(result),
        "valid_intervals": valid_intervals,
        "invalid_intervals": len(result) - valid_intervals,
        "valid_fraction": valid_intervals / len(result),
        "target_min_kW": float(result[TARGET_COLUMN].min()),
        "target_max_kW": float(result[TARGET_COLUMN].max()),
        "target_mean_kW": float(result[TARGET_COLUMN].mean()),
    }
    return result, stats


def _read_selected_weather(path: Path, feeder_ids: Sequence[int], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["feeder", SOURCE_TIMESTAMP_COLUMN, *columns],
        filters=[("feeder", "in", list(feeder_ids))],
    )
    if frame.empty:
        raise ValueError("FeederBW weather file has no rows for the selected feeders")
    frame["feeder"] = pd.to_numeric(frame["feeder"], errors="raise").astype("int64")
    frame[SOURCE_TIMESTAMP_COLUMN] = pd.to_datetime(frame[SOURCE_TIMESTAMP_COLUMN], utc=True, errors="raise")
    if frame.duplicated(["feeder", SOURCE_TIMESTAMP_COLUMN]).any():
        raise ValueError("FeederBW weather has duplicate feeder/UTC timestamp rows")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"FeederBW weather is missing configured historical fields: {missing}")
    return frame.sort_values(["feeder", SOURCE_TIMESTAMP_COLUMN], kind="stable")


def _align_hourly_weather(weather: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Apply a zero-order hold only while the latest hourly value is at most one hour old."""

    return weather.reindex(target_index, method="ffill", tolerance=WEATHER_MAX_AGE).astype("float32")


def _latest_metadata(
    metadata: pd.DataFrame,
    feeder_ids: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    selected = metadata.loc[metadata["feeder"].isin(feeder_ids)]
    latest = selected.sort_values("date", kind="stable").groupby("feeder", sort=True, as_index=False).tail(1)
    rows: list[dict[str, Any]] = []
    for _, row in latest.sort_values("feeder").iterrows():
        rows.append(
            {
                "feeder": int(row["feeder"]),
                "date": pd.Timestamp(row["date"]).isoformat(),
                **{field: None if pd.isna(row[field]) else float(row[field]) for field in METADATA_DIAGNOSTIC_FIELDS},
            }
        )
    totals = {
        field: float(latest[field].sum(min_count=1)) if latest[field].notna().any() else 0.0
        for field in METADATA_DIAGNOSTIC_FIELDS
    }
    return rows, totals


class FeederBWAdapter(FrameDatasetAdapter):
    """Frame adapter with a data-quality summary for the selected substation."""

    def __init__(
        self,
        config: ResolvedExperimentConfig,
        group: EntityGroup,
        frames: Sequence[EntityFrames],
        provenance: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> None:
        super().__init__(config, group, frames, provenance, past_weather_source="measured")
        self._diagnostics = diagnostics

    def diagnostics(self) -> dict[str, Any]:
        """Return group selection, temporal quality, targets, weather, and latest metadata summaries."""

        return self._diagnostics


def load_adapter(config: ResolvedExperimentConfig) -> DatasetAdapter:
    validate_data_config(config.data)
    if config.data.target_column != TARGET_COLUMN:
        raise ValueError(f"FeederBW target_column must be {TARGET_COLUMN!r}")
    if config.forecast.frequency_minutes != TARGET_RESOLUTION_MINUTES:
        raise ValueError("FeederBW initial adapter requires 15-minute forecast resolution")
    if config.covariates.future_weather_source != "oracle":
        raise ValueError(
            "FeederBW has no archived future forecasts; use future_weather_source: oracle "
            "to use realized weather (which leaks future information)"
        )
    if not config.covariates.calendar.enabled:
        raise ValueError("FeederBW requires future-known calendar covariates")
    weather_fields = config.covariates.weather
    if not weather_fields or any(field not in WEATHER_FEATURES for field in weather_fields):
        raise ValueError(f"FeederBW covariates.weather must select from {list(WEATHER_FEATURES)}")

    root = Path(config.data.local_dir).expanduser().resolve()
    metadata = read_metadata(root / METADATA_FILE)
    substation_id = int(config.data.options["substation_id"])
    feeder_ids = feeders_for_substation(metadata, substation_id)
    ordered_ids = [f"feederbw::feeder_{feeder:03d}" for feeder in feeder_ids]
    if list(config.protocol.ordered_entity_ids) != ordered_ids:
        raise ValueError(
            "protocol.ordered_entity_ids must list every feeder in ascending order for "
            f"substation {substation_id}: expected {ordered_ids}"
        )
    if config.protocol.entity_count != len(ordered_ids):
        raise ValueError("protocol.entity_count must equal the selected substation feeder count")

    weather_path = root / WEATHER_FILE
    selected_weather = _read_selected_weather(weather_path, feeder_ids, weather_fields)
    frames: list[EntityFrames] = []
    target_stats: list[dict[str, int | float | str]] = []
    for feeder_id in feeder_ids:
        target, stats = resample_active_power(
            root / MEASUREMENTS_DIR / f"feeder_{feeder_id:03d}.parquet",
            feeder_id=feeder_id,
            minimum_samples_per_interval=int(config.data.options["minimum_samples_per_interval"]),
        )
        target_stats.append(stats)
        weather = selected_weather.loc[selected_weather["feeder"] == feeder_id]
        weather_frame = weather.set_index(SOURCE_TIMESTAMP_COLUMN).loc[:, list(weather_fields)]
        weather_frame.index = pd.DatetimeIndex(weather_frame.index, name="timestamp")
        weather_frame = _align_hourly_weather(
            weather_frame,
            pd.DatetimeIndex(target.index, name="timestamp"),
        )
        entity = EntityMetadata(
            name=f"feeder_{feeder_id:03d}",
            group_name=ADAPTER_NAME,
            description=f"Low-voltage feeder in FeederBW substation {substation_id}",
            extra={"substation": substation_id, "feeder_number": feeder_id},
        )
        frames.append(
            EntityFrames(
                entity=entity,
                target=target,
                past_weather=weather_frame,
                future_weather=pd.DataFrame(index=target.index),
            )
        )

    group = EntityGroup(
        group_id=f"feederbw_substation_{substation_id}",
        entity_ids=ordered_ids,
        metadata={"entity_type": ADAPTER_NAME, "substation_id": substation_id, "entity_count": len(ordered_ids)},
        entities=[frame.entity for frame in frames],
    )
    target_valid = pd.concat(
        [frame.target[TARGET_COLUMN].notna().rename(frame.entity.entity_id) for frame in frames], axis=1
    ).fillna(False)
    common_valid = target_valid.all(axis=1)
    latest_metadata, capacity_totals = _latest_metadata(metadata, feeder_ids)
    weather_summary = {
        "fields": list(weather_fields),
        "past_usage": "historical measured weather",
        "future_weather_source": config.covariates.future_weather_source,
        "future_weather_usage": (
            "oracle experiment: realized future measurements; intentionally leaks future information and is not "
            "operational"
        ),
        "availability": "latest hourly observation at or before the target, only when no more than one hour old",
        "maximum_age_minutes": int(WEATHER_MAX_AGE.total_seconds() // 60),
        "future_weather_supplied": True,
        "source_rows": int(len(selected_weather)),
        "source_start_utc": pd.Timestamp(selected_weather[SOURCE_TIMESTAMP_COLUMN].min()).isoformat(),
        "source_end_utc": pd.Timestamp(selected_weather[SOURCE_TIMESTAMP_COLUMN].max()).isoformat(),
    }
    diagnostics = {
        "dataset_record": ZENODO_RECORD_ID,
        "dataset_version": DATASET_REVISION,
        "selected_substation_id": substation_id,
        "ordered_feeder_ids": feeder_ids,
        "entity_ids": ordered_ids,
        "entity_count": len(ordered_ids),
        "common_target_coverage_start_utc": pd.Timestamp(common_valid.index[common_valid][0]).isoformat(),
        "common_target_coverage_end_utc": pd.Timestamp(common_valid.index[common_valid][-1]).isoformat(),
        "common_complete_15m_intervals": int(common_valid.sum()),
        "common_complete_interval_fraction": float(common_valid.mean()),
        "per_feeder": target_stats,
        "target": {
            "column": TARGET_COLUMN,
            "aggregation": "mean of available one-minute observations; interval accepted only at configured threshold",
            "timestamp_semantics": "right boundary; target at t represents [t - 15 minutes, t)",
            "minimum_samples_per_interval": int(config.data.options["minimum_samples_per_interval"]),
            "group_min_kW": float(min(float(frame.target[TARGET_COLUMN].min()) for frame in frames)),
            "group_max_kW": float(max(float(frame.target[TARGET_COLUMN].max()) for frame in frames)),
            "sign_convention": "preserved; no rectification or clipping",
        },
        "weather": weather_summary,
        "metadata": {
            "schema_columns": list(metadata.columns),
            "row_count_for_group": int(metadata["feeder"].isin(feeder_ids).sum()),
            "date_start_utc": pd.Timestamp(
                metadata.loc[metadata["feeder"].isin(feeder_ids), "date"].min()
            ).isoformat(),
            "date_end_utc": pd.Timestamp(
                metadata.loc[metadata["feeder"].isin(feeder_ids), "date"].max()
            ).isoformat(),
            "usage": "substation grouping and diagnostics only; not supplied to Chronos",
            "latest_record_per_feeder": latest_metadata,
            "sum_of_latest_per_feeder": capacity_totals,
        },
    }
    provenance = {
        "adapter": ADAPTER_NAME,
        "source": {
            "record_id": ZENODO_RECORD_ID,
            "version": DATASET_REVISION,
            "doi": ZENODO_DOI,
            "license": LICENSE,
            "concept_record_id": CONCEPT_RECORD_ID,
            "v1_0_record_id": V1_0_RECORD_ID,
            "local_dir": str(root),
            "metadata_column_names": {"feeder_id": "feeder", "substation_id": "substation"},
        },
        "selected_substation_id": substation_id,
        "ordered_feeder_ids": feeder_ids,
        "measurement": {
            "column": TARGET_COLUMN,
            "source_resolution_minutes": SOURCE_RESOLUTION_MINUTES,
            "target_resolution_minutes": TARGET_RESOLUTION_MINUTES,
            "aggregation": config.data.options["aggregation"],
            "minimum_samples_per_interval": int(config.data.options["minimum_samples_per_interval"]),
            "timestamp_column": SOURCE_TIMESTAMP_COLUMN,
            "timezone": "UTC",
            "interval_label": "right boundary",
            "interval_convention": "target at t represents measurements in [t - 15 minutes, t)",
            "sign_convention": "preserved; no rectification or clipping",
        },
        "weather": weather_summary,
        "metadata_usage": config.data.options["metadata_usage"],
        "source_files": {
            name: {"size_bytes": ASSET_INFO[name][0], "md5": ASSET_INFO[name][1]}
            for name in allow_patterns(config.data, entity_ids=config.protocol.ordered_entity_ids)
        },
        "diagnostics": diagnostics,
    }
    LOGGER.info(
        "Loaded FeederBW substation %d: %d feeders, %d common complete 15-minute intervals",
        substation_id,
        len(ordered_ids),
        int(common_valid.sum()),
    )
    return FeederBWAdapter(config, group, frames, provenance, diagnostics)


def info(base: BaseExperimentConfig) -> dict[str, Any]:
    """Load the configured FeederBW group and return its diagnostics."""

    runtime = resolve_run_config(
        base,
        IndependentMethodConfig(kind="method", id="m0", family="independent"),
    )
    adapter = load_adapter(runtime)
    if not isinstance(adapter, FeederBWAdapter):
        raise TypeError("registry returned an unexpected FeederBW adapter")
    return adapter.diagnostics()


def main(base: Annotated[Path, typer.Option("--base", exists=True, dir_okay=False, readable=True)]) -> None:
    """Print the selected group's data-quality diagnostics as JSON."""

    from simcast.config import load_base_config

    config = load_base_config(base)
    typer.echo(json.dumps(info(config), indent=2, sort_keys=True, allow_nan=False))


def cli() -> None:
    typer.run(main)
