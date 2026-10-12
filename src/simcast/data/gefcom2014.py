"""GEFCom2014 Wind track: 10 wind farms (zones) with normalized hourly power and supplied NWP.

Source: Hugging Face dataset ``rebase-energy/gefcom2014`` (``data/wind_targets.parquet`` and
``data/wind_nwp.parquet``), a repack of the GEFCom2014 wind track.

NWP availability convention (inferred, not documented by the source):

* The repack's ``issue_time``/``knowledge_time`` columns are the four monthly task-release stamps of the
  competition, not physical forecast issuance times, so they are ignored here.
* The NWP has exactly one value per valid hour, with a discontinuity at valid hour 01:00 each day while the
  realized power shows none. That is the signature of stitched daily runs, each covering valid hours
  ``(S, S + 24 h]`` (e.g. 01:00 through 24:00 of one day).
* Timestamps are treated as UTC. The instant at which a run became available is not recoverable from the data,
  so it is an explicit, fingerprinted option: a run covering ``(S, S + L]`` is available at ``S``
  (``nwp_run_start_hour_utc`` and ``nwp_run_length_hours``). With the default one origin per day at 00:00, the
  run for the next 24 hours is available at the origin and no earlier run can leak the future.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from simcast.config import DataConfig, ResolvedExperimentConfig
from simcast.data.adapter import DatasetAdapter, EntityFrames, FrameDatasetAdapter
from simcast.types import EntityGroup, EntityMetadata

LOGGER = logging.getLogger(__name__)

ADAPTER_NAME = "gefcom2014_wind"
ENTITY_TYPE = "gefcom_wind"
ZONES = tuple(f"z{index}" for index in range(1, 11))
NWP_VARIABLES = ("u10", "v10", "u100", "v100")
TARGETS_FILE = "data/wind_targets.parquet"
NWP_FILE = "data/wind_nwp.parquet"
OPTION_KEYS = frozenset({"nwp_run_start_hour_utc", "nwp_run_length_hours"})


def allow_patterns(data: DataConfig) -> list[str]:
    del data
    return [TARGETS_FILE, NWP_FILE]


def default_cache_name(data: DataConfig) -> str:
    del data
    return "gefcom2014_wind"


def validate_data_config(data: DataConfig) -> None:
    if data.entity_type != ENTITY_TYPE:
        raise ValueError(f"gefcom2014_wind requires data.entity_type={ENTITY_TYPE!r}")
    if set(data.options) != OPTION_KEYS:
        raise ValueError(f"gefcom2014_wind requires exactly the data.options {sorted(OPTION_KEYS)}")
    start, length = data.options["nwp_run_start_hour_utc"], data.options["nwp_run_length_hours"]
    if not (isinstance(start, int) and not isinstance(start, bool) and 0 <= start < 24):
        raise ValueError("data.options.nwp_run_start_hour_utc must be an integer in [0, 24)")
    if not (isinstance(length, int) and not isinstance(length, bool) and length > 0 and 24 % length == 0):
        raise ValueError("data.options.nwp_run_length_hours must be a positive integer dividing 24")


def read_targets(path: str | Path) -> pd.DataFrame:
    """Hourly normalized power per zone on a UTC index; extra provenance columns are dropped."""

    frame = pd.read_parquet(Path(path))
    if frame.index.name != "TIMESTAMP":
        frame = frame.set_index("TIMESTAMP")
    missing = [zone for zone in ZONES if zone not in frame.columns]
    if missing:
        raise ValueError(f"GEFCom targets are missing zones: {missing}")
    result = frame.loc[:, list(ZONES)].astype("float64")
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, utc=True), name="timestamp")
    if result.index.has_duplicates:
        raise ValueError("GEFCom targets contain duplicate timestamps")
    return result.sort_index()


def read_nwp(path: str | Path) -> pd.DataFrame:
    """NWP columns ``z{k}_{variable}`` on a UTC ``valid_time`` index; the repack's ``issue_time`` is ignored."""

    frame = pd.read_parquet(Path(path))
    if isinstance(frame.index, pd.MultiIndex) or frame.index.name is not None:
        frame = frame.reset_index()
    frame = frame.drop(columns=["issue_time", "knowledge_time"], errors="ignore")
    if "valid_time" not in frame.columns:
        raise ValueError("GEFCom NWP requires a valid_time column or index level")
    frame = frame.set_index(pd.DatetimeIndex(pd.to_datetime(frame.pop("valid_time"), utc=True), name="timestamp"))
    if frame.index.has_duplicates:
        raise ValueError("GEFCom NWP must supply exactly one value per valid time")
    missing = [f"{zone}_{name}" for zone in ZONES for name in NWP_VARIABLES if f"{zone}_{name}" not in frame.columns]
    if missing:
        raise ValueError(f"GEFCom NWP is missing columns: {missing}")
    return frame.sort_index()


def nwp_available_at(valid_times: pd.DatetimeIndex, *, run_start_hour_utc: int, run_length_hours: int) -> pd.Series:
    """Availability instant of the stitched run that covers each valid time (see module docstring)."""

    run_index = (valid_times - pd.Timedelta(hours=run_start_hour_utc + 1)).floor(f"{run_length_hours}h")
    return pd.Series(run_index + pd.Timedelta(hours=run_start_hour_utc), index=valid_times, name="available_at")


def build_zone_frames(
    targets: pd.DataFrame,
    nwp: pd.DataFrame,
    *,
    target_column: str,
    run_start_hour_utc: int,
    run_length_hours: int,
) -> list[EntityFrames]:
    available_at = nwp_available_at(
        pd.DatetimeIndex(nwp.index), run_start_hour_utc=run_start_hour_utc, run_length_hours=run_length_hours
    )
    frames: list[EntityFrames] = []
    for zone in ZONES:
        entity = EntityMetadata(name=zone, group_name=ENTITY_TYPE, description="GEFCom2014 wind farm")
        weather = nwp.loc[:, [f"{zone}_{name}" for name in NWP_VARIABLES]]
        weather = weather.set_axis(list(NWP_VARIABLES), axis="columns").astype("float32")
        weather["available_at"] = available_at
        frames.append(
            EntityFrames(
                entity=entity,
                target=targets[[zone]].rename(columns={zone: target_column}),
                past_weather=weather,
                future_weather=weather,
            )
        )
    return frames


def load_adapter(config: ResolvedExperimentConfig) -> DatasetAdapter:
    if config.covariates.future_weather_source != "vintage":
        raise ValueError("GEFCom2014 has no realized weather; covariates.future_weather_source must be 'vintage'")
    unknown = [name for name in config.covariates.weather if name not in NWP_VARIABLES]
    if unknown:
        raise ValueError(f"unknown GEFCom NWP variables {unknown}; available: {list(NWP_VARIABLES)}")
    root = Path(config.data.local_dir).expanduser().resolve()
    options: dict[str, Any] = config.data.options
    frames = build_zone_frames(
        read_targets(root / TARGETS_FILE),
        read_nwp(root / NWP_FILE),
        target_column=config.data.target_column,
        run_start_hour_utc=options["nwp_run_start_hour_utc"],
        run_length_hours=options["nwp_run_length_hours"],
    )
    entities = [frame.entity for frame in frames]
    group = EntityGroup(
        group_id=ENTITY_TYPE,
        entity_ids=[entity.entity_id for entity in entities],
        metadata={"entity_type": ENTITY_TYPE, "entity_count": len(entities)},
        entities=entities,
    )
    provenance = {
        "adapter": ADAPTER_NAME,
        "source": {"repository": config.data.dataset_id, "revision": config.data.revision},
        "files": [TARGETS_FILE, NWP_FILE],
        "local_dir": str(root),
        "timestamps": "treated as UTC; hourly",
        "target": "normalized wind power in [0, 1]",
        "nwp_variables": list(NWP_VARIABLES),
        "nwp_availability": {
            "rule": "run covering valid hours (S, S + length] is available at S; repack issue_time ignored",
            "inferred": True,
            **options,
        },
    }
    LOGGER.info("Loaded GEFCom2014 wind: %d zones, NWP runs %s", len(entities), options)
    return FrameDatasetAdapter(config, group, frames, provenance, past_weather_source="versioned")
