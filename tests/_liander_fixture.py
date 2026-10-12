"""Deterministic synthetic Liander2024 snapshot used to pin adapter backwards compatibility."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from simcast.config import ResolvedExperimentConfig

ENTITIES = (("transformer", "A"), ("transformer", "B"), ("transformer", "C"), ("solar_park", "S"))
STEPS = 288  # three days of 15-minute data


def liander_fixture_config(root: Path) -> ResolvedExperimentConfig:
    return ResolvedExperimentConfig.model_validate(
        {
            "data": {"local_dir": str(root), "entity_type": "transformer"},
            "forecast": {
                "frequency_minutes": 15,
                "origin_time": "00:00",
                "lookback_steps": 8,
                "horizon_steps": 4,
                "origin_stride_steps": 4,
            },
            "covariates": {"weather": ["temperature_2m", "cloud_cover"], "calendar": {"enabled": True}},
        }
    )


def write_liander_fixture(root: Path) -> None:
    """Write targets, load, measured weather, and versioned weather forecasts under ``root``."""

    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-01", periods=STEPS, freq="15min", tz="UTC")
    lines = []
    for position, (group, name) in enumerate(ENTITIES):
        lines.append(
            f"- name: {name}\n  group_name: {group}\n  latitude: {52.0 + position}\n  longitude: {5.0 + position}\n"
        )
    (root / "liander2024_targets.yaml").write_text("".join(lines), encoding="utf-8")
    for position, (group, name) in enumerate(ENTITIES):
        for component in ("load_measurements", "weather_measurements", "weather_forecasts_versioned"):
            (root / component / group).mkdir(parents=True, exist_ok=True)
        load = rng.normal(10.0 + position, 1.0, STEPS).astype(np.float32)
        if name == "B":
            load[::37] = np.nan
        pd.DataFrame({"timestamp": index, "load": load}).to_parquet(
            root / "load_measurements" / group / f"{name}.parquet", index=False
        )
        measured = pd.DataFrame(
            {
                "timestamp": index,
                "temperature_2m": rng.normal(5.0, 2.0, STEPS),
                "cloud_cover": rng.uniform(0.0, 100.0, STEPS),
                "wind_speed_10m": rng.uniform(0.0, 12.0, STEPS),
            }
        )
        measured.to_parquet(root / "weather_measurements" / group / f"{name}.parquet", index=False)
        vintages = []
        for lag_hours, offset in ((6, 0.0), (3, 0.5)):
            vintages.append(
                pd.DataFrame(
                    {
                        "timestamp": index,
                        "available_at": index - pd.Timedelta(hours=lag_hours),
                        "temperature_2m": measured["temperature_2m"].to_numpy() + offset,
                        "cloud_cover": measured["cloud_cover"].to_numpy() - offset,
                        "wind_speed_10m": measured["wind_speed_10m"].to_numpy(),
                    }
                )
            )
        forecasts = pd.concat(vintages, ignore_index=True)
        if name == "C":  # a forecast outage makes every window that touches it ineligible
            outage = (forecasts["timestamp"] >= index[100]) & (forecasts["timestamp"] <= index[120])
            forecasts = forecasts.loc[~outage]
        forecasts.to_parquet(root / "weather_forecasts_versioned" / group / f"{name}.parquet", index=False)


def digest_windows(
    windows: Sequence[object],
    inputs_for: object,
) -> str:
    """Hash every resolved model input (or ineligibility reason) of a sequence of windows."""

    digest = hashlib.sha256()
    for window in windows:
        resolved = inputs_for(window)  # type: ignore[operator]
        if isinstance(resolved, str):
            digest.update(b"ineligible:" + resolved.encode())
            continue
        inputs, truth, missing = resolved
        for entity_input in inputs:
            digest.update(np.ascontiguousarray(entity_input["target"], dtype=np.float32).tobytes())
            for section in ("past_covariates", "future_covariates"):
                for key, values in entity_input[section].items():
                    digest.update(key.encode())
                    digest.update(np.ascontiguousarray(values, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(truth, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(missing, dtype=np.int64).tobytes())
    return digest.hexdigest()
