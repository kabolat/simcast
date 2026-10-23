"""Synthetic staggered-lifetime sources for variable-cardinality tests.

Entity ``k`` has target rows only on its inclusive step-index ``lifetime``; optional ``gaps`` blank target
values inside that lifetime. Weather either follows the lifetime (the entity does not exist outside it) or
spans the whole timeline (only the target is absent). The mock forecaster never runs Chronos and records
exactly which compact entity inputs it receives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

import simcast.cli.build_cache as cache_cli
from simcast.config import ResolvedExperimentConfig
from simcast.data.adapter import EntityFrames, FrameDatasetAdapter
from simcast.types import EntityGroup, EntityMetadata

START = pd.Timestamp("2024-01-01", tz="UTC")
STEP = pd.Timedelta(minutes=15)
LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
OFFSETS = (-1.645, -1.2816, -0.6745, 0.0, 0.6745, 1.2816, 1.645)


def timestamp(index: int) -> pd.Timestamp:
    return START + index * STEP


def staggered_config(
    tmp_path: Path,
    *,
    cardinality_mode: str,
    entity_count: int,
    lookback_steps: int = 4,
    horizon_steps: int = 2,
    origin_stride_steps: int = 2,
    min_entities_per_case: int = 2,
    tune_fraction: float = 0.8,
    validation_fraction_within_tune: float = 0.2,
    pit: Mapping[str, Any] | None = None,
) -> ResolvedExperimentConfig:
    return ResolvedExperimentConfig.model_validate(
        {
            "data": {"local_dir": str(tmp_path), "entity_type": "transformer"},
            "protocol": {
                "ordered_entity_ids": [f"transformer::e{index}" for index in range(entity_count)],
                "entity_count": entity_count,
                "cardinality_mode": cardinality_mode,
                "min_entities_per_case": min_entities_per_case,
            },
            "forecast": {
                "frequency_minutes": 15,
                "origin_time": "00:45",
                "lookback_steps": lookback_steps,
                "horizon_steps": horizon_steps,
                "origin_stride_steps": origin_stride_steps,
            },
            "covariates": {"weather": ["temperature_2m"], "calendar": {"enabled": False}},
            "split": {
                "tune_fraction": tune_fraction,
                "validation_fraction_within_tune": validation_fraction_within_tune,
            },
            "chronos": {"batch_size": 16},
            "pit": dict(pit or {"monotone_repair": "none"}),
            "evaluation": {"interval_levels": [0.5, 0.8]},
            "output": {"cache_dir": str(tmp_path / "cache")},
        }
    )


def staggered_adapter(
    config: ResolvedExperimentConfig,
    lifetimes: Sequence[tuple[int, int]],
    *,
    periods: int,
    gaps: Mapping[int, Sequence[tuple[int, int]]] | None = None,
    weather_follows_lifetime: bool = True,
    seed: int = 7,
) -> FrameDatasetAdapter:
    """Correlated synthetic targets; ``lifetimes`` and ``gaps`` use inclusive step indices."""

    rng = np.random.default_rng(seed)
    timeline = pd.date_range(START, periods=periods, freq=STEP, tz="UTC", name="timestamp")
    common = rng.normal(size=periods).astype(np.float32)
    entities = [
        EntityMetadata(name=f"e{index}", group_name="transformer", latitude=52.0 + index, longitude=5.0 + index)
        for index in range(len(lifetimes))
    ]
    group = EntityGroup(
        group_id="transformer",
        entity_ids=[entity.entity_id for entity in entities],
        metadata={"entity_type": "transformer"},
        entities=entities,
    )
    frames = []
    for index, (entity, (first, last)) in enumerate(zip(entities, lifetimes, strict=True)):
        own = timeline[first : last + 1]
        values = 10.0 + common + 0.5 * rng.normal(size=periods).astype(np.float32)
        target = pd.DataFrame({"load": values[first : last + 1], "available_at": own}, index=own)
        for gap_first, gap_last in (gaps or {}).get(index, ()):
            target.loc[timestamp(gap_first) : timestamp(gap_last), "load"] = np.nan
        weather_index = own if weather_follows_lifetime else timeline
        temperature = np.arange(len(weather_index), dtype=np.float32)
        measured = pd.DataFrame({"temperature_2m": temperature}, index=weather_index)
        versioned = pd.DataFrame(
            {"temperature_2m": temperature, "available_at": weather_index - pd.Timedelta(days=1)},
            index=weather_index,
        )
        frames.append(EntityFrames(entity, target, measured, versioned))
    return FrameDatasetAdapter(config, group, frames, {"adapter": "staggered-test"})


def expected_forecastable(
    origin_index: int,
    lifetime: tuple[int, int],
    *,
    lookback_steps: int = 4,
    horizon_steps: int = 2,
) -> bool:
    """Weather following the lifetime requires the whole lookback and horizon inside it."""

    first, last = lifetime
    return first <= origin_index - lookback_steps + 1 and origin_index + horizon_steps <= last


@dataclass
class _MockForecast:
    entity_ids: list[str]
    quantile_levels: torch.Tensor
    quantile_predictions: torch.Tensor
    forecast_embeddings: torch.Tensor
    output_patch_size: int


@dataclass
class RecordingForecaster:
    """Quantiles centred on the last finite history value; records every compact call."""

    calls: list[list[str]] = field(default_factory=list)
    finite_history_counts: list[list[int]] = field(default_factory=list)

    def predict(
        self,
        entity_ids: Sequence[str],
        inputs: Sequence[Mapping[str, object]],
        prediction_length: int,
        *,
        batch_size: int = 256,
        context_length: int | None = None,
    ) -> _MockForecast:
        del batch_size, context_length
        histories = [np.asarray(item["target"], dtype=np.float32) for item in inputs]
        self.calls.append(list(entity_ids))
        self.finite_history_counts.append([int(np.isfinite(history).sum()) for history in histories])
        medians = torch.tensor([float(history[np.isfinite(history)][-1]) for history in histories])
        offsets = torch.tensor(OFFSETS)
        predictions = medians[:, None, None] + offsets.expand(len(histories), prediction_length, -1)
        return _MockForecast(
            entity_ids=list(entity_ids),
            quantile_levels=torch.tensor(LEVELS),
            quantile_predictions=predictions,
            forecast_embeddings=medians[:, None, None].expand(-1, 1, 4).clone(),
            output_patch_size=2,
        )


def inject_adapter(monkeypatch: Any, adapter: FrameDatasetAdapter) -> None:
    monkeypatch.setattr(cache_cli, "adapter_spec", lambda _name: SimpleNamespace(load=lambda _config: adapter))
