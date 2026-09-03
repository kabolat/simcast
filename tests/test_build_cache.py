from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import simcast.cli.build_cache as cache_cli
from simcast.config import SimcastConfig
from simcast.fm.cache import load_pit_library
from simcast.types import EntityGroup, EntityMetadata


@dataclass
class _MockForecast:
    entity_ids: list[str]
    quantile_levels: torch.Tensor
    quantile_predictions: torch.Tensor
    forecast_embeddings: torch.Tensor
    output_patch_size: int


class _MockForecaster:
    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self,
        entity_ids,
        inputs,
        prediction_length,
        *,
        batch_size=256,
        context_length=None,
    ) -> _MockForecast:
        del batch_size, context_length
        self.calls += 1
        medians = torch.tensor([float(np.asarray(item["target"])[-1]) for item in inputs])
        medians = medians[:, None].expand(-1, prediction_length)
        predictions = torch.stack((medians - 1.0, medians - 0.5, medians, medians + 0.5, medians + 1.0), -1)
        return _MockForecast(
            entity_ids=list(entity_ids),
            quantile_levels=torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]),
            quantile_predictions=predictions,
            forecast_embeddings=torch.full((len(entity_ids), 1, 4), float(self.calls)),
            output_patch_size=2,
        )


def _config(tmp_path: Path) -> SimcastConfig:
    return SimcastConfig.model_validate(
        {
            "data": {"local_dir": str(tmp_path), "entity_type": "transformer"},
            "forecast": {
                "frequency_minutes": 15,
                "origin_time": "00:45",
                "lookback_steps": 4,
                "horizon_steps": 2,
                "origin_stride_steps": 2,
            },
            "covariates": {"weather": ["temperature_2m"], "calendar": {"enabled": False}},
            "split": {"tune_fraction": 0.8, "validation_fraction_within_tune": 0.2},
            "chronos": {"batch_size": 16},
            "evaluation": {"interval_levels": [0.5, 0.8]},
            "output": {"cache_dir": str(tmp_path / "cache")},
        }
    )


def _fixtures() -> tuple[EntityGroup, list[cache_cli._EntityFrames]]:
    timestamps = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC", name="timestamp")
    entities = [
        EntityMetadata(name=name, group_name="transformer", latitude=52.0 + idx, longitude=5.0 + idx)
        for idx, name in enumerate(("a", "b", "c"))
    ]
    group = EntityGroup(
        group_id="transformer",
        entity_ids=[entity.entity_id for entity in entities],
        metadata={"entity_type": "transformer"},
        entities=entities,
    )
    frames = []
    for idx, entity in enumerate(entities):
        target = pd.DataFrame(
            {"load": np.arange(len(timestamps), dtype=np.float32) + idx, "available_at": timestamps},
            index=timestamps,
        )
        measured = pd.DataFrame({"temperature_2m": np.arange(len(timestamps))}, index=timestamps)
        versioned = pd.DataFrame(
            {
                "temperature_2m": np.arange(len(timestamps), dtype=np.float32),
                "available_at": timestamps - pd.Timedelta(days=1),
            },
            index=timestamps,
        )
        frames.append(cache_cli._EntityFrames(entity, target, measured, versioned))
    return group, frames


def test_mocked_cache_pipeline_seals_test_truth(monkeypatch, tmp_path: Path) -> None:
    group, frames = _fixtures()
    monkeypatch.setattr(cache_cli, "build_entity_group", lambda *_args, **_kwargs: group)
    monkeypatch.setattr(cache_cli, "_load_frames", lambda *_args, **_kwargs: frames)
    forecaster = _MockForecaster()

    destination = cache_cli.build_cache_from_config(
        _config(tmp_path),
        forecaster=forecaster,
        output_dir=tmp_path / "library",
    )

    training = load_pit_library(destination)
    assert training.dataset.sizes == {"origin": 4, "entity": 3, "lead": 2, "quantile": 5, "patch": 1, "hidden": 4}
    assert list(training.dataset["split"].values) == ["train", "train", "train", "test"]
    assert np.isnan(training.dataset["true_y"].isel(origin=-1)).all()
    assert forecaster.calls == 4
    evaluation = load_pit_library(destination, access="evaluation")
    assert np.isfinite(evaluation.test_data()["true_y"]).all()
    assert (destination / "marginal_diagnostics.json").is_file()
    assert (destination / "resolved_config.json").is_file()
