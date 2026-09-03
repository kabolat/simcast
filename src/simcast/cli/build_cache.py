"""Build the revision-pinned historical Chronos forecast and PIT library."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Annotated, Any, Protocol, cast

import numpy as np
import pandas as pd
import torch
import typer

from simcast.config import SimcastConfig, load_config
from simcast.data.availability import select_available_past_targets
from simcast.data.covariates import build_future_covariates, build_past_covariates
from simcast.data.grouping import build_entity_group
from simcast.data.liander2024 import load_entity_frame
from simcast.data.windows import ChronologicalSplit, ForecastWindow, build_aligned_group_windows, chronological_split
from simcast.fm.cache import build_cache_dataset, save_pit_library
from simcast.fm.diagnostics import compute_marginal_diagnostics
from simcast.fm.pit import build_group_pit
from simcast.types import EntityGroup, EntityMetadata

LOGGER = logging.getLogger(__name__)


class ForecastResult(Protocol):
    """Output contract shared by the real and mock frozen forecasters."""

    entity_ids: list[str]
    quantile_levels: torch.Tensor
    quantile_predictions: torch.Tensor
    forecast_embeddings: torch.Tensor
    output_patch_size: int


class FrozenForecaster(Protocol):
    """Minimal injectable Chronos wrapper contract used by cache construction."""

    def predict(
        self,
        entity_ids: Sequence[str],
        inputs: Sequence[Mapping[str, object]],
        prediction_length: int,
        *,
        batch_size: int = 256,
        context_length: int | None = None,
    ) -> ForecastResult: ...


@dataclass(frozen=True, slots=True)
class _EntityFrames:
    entity: EntityMetadata
    target: pd.DataFrame
    measured_weather: pd.DataFrame
    forecast_weather: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _PreparedWindow:
    inputs: list[dict[str, object]]
    true_y: np.ndarray
    history_missing: np.ndarray


class _IneligibleWindow(ValueError):
    pass


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _window_slice(frame: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Limit work before applying the availability selectors."""

    return frame.loc[timestamps[0] : timestamps[-1]]


def _finite_covariates(frame: pd.DataFrame) -> bool:
    return bool(np.isfinite(frame.to_numpy(dtype=np.float64, copy=False)).all())


def _covariate_kwargs(config: SimcastConfig) -> dict[str, bool]:
    calendar = config.covariates.calendar
    return {
        "calendar": calendar.enabled,
        "include_hour": calendar.include_hour,
        "include_day_of_week": calendar.include_day_of_week,
        "include_day_of_year": calendar.include_day_of_year,
    }


def _prepare_entity(
    frames: _EntityFrames,
    window: ForecastWindow,
    config: SimcastConfig,
) -> tuple[dict[str, object], np.ndarray, int]:
    target_window = frames.target.reindex(window.past_timestamps.union(window.future_timestamps))
    history = select_available_past_targets(
        target_window.loc[window.past_timestamps],
        window.origin_timestamp,
        entity_type=frames.entity.group_name,
        target_column=config.data.target_column,
    ).reindex(window.past_timestamps)
    truth = target_window.loc[window.future_timestamps, config.data.target_column]

    kwargs = _covariate_kwargs(config)
    measured = _window_slice(frames.measured_weather, window.past_timestamps)
    forecasts = _window_slice(frames.forecast_weather, window.future_timestamps)
    past_covariates = build_past_covariates(
        measured,
        window.past_timestamps,
        config.covariates.weather,
        origin_timestamp=window.origin_timestamp,
        **kwargs,
    )
    future_covariates = build_future_covariates(
        forecasts,
        window.origin_timestamp,
        window.future_timestamps,
        config.covariates.weather,
        **kwargs,
    )
    if not _finite_covariates(past_covariates) or not _finite_covariates(future_covariates):
        raise _IneligibleWindow(f"covariates unavailable for {frames.entity.entity_id}")
    if list(past_covariates.columns) != list(future_covariates.columns):
        raise _IneligibleWindow(f"past/future covariate columns differ for {frames.entity.entity_id}")

    prepared = {
        "target": history.to_numpy(dtype=np.float32, copy=True),
        "past_covariates": {
            str(column): past_covariates[column].to_numpy(dtype=np.float32, copy=True)
            for column in past_covariates.columns
        },
        "future_covariates": {
            str(column): future_covariates[column].to_numpy(dtype=np.float32, copy=True)
            for column in future_covariates.columns
        },
    }
    return (
        prepared,
        truth.to_numpy(dtype=np.float32, copy=True),
        int(history.isna().sum()),
    )


def _prepare_window(
    frames: Sequence[_EntityFrames],
    window: ForecastWindow,
    config: SimcastConfig,
) -> _PreparedWindow:
    inputs: list[dict[str, object]] = []
    truth: list[np.ndarray] = []
    history_missing: list[int] = []
    for entity_frames in frames:
        entity_input, entity_truth, missing_count = _prepare_entity(entity_frames, window, config)
        inputs.append(entity_input)
        truth.append(entity_truth)
        history_missing.append(missing_count)
    return _PreparedWindow(
        inputs=inputs,
        true_y=np.stack(truth),
        history_missing=np.asarray(history_missing, dtype=np.int64),
    )


def _load_frames(root: Path, group: EntityGroup) -> list[_EntityFrames]:
    return [
        _EntityFrames(
            entity=entity,
            target=load_entity_frame(root, "load_measurements", entity),
            measured_weather=load_entity_frame(root, "weather_measurements", entity),
            forecast_weather=load_entity_frame(root, "weather_forecasts_versioned", entity),
        )
        for entity in group.entities
    ]


def _unpurged_split(origins: Sequence[pd.Timestamp], config: SimcastConfig) -> ChronologicalSplit:
    tune_count = floor(len(origins) * config.split.tune_fraction)
    validation_count = floor(tune_count * config.split.validation_fraction_within_tune)
    train_count = tune_count - validation_count
    ordered = tuple(origins)
    return ChronologicalSplit(
        train=ordered[:train_count],
        validation=ordered[train_count:tune_count],
        test=ordered[tune_count:],
    )


def _split_origins(origins: Sequence[pd.Timestamp], config: SimcastConfig) -> ChronologicalSplit:
    if not config.split.purge_overlapping_horizons:
        return _unpurged_split(origins, config)
    return chronological_split(
        origins,
        horizon_steps=config.forecast.horizon_steps,
        frequency_minutes=config.forecast.frequency_minutes,
        tune_fraction=config.split.tune_fraction,
        validation_fraction_within_tune=config.split.validation_fraction_within_tune,
    )


def _split_lookup(split: ChronologicalSplit) -> dict[pd.Timestamp, str]:
    return {
        **dict.fromkeys(split.train, "train"),
        **dict.fromkeys(split.validation, "validation"),
        **dict.fromkeys(split.test, "test"),
    }


def _default_forecaster(config: SimcastConfig) -> FrozenForecaster:
    from chronos import Chronos2Pipeline  # type: ignore[import-untyped]

    from simcast.fm.chronos2_features import Chronos2FeatureExtractor

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.chronos.dtype]
    pipeline = Chronos2Pipeline.from_pretrained(
        config.chronos.model_id,
        revision=config.chronos.model_revision,
        device_map=config.chronos.device,
        dtype=dtype,
    )
    return cast(FrozenForecaster, Chronos2FeatureExtractor(pipeline))


def _cache_destination(config: SimcastConfig, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    name = config.output.experiment_name or f"liander2024_{config.data.entity_type}"
    return (Path(config.output.cache_dir).expanduser() / name).resolve()


def _coverage_and_missingness(
    frames: Sequence[_EntityFrames], target_column: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    coverage_start = max(frame.target.index.min() for frame in frames)
    coverage_end = min(frame.target.index.max() for frame in frames)
    total_rows = sum(len(frame.target) for frame in frames)
    missing = sum(int(frame.target[target_column].isna().sum()) for frame in frames)
    LOGGER.info(
        "Aligned target coverage %s to %s; %d rows and %.4f%% missing observations",
        coverage_start,
        coverage_end,
        total_rows,
        100.0 * missing / total_rows,
    )
    return pd.Timestamp(coverage_start), pd.Timestamp(coverage_end)


def _validate_forecast(
    result: ForecastResult,
    *,
    entity_ids: list[str],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if list(result.entity_ids) != entity_ids:
        raise ValueError("forecaster returned entities in a different order")
    levels = _as_numpy(result.quantile_levels).astype(np.float32, copy=False)
    predictions = _as_numpy(result.quantile_predictions).astype(np.float32, copy=False)
    embeddings = _as_numpy(result.forecast_embeddings).astype(np.float32, copy=False)
    if levels.ndim != 1 or levels.size == 0 or np.any(np.diff(levels) <= 0):
        raise ValueError("forecaster returned invalid native quantile levels")
    expected_prediction_prefix = (len(entity_ids), horizon)
    if predictions.ndim != 3 or predictions.shape[:2] != expected_prediction_prefix:
        raise ValueError("forecaster returned quantiles with an invalid shape")
    if predictions.shape[-1] != levels.size:
        raise ValueError("forecaster quantile axis does not match native levels")
    if embeddings.ndim != 3 or embeddings.shape[0] != len(entity_ids):
        raise ValueError("forecaster returned embeddings with an invalid shape")
    if result.output_patch_size <= 0:
        raise ValueError("forecaster returned an invalid output patch size")
    return predictions, embeddings, levels, int(result.output_patch_size)


def build_cache_from_config(
    config: SimcastConfig,
    *,
    forecaster: FrozenForecaster | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Run frozen entity-wise inference and persist the labeled PIT library."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    root = Path(config.data.local_dir).expanduser().resolve()
    group = build_entity_group(root / "liander2024_targets.yaml", config.data.entity_type)
    frames = _load_frames(root, group)
    coverage_start, coverage_end = _coverage_and_missingness(frames, config.data.target_column)
    candidates = build_aligned_group_windows(
        coverage_start,
        coverage_end,
        entity_ids=group.entity_ids,
        lookback_steps=config.forecast.lookback_steps,
        horizon_steps=config.forecast.horizon_steps,
        frequency_minutes=config.forecast.frequency_minutes,
        origin_stride_steps=config.forecast.origin_stride_steps,
        origin_time=config.forecast.origin_time,
    )
    if not candidates:
        raise ValueError("no candidate forecast origins fit the configured lookback and horizon")

    model = forecaster or _default_forecaster(config)
    origins: list[pd.Timestamp] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    missing_histories: list[np.ndarray] = []
    quantile_levels: np.ndarray | None = None
    output_patch_size: int | None = None
    skipped: list[dict[str, str]] = []

    for window in candidates:
        try:
            prepared = _prepare_window(frames, window, config)
        except _IneligibleWindow as error:
            skipped.append({"origin_timestamp": window.origin_timestamp.isoformat(), "reason": str(error)})
            LOGGER.warning("Skipping ineligible origin %s: %s", window.origin_timestamp, error)
            continue
        result = model.predict(
            group.entity_ids,
            prepared.inputs,
            config.forecast.horizon_steps,
            batch_size=config.chronos.batch_size,
            context_length=config.forecast.lookback_steps,
        )
        forecast, embedding, levels, patch_size = _validate_forecast(
            result,
            entity_ids=group.entity_ids,
            horizon=config.forecast.horizon_steps,
        )
        if quantile_levels is not None and not np.array_equal(levels, quantile_levels):
            raise ValueError("native quantile levels changed between forecast origins")
        if output_patch_size is not None and patch_size != output_patch_size:
            raise ValueError("output patch size changed between forecast origins")
        if embeddings and embedding.shape != embeddings[0].shape:
            raise ValueError("forecast embedding shape changed between forecast origins")
        quantile_levels = levels
        output_patch_size = patch_size
        origins.append(window.origin_timestamp)
        truths.append(prepared.true_y)
        predictions.append(forecast)
        embeddings.append(embedding)
        missing_histories.append(prepared.history_missing)

    if not origins or quantile_levels is None or output_patch_size is None:
        raise ValueError("no aligned forecast origins have complete available covariates")

    split = _split_origins(origins, config)
    split_by_origin = _split_lookup(split)
    kept_indices = [idx for idx, origin in enumerate(origins) if origin in split_by_origin]
    if not kept_indices:
        raise ValueError("chronological splitting and purging removed every origin")
    kept_origins = [origins[idx] for idx in kept_indices]
    split_labels = [split_by_origin[origin] for origin in kept_origins]
    true_y = np.stack([truths[idx] for idx in kept_indices]).astype(np.float32, copy=False)
    quantile_predictions = np.stack([predictions[idx] for idx in kept_indices]).astype(np.float32, copy=False)
    forecast_embeddings = np.stack([embeddings[idx] for idx in kept_indices]).astype(np.float16)
    history_missing = np.stack([missing_histories[idx] for idx in kept_indices])

    pit = build_group_pit(
        torch.from_numpy(true_y),
        torch.from_numpy(quantile_predictions),
        torch.from_numpy(quantile_levels),
        monotone_repair=config.pit.monotone_repair,
        eps=config.pit.eps,
    )
    pit_u = pit.u.cpu().numpy().astype(np.float32, copy=False)
    pit_z = pit.z.cpu().numpy().astype(np.float32, copy=False)
    valid_origin_lead = pit.valid_origin_lead.cpu().numpy()
    dropped_vectors = int((~valid_origin_lead).sum())
    diagnostics = pit.crossing_diagnostics
    crossing_by_entity = {
        entity_id: float(diagnostics.by_entity[idx]) for idx, entity_id in enumerate(group.entity_ids)
    }
    LOGGER.info(
        "Quantile crossing frequency %.6f; by entity: %s",
        diagnostics.overall,
        crossing_by_entity,
    )
    LOGGER.info("Quantile crossing frequency by lead: %s", diagnostics.by_lead.tolist())
    LOGGER.info(
        "Dropped %d/%d complete (origin, lead) PIT vectors due to missing truth, non-finite forecasts, or crossings",
        dropped_vectors,
        valid_origin_lead.size,
    )
    LOGGER.info(
        "Retained origins %s to %s; train=%d validation=%d test=%d; masked historical targets by entity=%s",
        kept_origins[0],
        kept_origins[-1],
        len(split.train),
        len(split.validation),
        len(split.test),
        dict(zip(group.entity_ids, history_missing.sum(axis=0).tolist(), strict=True)),
    )

    dataset = build_cache_dataset(
        origin_timestamps=np.asarray([origin.to_datetime64() for origin in kept_origins]),
        entity_ids=group.entity_ids,
        true_y=true_y,
        quantile_predictions=quantile_predictions,
        pit_u=pit_u,
        pit_z=pit_z,
        forecast_embeddings=forecast_embeddings,
        quantile_levels=quantile_levels,
        split=split_labels,
        latitude=[entity.latitude for entity in group.entities],
        longitude=[entity.longitude for entity in group.entities],
    )
    dataset["pit_valid"] = (("origin", "lead"), valid_origin_lead)
    dataset.attrs.update(
        {
            "output_patch_size": output_patch_size,
            "entity_type": config.data.entity_type,
            "timezone": "UTC",
        }
    )

    truth_missing_by_entity = np.isnan(true_y).sum(axis=(0, 2)).astype(int)
    metadata: dict[str, Any] = {
        "dataset": {
            "id": config.data.dataset_id,
            "revision": config.data.revision,
            "local_dir": str(root),
        },
        "entity_group": {
            "group_id": group.group_id,
            "entity_type": config.data.entity_type,
            "entity_ids": group.entity_ids,
            "entity_names": [entity.name for entity in group.entities],
            "latitude": [entity.latitude for entity in group.entities],
            "longitude": [entity.longitude for entity in group.entities],
        },
        "chronos": {
            "repository_commit": config.chronos.source_revision,
            "model_id": config.chronos.model_id,
            "model_revision": config.chronos.model_revision,
            "native_quantile_levels": quantile_levels.tolist(),
            "output_patch_size": output_patch_size,
        },
        "forecast_protocol": {
            "lookback_steps": config.forecast.lookback_steps,
            "horizon_steps": config.forecast.horizon_steps,
            "frequency_minutes": config.forecast.frequency_minutes,
            "origin_stride_steps": config.forecast.origin_stride_steps,
            "origin_time": config.forecast.origin_time.isoformat(),
            "timezone": config.forecast.timezone,
            "coverage_start": coverage_start.isoformat(),
            "coverage_end": coverage_end.isoformat(),
        },
        "covariates": config.covariates.model_dump(mode="json"),
        "pit": {
            "method": config.pit.mode,
            "monotone_repair": config.pit.monotone_repair,
            "eps": config.pit.eps,
            "crossing_frequency": diagnostics.overall,
            "crossing_frequency_by_entity": crossing_by_entity,
            "crossing_frequency_by_lead": diagnostics.by_lead.tolist(),
            "dropped_complete_vector_count": dropped_vectors,
            "dropped_complete_vector_fraction": dropped_vectors / valid_origin_lead.size,
        },
        "missingness": {
            "masked_history_count_by_entity": dict(
                zip(group.entity_ids, history_missing.sum(axis=0).astype(int).tolist(), strict=True)
            ),
            "future_truth_count_by_entity": dict(
                zip(group.entity_ids, truth_missing_by_entity.tolist(), strict=True)
            ),
        },
        "origins": {
            "candidate_count": len(candidates),
            "eligible_count": len(origins),
            "retained_count": len(kept_origins),
            "skipped": skipped,
            "train_count": len(split.train),
            "validation_count": len(split.validation),
            "test_count": len(split.test),
            "purged_train": [value.isoformat() for value in split.purged_train],
            "purged_validation": [value.isoformat() for value in split.purged_validation],
        },
        "resolved_config": config.model_dump(mode="json"),
    }

    # Diagnostics created during cache construction are tune-only: test labels
    # remain sealed until the explicit evaluation access path is requested.
    tune_mask = np.asarray(split_labels) != "test"
    marginal_report = compute_marginal_diagnostics(
        true_y[tune_mask],
        quantile_predictions[tune_mask],
        quantile_levels,
        pit_u[tune_mask],
        entity_ids=group.entity_ids,
        interval_levels=config.evaluation.interval_levels,
    ).as_dict()
    marginal_payload = {"scope": "train_and_validation", **marginal_report}

    destination = _cache_destination(config, output_dir)
    save_pit_library(destination, dataset, metadata, overwrite=overwrite)
    (destination / "marginal_diagnostics.json").write_text(
        json.dumps(marginal_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if config.output.save_resolved_config:
        (destination / "resolved_config.json").write_text(
            json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    LOGGER.info("Saved PIT library to %s", destination)
    return destination


def build_cache(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
    forecaster: FrozenForecaster | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Load a YAML configuration and build its historical FM/PIT cache."""

    return build_cache_from_config(
        load_config(config_path, overrides=overrides),
        forecaster=forecaster,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False, readable=True)],
    override: Annotated[
        list[str] | None,
        typer.Option("--set", help="Configuration override as dotted.path=value"),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Build frozen Chronos forecasts, embeddings, and discretized PIT scores."""

    resolved = load_config(config, overrides=override or ())
    logging.basicConfig(
        level=getattr(logging, resolved.runtime.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    path = build_cache_from_config(resolved, output_dir=output_dir, overwrite=overwrite)
    typer.echo(path)


if __name__ == "__main__":
    typer.run(main)
