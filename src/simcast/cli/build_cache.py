"""Build the revision-pinned historical Chronos forecast and PIT library."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import floor
from pathlib import Path
from typing import Annotated, Any, Protocol, cast

import numpy as np
import pandas as pd
import torch
import typer
from tqdm.auto import tqdm

from simcast.config import (
    IndependentMethodConfig,
    ResolvedExperimentConfig,
    load_base_config,
    resolve_run_config,
)
from simcast.data.adapter import IneligibleWindow
from simcast.data.registry import adapter_spec
from simcast.data.windows import ChronologicalSplit, build_aligned_group_windows, chronological_split
from simcast.fm.cache import build_cache_dataset, save_pit_library
from simcast.fm.diagnostics import compute_marginal_diagnostics
from simcast.fm.pit import build_group_pit, repair_quantiles_isotonic

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


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _unpurged_split(origins: Sequence[pd.Timestamp], config: ResolvedExperimentConfig) -> ChronologicalSplit:
    tune_count = floor(len(origins) * config.split.tune_fraction)
    validation_count = floor(tune_count * config.split.validation_fraction_within_tune)
    train_count = tune_count - validation_count
    ordered = tuple(origins)
    return ChronologicalSplit(
        train=ordered[:train_count],
        validation=ordered[train_count:tune_count],
        test=ordered[tune_count:],
    )


def _split_origins(origins: Sequence[pd.Timestamp], config: ResolvedExperimentConfig) -> ChronologicalSplit:
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


def _default_forecaster(config: ResolvedExperimentConfig) -> FrozenForecaster:
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


def _cache_destination(config: ResolvedExperimentConfig, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    name = config.output.cache_name or adapter_spec(config.data.adapter).default_cache_name(config.data)
    return (Path(config.output.cache_dir).expanduser() / name).resolve()


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
    config: ResolvedExperimentConfig,
    *,
    forecaster: FrozenForecaster | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Run frozen entity-wise inference and persist the labeled PIT library."""

    destination = _cache_destination(config, output_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing cache: {destination}")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    adapter = adapter_spec(config.data.adapter).load(config)
    group = adapter.group
    if config.protocol.ordered_entity_ids and config.protocol.ordered_entity_ids != group.entity_ids:
        raise ValueError(
            "protocol.ordered_entity_ids does not match the adapter's entity group: "
            f"{config.protocol.ordered_entity_ids} != {group.entity_ids}"
        )
    coverage_start, coverage_end = adapter.coverage()
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
    LOGGER.info(
        "Building %s cache for %d entities across %d candidate origins",
        config.data.entity_type,
        len(group.entity_ids),
        len(candidates),
    )

    model = forecaster or _default_forecaster(config)
    origins: list[pd.Timestamp] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    missing_histories: list[np.ndarray] = []
    forecastable_entities: list[np.ndarray] = []
    quantile_levels: np.ndarray | None = None
    output_patch_size: int | None = None
    skipped: list[dict[str, str]] = []

    for window in tqdm(
        candidates,
        desc=f"Caching {config.data.entity_type}",
        unit="origin",
        dynamic_ncols=True,
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
        try:
            prepared = adapter.prepare_window(window)
        except IneligibleWindow as error:
            skipped.append({"origin_timestamp": window.origin_timestamp.isoformat(), "reason": str(error)})
            LOGGER.warning("Skipping ineligible origin %s: %s", window.origin_timestamp, error)
            continue
        forecastable = np.asarray(prepared.forecastable_entity, dtype=bool)
        entity_indices = np.asarray(prepared.entity_indices, dtype=np.int64)
        if forecastable.shape != (len(group.entity_ids),) or not np.array_equal(
            np.flatnonzero(forecastable), entity_indices
        ):
            raise ValueError("prepared window forecastability does not match its canonical entity indices")
        if entity_indices.size == 0:
            skipped.append(
                {"origin_timestamp": window.origin_timestamp.isoformat(), "reason": "no forecastable entities"}
            )
            LOGGER.warning("Skipping origin %s: no entities have usable forecast inputs", window.origin_timestamp)
            continue
        forecast_entity_ids = [group.entity_ids[index] for index in entity_indices]
        result = model.predict(
            forecast_entity_ids,
            prepared.inputs,
            config.forecast.horizon_steps,
            batch_size=config.chronos.batch_size,
            context_length=config.forecast.lookback_steps,
        )
        forecast, embedding, levels, patch_size = _validate_forecast(
            result,
            entity_ids=forecast_entity_ids,
            horizon=config.forecast.horizon_steps,
        )
        full_forecast = np.full(
            (len(group.entity_ids), *forecast.shape[1:]),
            np.nan,
            dtype=np.float32,
        )
        full_embedding = np.full(
            (len(group.entity_ids), *embedding.shape[1:]),
            np.nan,
            dtype=np.float32,
        )
        full_forecast[entity_indices] = forecast
        full_embedding[entity_indices] = embedding
        if quantile_levels is not None and not np.array_equal(levels, quantile_levels):
            raise ValueError("native quantile levels changed between forecast origins")
        if output_patch_size is not None and patch_size != output_patch_size:
            raise ValueError("output patch size changed between forecast origins")
        if embeddings and full_embedding.shape != embeddings[0].shape:
            raise ValueError("forecast embedding shape changed between forecast origins")
        quantile_levels = levels
        output_patch_size = patch_size
        origins.append(window.origin_timestamp)
        truths.append(prepared.true_y)
        predictions.append(full_forecast)
        embeddings.append(full_embedding)
        missing_histories.append(prepared.history_missing)
        forecastable_entities.append(forecastable)

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
    forecastable_entity = np.stack([forecastable_entities[idx] for idx in kept_indices])

    raw_pit = build_group_pit(
        torch.from_numpy(true_y),
        torch.from_numpy(quantile_predictions),
        torch.from_numpy(quantile_levels),
        mode=config.pit.mode,
        monotone_repair="none",
        eps=config.pit.eps,
        forecastable_entity=torch.from_numpy(forecastable_entity),
        cardinality_mode=config.protocol.cardinality_mode,
        min_entities_per_case=config.protocol.min_entities_per_case,
    )
    pit = (
        raw_pit
        if config.pit.monotone_repair == "none"
        else build_group_pit(
            torch.from_numpy(true_y),
            torch.from_numpy(quantile_predictions),
            torch.from_numpy(quantile_levels),
            mode=config.pit.mode,
            monotone_repair=config.pit.monotone_repair,
            eps=config.pit.eps,
            forecastable_entity=torch.from_numpy(forecastable_entity),
            cardinality_mode=config.protocol.cardinality_mode,
            min_entities_per_case=config.protocol.min_entities_per_case,
        )
    )
    pit_u = pit.u.cpu().numpy().astype(np.float32, copy=False)
    pit_z = pit.z.cpu().numpy().astype(np.float32, copy=False)
    crossing_indices = None
    crossing_raw_quantile_predictions = None
    if config.pit.monotone_repair == "isotonic":
        crossing_indices = np.argwhere(raw_pit.crossing_diagnostics.crossed)
        crossing_raw_quantile_predictions = quantile_predictions[tuple(crossing_indices.T)]
        quantile_predictions = (
            repair_quantiles_isotonic(torch.from_numpy(quantile_predictions)).numpy().astype(np.float32, copy=False)
        )
    valid_origin_lead = pit.valid_origin_lead.cpu().numpy()
    tune_mask = np.asarray(split_labels) != "test"
    tune_valid_origin_lead = valid_origin_lead[tune_mask]
    dropped_cases = int((~tune_valid_origin_lead).sum())
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
        "Excluded %d/%d train/validation origin-lead cases under %s cardinality rules",
        dropped_cases,
        tune_valid_origin_lead.size,
        config.protocol.cardinality_mode,
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

    has_locations = all(entity.latitude is not None and entity.longitude is not None for entity in group.entities)
    latitudes = [float(cast(float, entity.latitude)) for entity in group.entities] if has_locations else None
    longitudes = [float(cast(float, entity.longitude)) for entity in group.entities] if has_locations else None
    dataset = build_cache_dataset(
        origin_timestamps=np.asarray([origin.to_datetime64() for origin in kept_origins]),
        entity_ids=group.entity_ids,
        true_y=true_y,
        quantile_predictions=quantile_predictions,
        crossing_indices=crossing_indices,
        crossing_raw_quantile_predictions=crossing_raw_quantile_predictions,
        pit_u=pit_u,
        pit_z=pit_z,
        forecast_embeddings=forecast_embeddings,
        forecastable_entity=forecastable_entity,
        quantile_levels=quantile_levels,
        split=split_labels,
        latitude=latitudes,
        longitude=longitudes,
    )
    dataset["pit_valid"] = (("origin", "lead"), valid_origin_lead)
    dataset.attrs.update(
        {
            "output_patch_size": output_patch_size,
            "entity_type": config.data.entity_type,
            "timezone": "UTC",
        }
    )

    truth_missing_by_entity = np.isnan(true_y[tune_mask]).sum(axis=(0, 2)).astype(int)
    metadata: dict[str, Any] = {
        "dataset": {
            "adapter": config.data.adapter,
            "id": config.data.dataset_id,
            "revision": config.data.revision,
            "provenance": dict(adapter.provenance),
        },
        "entity_group": {
            "group_id": group.group_id,
            "entity_type": config.data.entity_type,
            "entity_ids": group.entity_ids,
            "entity_names": [entity.name for entity in group.entities],
            "latitude": latitudes,
            "longitude": longitudes,
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
            "timezone": "UTC",
            "coverage_start": coverage_start.isoformat(),
            "coverage_end": coverage_end.isoformat(),
        },
        "covariates": config.covariates.model_dump(mode="json"),
        "protocol": {
            "cardinality_mode": config.protocol.cardinality_mode,
            "min_entities_per_case": config.protocol.min_entities_per_case,
        },
        "pit": {
            "method": config.pit.mode,
            "monotone_repair": config.pit.monotone_repair,
            "eps": config.pit.eps,
            "crossing_frequency": diagnostics.overall,
            "crossing_frequency_by_entity": crossing_by_entity,
            "crossing_frequency_by_lead": diagnostics.by_lead.tolist(),
            "excluded_train_validation_case_count": dropped_cases,
            "excluded_train_validation_case_fraction": dropped_cases / tune_valid_origin_lead.size,
            "valid_train_validation_case_count_without_repair": int(
                raw_pit.valid_origin_lead.cpu().numpy()[tune_mask].sum()
            ),
            "valid_train_validation_case_count_with_configured_repair": int(tune_valid_origin_lead.sum()),
        },
        "missingness": {
            "masked_history_count_by_entity": dict(
                zip(group.entity_ids, history_missing.sum(axis=0).astype(int).tolist(), strict=True)
            ),
            "future_truth_count_by_entity": dict(zip(group.entity_ids, truth_missing_by_entity.tolist(), strict=True)),
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
    marginal_report = asdict(compute_marginal_diagnostics(
        true_y[tune_mask],
        quantile_predictions[tune_mask],
        quantile_levels,
        pit_u[tune_mask],
        entity_ids=group.entity_ids,
        interval_levels=config.evaluation.interval_levels,
        pit_mode=config.pit.mode,
    ))
    marginal_payload = {"scope": "train_and_validation", **marginal_report}

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
    base_path: str | Path,
    *,
    overrides: Sequence[str] = (),
    forecaster: FrozenForecaster | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Load a base configuration and build its historical FM/PIT cache."""

    base = load_base_config(base_path, overrides=overrides)
    return build_cache_from_config(
        resolve_run_config(base, IndependentMethodConfig(kind="method", id="m0", family="independent")),
        forecaster=forecaster,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def main(
    base: Annotated[
        Path,
        typer.Option(
            "--base",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Base YAML defining the frozen Chronos marginal and PIT cache.",
        ),
    ],
    override: Annotated[
        list[str] | None,
        typer.Option("--set", help="Configuration override as dotted.path=value"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", file_okay=False, help="Explicit cache directory; otherwise use the canonical path."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing cache directory deliberately."),
    ] = False,
) -> None:
    """Build frozen Chronos forecasts, embeddings, and configured PIT scores."""

    base_config = load_base_config(base, overrides=override or ())
    logging.basicConfig(
        level=getattr(logging, base_config.runtime.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    path = build_cache(base, overrides=override or (), output_dir=output_dir, overwrite=overwrite)
    typer.echo(path)


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
