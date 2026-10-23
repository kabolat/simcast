"""Explicit HEAPO source-availability and (access-gated) cached PIT diagnostics."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from simcast.cli.build_cache import _split_lookup, _split_origins
from simcast.config import (
    IndependentMethodConfig,
    ProtocolConfig,
    ResolvedExperimentConfig,
    load_base_config,
    resolve_run_config,
)
from simcast.data.adapter import EntityFrames, FrameDatasetAdapter
from simcast.data.heapo import (
    TARGET_COLUMN,
    archive_members,
    distribution,
    load_adapter,
    read_households,
    read_target,
    read_weather,
    select_households,
    target_summary,
    verify_archive,
)
from simcast.data.windows import ForecastWindow, build_aligned_group_windows
from simcast.fm.cache import load_pit_library
from simcast.fm.pit import training_frequency_fallback_summary
from simcast.types import EntityGroup, EntityMetadata


def _availability(
    adapter: FrameDatasetAdapter,
) -> tuple[list[ForecastWindow], np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of frame preparation for source availability only, not forecast/PIT validity."""

    config = adapter.config
    start, end = adapter.coverage()
    windows = build_aligned_group_windows(
        start, end, entity_ids=adapter.group.entity_ids,
        lookback_steps=config.forecast.lookback_steps, horizon_steps=config.forecast.horizon_steps,
        frequency_minutes=config.forecast.frequency_minutes, origin_stride_steps=config.forecast.origin_stride_steps,
        origin_time=config.forecast.origin_time,
    )
    if not windows:
        raise ValueError("no HEAPO candidate origins fit the configured protocol")
    origins = pd.DatetimeIndex([window.origin_timestamp for window in windows])
    history_start = pd.DatetimeIndex([window.past_timestamps[0] for window in windows])
    future = pd.DatetimeIndex([time for window in windows for time in window.future_timestamps])
    grid = pd.date_range(history_start[0], future[-1], freq=f"{config.forecast.frequency_minutes}min")
    weather_cache: dict[int, np.ndarray] = {}
    forecastable = []
    empty_histories = []
    labels = []
    for frame in adapter.frames:
        finite_times = pd.DatetimeIndex(frame.target.index[np.isfinite(frame.target[TARGET_COLUMN])])
        history_count = finite_times.searchsorted(origins, side="right") - finite_times.searchsorted(history_start)
        empty = history_count == 0
        key = id(frame.past_weather)
        if key not in weather_cache:
            weather = np.isfinite(frame.past_weather.reindex(grid).to_numpy(dtype=float)).all(axis=1)
            complete = pd.Series(weather.astype(int), index=grid)
            past_ok = complete.rolling(config.forecast.lookback_steps).sum().eq(config.forecast.lookback_steps)
            future_ok = complete.rolling(config.forecast.horizon_steps).sum().shift(-config.forecast.horizon_steps)
            weather_cache[key] = (
                past_ok.reindex(origins).fillna(False).to_numpy(dtype=bool)
                & future_ok.reindex(origins).eq(config.forecast.horizon_steps).to_numpy(dtype=bool)
            )
        forecastable.append(~empty & weather_cache[key])
        empty_histories.append(empty)
        labels.append(np.isfinite(frame.target.reindex(future)[TARGET_COLUMN].to_numpy(dtype=float)).reshape(
            len(windows), config.forecast.horizon_steps,
        ))
    forecast = np.stack(forecastable, axis=1)
    active = np.stack(labels, axis=1) & forecast[:, :, None]
    return list(windows), forecast, np.stack(empty_histories, axis=1), active


def _active_report(active: np.ndarray, splits: np.ndarray) -> dict[str, Any]:
    counts = active.sum(axis=1)
    return {
        "overall": distribution(counts.ravel()),
        "by_split": {name: distribution(counts[splits == name].ravel()) for name in ("train", "validation", "test")},
        "by_lead": {str(lead + 1): distribution(counts[:, lead]) for lead in range(counts.shape[1])},
    }


def training_overlap_report(active: np.ndarray, entity_ids: list[str]) -> dict[str, Any]:
    """Count train-only eligible overlaps; low-overlap pairs get M1's independence initialization."""

    import torch

    counts = active.sum(axis=0)
    fallback = training_frequency_fallback_summary(torch.as_tensor(counts == 0), entity_ids)
    pairs = np.triu_indices(len(entity_ids), k=1)
    overlaps = np.stack([
        active[:, :, lead].astype(np.int64).T @ active[:, :, lead].astype(np.int64)
        for lead in range(active.shape[-1])
    ])
    return {
        "households_without_training_observations": [
            entity_ids[index] for index in range(len(entity_ids)) if counts[index].sum() == 0
        ],
        "training_frequency_fallback": fallback,
        "pairwise_overlap_counts": overlaps.tolist(),
        "zero_overlap_pairs_by_lead": [(overlap[pairs] == 0).sum().item() for overlap in overlaps],
        "m1_low_overlap_independence_pairs_by_lead": [(overlap[pairs] < 2).sum().item() for overlap in overlaps],
        "independence_note": (
            "Counts are unordered pairs per lead before PSD repair. M1 also initializes constant-score pairs "
            "to independence; inspect the fitted m1_diagnostics.json and model for final PIT-based estimates."
        ),
    }


def source_diagnostics(adapter: FrameDatasetAdapter, *, include_training_report: bool = True) -> dict[str, Any]:
    windows, forecastable, empty, active = _availability(adapter)
    config = adapter.config
    retained = forecastable.any(axis=1) if config.protocol.cardinality_mode == "variable" else forecastable.all(axis=1)
    retained_origins = [window.origin_timestamp for window, keep in zip(windows, retained, strict=True) if keep]
    split = _split_origins(retained_origins, config)
    lookup = _split_lookup(split)
    splits = np.asarray([lookup.get(window.origin_timestamp, "excluded") for window in windows])
    eligible = active.sum(axis=1) >= config.protocol.min_entities_per_case
    if config.protocol.cardinality_mode == "fixed":
        eligible = active.all(axis=1)
    admitted = active & eligible[:, None, :] & (splits != "excluded")[:, None, None]
    ids = adapter.group.entity_ids
    train_origins = [
        window.origin_timestamp for window, label in zip(windows, splits, strict=True) if label == "train"
    ]
    late_joiners = [
        frame.entity.entity_id for frame in adapter.frames
        if train_origins and frame.target.index.min() > max(train_origins)
    ]
    households = []
    for index, frame in enumerate(adapter.frames):
        forecast_rows = np.flatnonzero(forecastable[:, index])
        households.append({
            "entity_id": ids[index], **target_summary(frame.target),
            "forecast_entry": windows[forecast_rows[0]].origin_timestamp.isoformat() if len(forecast_rows) else None,
            "forecast_exit": windows[forecast_rows[-1]].origin_timestamp.isoformat() if len(forecast_rows) else None,
            "active_fraction": float(admitted[:, index].mean()),
            "empty_history_exclusions": int(empty[:, index].sum()),
            "covariate_exclusions": int((~empty[:, index] & ~forecastable[:, index]).sum()),
            "forecastable_target_label_missing_count": int(
                (forecastable[:, index, None] & ~active[:, index]).sum(),
            ),
        })
    return {
        "scope": "source_availability_only_not_model_inputs",
        "active_definition": "forecastable inputs AND finite target label; not final marginal/PIT validity",
        "canonical_K_max": len(ids), "entity_ids": ids, "candidate_origin_count": len(windows),
        "households_starting_after_training": late_joiners,
        "late_joiner_count": len(late_joiners),
        "retained_origin_count": int(retained.sum()),
        "forecastable_K": distribution(forecastable.sum(axis=1)),
        "forecastable_K_by_split": {
            name: distribution(forecastable.sum(axis=1)[splits == name])
            for name in ("train", "validation", "test")
        },
        "source_label_active_K": _active_report(active[retained], splits[retained]),
        "admitted_source_label_active_K": _active_report(admitted[retained], splits[retained]),
        "households": households,
        **({"potential_training_diagnostics": training_overlap_report(admitted[splits == "train"], ids)}
           if include_training_report else {}),
        "warning": "Source-label potential diagnostics are not fitted PIT diagnostics. Use --cache after inference.",
    }


def cache_diagnostics(
    cache: Path, config: ResolvedExperimentConfig, *, include_test_labels: bool = False,
) -> dict[str, Any]:
    library = load_pit_library(cache, access="evaluation" if include_test_labels else "training")
    data = library.dataset
    ids = [str(value) for value in data["entity_id"].values]
    if ids != config.protocol.ordered_entity_ids:
        raise ValueError("HEAPO diagnostics require the exact configured canonical cache entities")
    splits = data["split"].values.astype(str)
    visible = np.ones(len(splits), dtype=bool) if include_test_labels else splits != "test"
    forecast = data["forecastable_entity"].values.astype(bool)
    active = (
        forecast[:, :, None]
        & np.isfinite(data["pit_u"].values)
        & np.isfinite(data["pit_z"].values)
        & (data["entity_valid"].values == 1)
    )
    valid = active.sum(axis=1) >= config.protocol.min_entities_per_case
    train = splits == "train"
    admitted_train = active[train] & valid[train, None, :]
    return {
        "scope": "evaluation" if include_test_labels else "train_and_validation_test_labels_sealed",
        "canonical_K_max": len(ids),
        "forecastable_K": distribution(forecast.sum(axis=1)),
        "active_K": _active_report(active[visible], splits[visible]),
        "active_fraction_by_household": {
            entity: float(active[visible, index].mean()) for index, entity in enumerate(ids)
        },
        "pit_training_frequency_fallback": training_overlap_report(
            np.isfinite(data["pit_u"].values[train]), ids,
        )["training_frequency_fallback"],
        "m1_training_overlap": training_overlap_report(admitted_train, ids),
        "missingness": library.metadata.get("missingness", {}),
    }


def archive_station_diagnostics(archive_path: Path, config: ResolvedExperimentConfig) -> dict[str, Any]:
    """Audit all control station cohorts in-place, without extracting unselected households."""

    verify_archive(archive_path)
    result = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_members(archive)
        with archive.open(members["meta_data/households.csv"]) as stream:
            households = read_households(stream)
        for station in sorted(households["Weather_ID"].unique()):
            ids = select_households(households, station, maximum=len(households))
            entities = [EntityMetadata(name=str(identifier), group_name="household") for identifier in ids]
            canonical = [entity.entity_id for entity in entities]
            station_config = config.model_copy(update={"protocol": ProtocolConfig(
                cardinality_mode="variable", entity_count=len(ids), ordered_entity_ids=canonical,
                min_entities_per_case=config.protocol.min_entities_per_case,
            )})
            with archive.open(members[f"weather_data/hourly/{station}.csv"]) as stream:
                weather = read_weather(stream, station, config.covariates.weather)
            frames = []
            for identifier, entity in zip(ids, entities, strict=True):
                with archive.open(members[f"smart_meter_data/15min/{identifier}.csv"]) as stream:
                    target = read_target(stream, identifier)
                frames.append(EntityFrames(entity, target, weather, pd.DataFrame(index=weather.index)))
            adapter = FrameDatasetAdapter(
                station_config, EntityGroup(f"station_{station}", canonical, entities=entities), frames, {},
            )
            report = source_diagnostics(adapter, include_training_report=False)
            rows = households.loc[households["Weather_ID"].eq(station)]
            result[station] = {
                "total_households": len(rows), "control_households": int(rows["Group"].eq("control").sum()),
                "control_households_15min": len(ids),
                "lifetime_days": distribution([target_summary(frame.target)["lifetime_days"] for frame in frames]),
                "available_equivalent_days": distribution([
                    target_summary(frame.target)["equivalent_available_days"] for frame in frames
                ]),
                "missing_fraction_within_lifetime": distribution([
                    target_summary(frame.target)["missing_fraction_within_lifetime"] for frame in frames
                ]),
                "forecastable_K": report["forecastable_K"],
                "active_K": report["source_label_active_K"],
            }
    return result


def main(
    base: Annotated[Path, typer.Option("--base", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    cache: Annotated[Path | None, typer.Option("--cache", exists=True)] = None,
    include_test_labels: Annotated[bool, typer.Option("--include-test-labels")] = False,
    all_stations_archive: Annotated[Path | None, typer.Option("--all-stations-archive", exists=True)] = None,
) -> None:
    """Write source-cohort availability and optional access-gated cache diagnostics."""

    config = resolve_run_config(
        load_base_config(base), IndependentMethodConfig(kind="method", id="m0", family="independent"),
    )
    adapter = load_adapter(config)
    if not isinstance(adapter, FrameDatasetAdapter):
        raise TypeError("HEAPO diagnostics require the frame adapter")
    report = {"source": source_diagnostics(adapter)}
    if cache is not None:
        report["cache"] = cache_diagnostics(cache, config, include_test_labels=include_test_labels)
    if all_stations_archive is not None:
        report["stations"] = archive_station_diagnostics(all_stations_archive, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    typer.echo(str(output))


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    cli()
