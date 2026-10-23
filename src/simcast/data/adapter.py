"""Dataset adapter interface and the shared per-window assembly used by every adapter.

An adapter turns one dataset into what the frozen-marginal pipeline needs: an ordered entity group, aligned
target coverage, and, for each forecast window, point-in-time model inputs (past target, past covariates,
legitimately available future covariates) plus the sealed truth. Nothing downstream may branch on the dataset.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from simcast.config import ResolvedExperimentConfig
from simcast.data.availability import select_available_past_targets, select_latest_weather_forecast
from simcast.data.covariates import (
    build_future_covariates,
    build_past_covariates,
    select_weather_features,
    with_calendar,
)
from simcast.data.windows import ForecastWindow
from simcast.types import EntityGroup, EntityMetadata

LOGGER = logging.getLogger(__name__)


class IneligibleWindow(ValueError):
    """A forecast window lacks some required covariate or target information."""


@dataclass(frozen=True, slots=True)
class PreparedWindow:
    """Full-universe truth and compact inputs for forecastable entities at one origin."""

    inputs: list[dict[str, object]]
    true_y: np.ndarray
    history_missing: np.ndarray
    forecastable_entity: np.ndarray
    entity_indices: np.ndarray


class DatasetAdapter(Protocol):
    """What the cache builder requires from a dataset."""

    group: EntityGroup
    """Ordered entities of the one fixed group (ids, names, optional coordinates, static metadata)."""

    provenance: Mapping[str, Any]
    """JSON-serializable source description stored in the cache metadata."""

    def coverage(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last UTC timestamp for which every entity has a target row."""

    def prepare_window(self, window: ForecastWindow) -> PreparedWindow:
        """Point-in-time inputs for ``window``; raise :class:`IneligibleWindow` if they are incomplete."""


@dataclass(frozen=True, slots=True)
class EntityFrames:
    """UTC-indexed tables for one entity.

    ``target`` has the configured target column. ``past_weather`` holds past covariates; ``future_weather``
    holds versioned (``available_at``) covariates that may be known ahead of the origin.
    """

    entity: EntityMetadata
    target: pd.DataFrame
    past_weather: pd.DataFrame
    future_weather: pd.DataFrame


def _finite(frame: pd.DataFrame) -> bool:
    return bool(np.isfinite(frame.to_numpy(dtype=np.float64, copy=False)).all())


def _window_slice(frame: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Limit work before applying the availability selectors."""

    return frame.loc[timestamps[0] : timestamps[-1]]


class FrameDatasetAdapter:
    """Adapter over in-memory per-entity frames.

    ``past_weather_source`` states how ``past_weather`` becomes past covariates: ``"measured"`` rows are
    available at their own timestamp; ``"versioned"`` rows carry ``available_at`` like forecasts do.
    """

    def __init__(
        self,
        config: ResolvedExperimentConfig,
        group: EntityGroup,
        frames: Sequence[EntityFrames],
        provenance: Mapping[str, Any],
        *,
        past_weather_source: Literal["measured", "versioned"] = "measured",
    ) -> None:
        self.config = config
        self.group = group
        self.frames = list(frames)
        self.provenance = provenance
        self.past_weather_source = past_weather_source

    def coverage(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = max(frame.target.index.min() for frame in self.frames)
        end = min(frame.target.index.max() for frame in self.frames)
        total_rows = sum(len(frame.target) for frame in self.frames)
        column = self.config.data.target_column
        missing = sum(int(frame.target[column].isna().sum()) for frame in self.frames)
        LOGGER.info(
            "Aligned target coverage %s to %s; %d rows and %.4f%% missing observations",
            start,
            end,
            total_rows,
            100.0 * missing / total_rows,
        )
        return pd.Timestamp(start), pd.Timestamp(end)

    def _covariate_kwargs(self) -> dict[str, bool]:
        calendar = self.config.covariates.calendar
        return {
            "calendar": calendar.enabled,
            "include_hour": calendar.include_hour,
            "include_day_of_week": calendar.include_day_of_week,
            "include_is_weekend": calendar.include_is_weekend,
        }

    def _past_covariates(self, frames: EntityFrames, window: ForecastWindow) -> pd.DataFrame:
        weather = self.config.covariates.weather
        kwargs = self._covariate_kwargs()
        past = _window_slice(frames.past_weather, window.past_timestamps)
        if self.past_weather_source == "measured":
            return build_past_covariates(
                past, window.past_timestamps, weather, origin_timestamp=window.origin_timestamp, **kwargs
            )
        selected = select_latest_weather_forecast(past, window.origin_timestamp, window.past_timestamps)
        return with_calendar(select_weather_features(selected, weather), **kwargs)

    def _prepare_entity(
        self,
        frames: EntityFrames,
        window: ForecastWindow,
        *,
        allow_unavailable: bool = False,
    ) -> tuple[dict[str, object] | None, np.ndarray, int]:
        config = self.config
        target_window = frames.target.reindex(window.past_timestamps.union(window.future_timestamps))
        history = select_available_past_targets(
            target_window.loc[window.past_timestamps],
            window.origin_timestamp,
            target_column=config.data.target_column,
        ).reindex(window.past_timestamps)
        truth = target_window.loc[window.future_timestamps, config.data.target_column]

        try:
            past_covariates = self._past_covariates(frames, window)
            future_covariates = build_future_covariates(
                _window_slice(frames.future_weather, window.future_timestamps),
                window.origin_timestamp,
                window.future_timestamps,
                config.covariates.weather,
                measurements=frames.past_weather,
                future_weather_source=config.covariates.future_weather_source,
                **self._covariate_kwargs(),
            )
        except ValueError as error:
            if allow_unavailable and "weather columns are missing:" in str(error):
                return None, truth.to_numpy(dtype=np.float32, copy=True), int(history.isna().sum())
            raise
        if not _finite(past_covariates) or not _finite(future_covariates):
            if allow_unavailable:
                return None, truth.to_numpy(dtype=np.float32, copy=True), int(history.isna().sum())
            raise IneligibleWindow(f"covariates unavailable for {frames.entity.entity_id}")
        if list(past_covariates.columns) != list(future_covariates.columns):
            if allow_unavailable:
                return None, truth.to_numpy(dtype=np.float32, copy=True), int(history.isna().sum())
            raise IneligibleWindow(f"past/future covariate columns differ for {frames.entity.entity_id}")

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
        return prepared, truth.to_numpy(dtype=np.float32, copy=True), int(history.isna().sum())

    def prepare_window(self, window: ForecastWindow) -> PreparedWindow:
        inputs: list[dict[str, object]] = []
        truth: list[np.ndarray] = []
        history_missing: list[int] = []
        entity_indices: list[int] = []
        forecastable = np.zeros(len(self.frames), dtype=bool)
        variable = self.config.protocol.cardinality_mode == "variable"
        for entity_index, entity_frames in enumerate(self.frames):
            entity_input, entity_truth, missing_count = self._prepare_entity(
                entity_frames,
                window,
                allow_unavailable=variable,
            )
            if entity_input is not None:
                inputs.append(entity_input)
                entity_indices.append(entity_index)
                forecastable[entity_index] = True
            truth.append(entity_truth)
            history_missing.append(missing_count)
        return PreparedWindow(
            inputs=inputs,
            true_y=np.stack(truth),
            history_missing=np.asarray(history_missing, dtype=np.int64),
            forecastable_entity=forecastable,
            entity_indices=np.asarray(entity_indices, dtype=np.int64),
        )
