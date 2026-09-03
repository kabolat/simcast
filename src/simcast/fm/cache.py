"""Labeled, leakage-gated storage for historical FM forecasts and PIT scores."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

CacheAccess = Literal["training", "evaluation"]
LABEL_VARIABLES = ("true_y", "pit_u", "pit_z")


@dataclass
class PITLibrary:
    """An in-memory labeled PIT dictionary.

    Test realizations are absent unless the library was explicitly loaded with
    ``access="evaluation"``. The split coordinate remains visible so callers
    can select train/validation origins without consulting hidden labels.
    """

    dataset: xr.Dataset
    metadata: dict[str, Any]
    access: CacheAccess = "training"

    def dependence_data(self, *, include_validation: bool = False) -> xr.Dataset:
        """Return label-bearing origins permitted for dependence fitting."""

        allowed = ["train", "validation"] if include_validation else ["train"]
        return self.dataset.where(self.dataset["split"].isin(allowed), drop=True)

    def test_data(self) -> xr.Dataset:
        """Return untouched test data only after explicit evaluation access."""

        if self.access != "evaluation":
            raise PermissionError("test labels are sealed; reload the PIT library with access='evaluation'")
        return self.dataset.where(self.dataset["split"] == "test", drop=True)


def build_cache_dataset(
    *,
    origin_timestamps: Sequence[np.datetime64] | np.ndarray,
    entity_ids: Sequence[str],
    true_y: np.ndarray,
    quantile_predictions: np.ndarray,
    pit_u: np.ndarray,
    pit_z: np.ndarray,
    forecast_embeddings: np.ndarray,
    quantile_levels: Sequence[float] | np.ndarray,
    split: Sequence[str] | np.ndarray,
    latitude: Sequence[float] | np.ndarray | None = None,
    longitude: Sequence[float] | np.ndarray | None = None,
) -> xr.Dataset:
    """Build the canonical cache dataset with explicit named dimensions.

    Shapes are ``true_y/pit: [N,K,H]``, predictions ``[N,K,H,Q]``, and
    embeddings ``[N,K,P,D]``. Patch embeddings are stored once per patch.
    """

    truth = np.asarray(true_y)
    predictions = np.asarray(quantile_predictions)
    u = np.asarray(pit_u)
    z = np.asarray(pit_z)
    embeddings = np.asarray(forecast_embeddings)
    levels = np.asarray(quantile_levels, dtype=np.float32)
    timestamps = np.asarray(origin_timestamps, dtype="datetime64[ns]")
    split_array = np.asarray(split, dtype=str)
    if truth.ndim != 3:
        raise ValueError("true_y must have shape [origin, entity, lead]")
    n_origin, n_entity, horizon = truth.shape
    if predictions.shape != (n_origin, n_entity, horizon, levels.size):
        raise ValueError("quantile_predictions shape does not match truth and quantile levels")
    if u.shape != truth.shape or z.shape != truth.shape:
        raise ValueError("pit_u and pit_z must match true_y")
    if embeddings.ndim != 4 or embeddings.shape[:2] != (n_origin, n_entity):
        raise ValueError("forecast_embeddings must have shape [origin, entity, patch, hidden]")
    if timestamps.shape != (n_origin,) or split_array.shape != (n_origin,):
        raise ValueError("one timestamp and split label are required per origin")
    if len(entity_ids) != n_entity or len(set(entity_ids)) != n_entity:
        raise ValueError("entity_ids must be unique and match the entity dimension")
    if not set(split_array).issubset({"train", "validation", "test"}):
        raise ValueError("split labels must be train, validation, or test")
    if levels.ndim != 1 or levels.size == 0 or np.any(np.diff(levels) <= 0):
        raise ValueError("quantile_levels must be a non-empty increasing vector")

    n_patch, hidden_dim = embeddings.shape[2:]
    data_vars: dict[str, Any] = {
        "origin_timestamp": (("origin",), timestamps),
        "entity_id": (("entity",), np.asarray(entity_ids, dtype=str)),
        "split": (("origin",), split_array),
        "true_y": (("origin", "entity", "lead"), truth),
        "quantile_prediction": (("origin", "entity", "lead", "quantile"), predictions),
        "pit_u": (("origin", "entity", "lead"), u),
        "pit_z": (("origin", "entity", "lead"), z),
        "forecast_embedding": (("origin", "entity", "patch", "hidden"), embeddings),
    }
    if latitude is not None:
        lat = np.asarray(latitude, dtype=np.float32)
        if lat.shape != (n_entity,):
            raise ValueError("latitude must have one value per entity")
        data_vars["latitude"] = (("entity",), lat)
    if longitude is not None:
        lon = np.asarray(longitude, dtype=np.float32)
        if lon.shape != (n_entity,):
            raise ValueError("longitude must have one value per entity")
        data_vars["longitude"] = (("entity",), lon)

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "origin": np.arange(n_origin, dtype=np.int64),
            "entity": np.arange(n_entity, dtype=np.int64),
            "lead": np.arange(1, horizon + 1, dtype=np.int64),
            "quantile": levels,
            "patch": np.arange(n_patch, dtype=np.int64),
            "hidden": np.arange(hidden_dim, dtype=np.int64),
        },
        attrs={"schema": "simcast.pit_library", "schema_version": 1},
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def save_pit_library(
    library_dir: str | Path,
    dataset: xr.Dataset,
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Persist a PIT library while physically separating untouched test labels."""

    destination = Path(library_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"PIT library already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        test_mask = np.asarray(dataset["split"].values) == "test"
        public = dataset.copy(deep=True)
        if test_mask.any():
            for name in LABEL_VARIABLES:
                values = np.asarray(public[name].values).copy()
                values[test_mask] = np.nan
                public[name] = (public[name].dims, values)
            labels = dataset[list(LABEL_VARIABLES)].isel(origin=np.flatnonzero(test_mask))
            labels.to_zarr(temp_dir / "test_labels.zarr", mode="w", consolidated=True)
        public.to_zarr(temp_dir / "dataset.zarr", mode="w", consolidated=True)
        payload = dict(metadata)
        payload.update({"schema": "simcast.pit_library", "schema_version": 1, "test_labels_sealed": True})
        (temp_dir / "metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temp_dir, destination)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return destination


def load_pit_library(library_dir: str | Path, *, access: CacheAccess = "training") -> PITLibrary:
    """Load a cache; test truth is merged only for explicit evaluation access."""

    if access not in {"training", "evaluation"}:
        raise ValueError("access must be 'training' or 'evaluation'")
    path = Path(library_dir)
    dataset = xr.open_zarr(path / "dataset.zarr", consolidated=True).load()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if access == "evaluation" and (path / "test_labels.zarr").exists():
        labels = xr.open_zarr(path / "test_labels.zarr", consolidated=True).load()
        positions = np.asarray(labels["origin"].values, dtype=np.int64)
        for name in LABEL_VARIABLES:
            values = np.asarray(dataset[name].values).copy()
            values[positions] = np.asarray(labels[name].values)
            dataset[name] = (dataset[name].dims, values)
    return PITLibrary(dataset=dataset, metadata=metadata, access=access)
