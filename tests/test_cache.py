from pathlib import Path

import numpy as np
import pytest

from simcast.fm.cache import build_cache_dataset, load_pit_library, save_pit_library


def _dataset():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(4, 3, 2)).astype(np.float32)
    return build_cache_dataset(
        origin_timestamps=np.arange(4).astype("datetime64[D]"),
        entity_ids=["a", "b", "c"],
        true_y=truth,
        quantile_predictions=rng.normal(size=(4, 3, 2, 3)).astype(np.float32),
        pit_u=np.full((4, 3, 2), 0.5, dtype=np.float32),
        pit_z=np.zeros((4, 3, 2), dtype=np.float32),
        forecast_embeddings=rng.normal(size=(4, 3, 1, 5)).astype(np.float16),
        quantile_levels=[0.1, 0.5, 0.9],
        split=["train", "train", "validation", "test"],
        latitude=[1, 2, 3],
        longitude=[4, 5, 6],
    )


def test_cache_dimensions_and_patch_storage() -> None:
    dataset = _dataset()
    assert dataset["quantile_prediction"].dims == ("origin", "entity", "lead", "quantile")
    assert dataset["forecast_embedding"].dims == ("origin", "entity", "patch", "hidden")
    assert dataset.sizes == {"origin": 4, "entity": 3, "lead": 2, "quantile": 3, "patch": 1, "hidden": 5}


def test_test_labels_are_physically_sealed_and_api_gated(tmp_path: Path) -> None:
    dataset = _dataset()
    expected = dataset["true_y"].isel(origin=3).values.copy()
    path = save_pit_library(tmp_path / "pit", dataset, {"chronos_commit": "abc"})
    training = load_pit_library(path)
    assert np.isnan(training.dataset["true_y"].isel(origin=3)).all()
    with pytest.raises(PermissionError):
        training.test_data()
    assert training.dependence_data().sizes["origin"] == 2

    evaluation = load_pit_library(path, access="evaluation")
    np.testing.assert_allclose(evaluation.test_data()["true_y"].values[0], expected)
    assert evaluation.metadata["chronos_commit"] == "abc"


def test_cache_refuses_accidental_overwrite(tmp_path: Path) -> None:
    path = save_pit_library(tmp_path / "pit", _dataset(), {})
    with pytest.raises(FileExistsError):
        save_pit_library(path, _dataset(), {})
