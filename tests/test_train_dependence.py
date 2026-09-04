from pathlib import Path

import numpy as np
import torch

from simcast.cli.reproduce import reproduce_run
from simcast.cli.train_dependence import train_from_config
from simcast.config import SimcastConfig
from simcast.fm.cache import build_cache_dataset, save_pit_library
from simcast.training.checkpoint import load_conditional_checkpoint


def _cache(path: Path, *, invalid_train_prefix: int = 0) -> Path:
    rng = np.random.default_rng(31)
    n, k, h, p, d = 10, 4, 2, 1, 5
    medians = rng.normal(size=(n, k, h)).astype(np.float32)
    predictions = np.stack((medians - 1, medians, medians + 1), axis=-1)
    pit_z = rng.normal(size=(n, k, h)).astype(np.float32)
    pit_z[:invalid_train_prefix] = np.nan
    dataset = build_cache_dataset(
        origin_timestamps=np.arange(n).astype("datetime64[D]"),
        entity_ids=[f"e-{idx}" for idx in range(k)],
        true_y=medians,
        quantile_predictions=predictions,
        pit_u=np.full((n, k, h), 0.5, dtype=np.float32),
        pit_z=pit_z,
        forecast_embeddings=rng.normal(size=(n, k, p, d)).astype(np.float16),
        quantile_levels=[0.1, 0.5, 0.9],
        split=["train"] * 6 + ["validation"] * 2 + ["test"] * 2,
        latitude=np.arange(k),
        longitude=np.arange(k) + 5,
    )
    dataset.attrs["output_patch_size"] = 2
    return save_pit_library(path, dataset, {})


def _config(tmp_path: Path, method: str) -> SimcastConfig:
    return SimcastConfig.model_validate(
        {
            "chronos": {"device": "cpu"},
            "dependence": {
                "method": method,
                "conditional_low_rank": {"hidden_dims": [8], "latent_rank": 2, "dropout": 0.0},
                "set_aware_low_rank": {
                    "model_dim": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "latent_rank": 2,
                    "dropout": 0.0,
                },
                "conditional_kernel": {
                    "hidden_dims": [8],
                    "embedding_dim": 3,
                    "dropout": 0.0,
                    "smoke_only": True,
                    "smoke_max_origins": 4,
                },
            },
            "features": {"use_location": True},
            "subset_training": {"enabled": False},
            "training": {"batch_size": 4, "epochs": 2, "patience": 2},
            "output": {"root_dir": str(tmp_path / "runs")},
        }
    )


def test_train_static_from_sealed_cache(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    run = train_from_config(
        _config(tmp_path, "static_gaussian"),
        cache_dir=cache,
        output_dir=tmp_path / "m1",
    )
    assert (run / "model.npz").is_file()
    assert (run / "resolved_config.yaml").is_file()
    assert (run / "run_metadata.json").is_file()
    reproduced = reproduce_run(run, output_dir=tmp_path / "m1-reproduced")
    assert (reproduced / "model.npz").is_file()


def test_train_and_restore_conditional_model(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    run = train_from_config(
        _config(tmp_path, "conditional_low_rank"),
        cache_dir=cache,
        output_dir=tmp_path / "m2",
    )
    loaded = load_conditional_checkpoint(run / "best.pt")
    assert loaded.method == "conditional_low_rank"
    assert loaded.entity_ids == ("e-0", "e-1", "e-2", "e-3")
    assert torch.isfinite(loaded.model(torch.randn(2, 4, 13))).all()


def test_train_and_restore_set_aware_model(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    run = train_from_config(
        _config(tmp_path, "set_aware_low_rank"),
        cache_dir=cache,
        output_dir=tmp_path / "m3",
    )
    loaded = load_conditional_checkpoint(run / "best.pt")
    assert loaded.method == "set_aware_low_rank"
    assert torch.isfinite(loaded.model(torch.randn(2, 4, 13))).all()


def test_train_and_restore_bounded_kernel_smoke(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache", invalid_train_prefix=4)
    run = train_from_config(
        _config(tmp_path, "conditional_kernel"),
        cache_dir=cache,
        output_dir=tmp_path / "m4",
    )
    loaded = load_conditional_checkpoint(run / "best.pt")
    assert loaded.method == "conditional_kernel"
    assert torch.isfinite(loaded.model(torch.randn(2, 4, 13))).all()


def test_train_and_restore_full_kernel(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    config = _config(tmp_path, "conditional_kernel")
    kernel = config.dependence.conditional_kernel.model_copy(update={"smoke_only": False})
    dependence = config.dependence.model_copy(update={"conditional_kernel": kernel})
    run = train_from_config(
        config.model_copy(update={"dependence": dependence}),
        cache_dir=cache,
        output_dir=tmp_path / "m4_full",
    )

    loaded = load_conditional_checkpoint(run / "best.pt")
    assert loaded.method == "conditional_kernel"
    assert torch.isfinite(loaded.model(torch.randn(2, 4, 13))).all()
