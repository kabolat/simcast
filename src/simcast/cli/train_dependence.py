"""Fit a configured dependence model from the sealed training PIT library."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
import yaml  # type: ignore[import-untyped]

from simcast.config import SimcastConfig, load_config
from simcast.dependence import IndependentCopula, StaticGaussianCopula
from simcast.dependence.conditional_kernel import ConditionalKernelGaussianCopula
from simcast.dependence.conditional_low_rank import ConditionalLowRankGaussianCopula
from simcast.dependence.set_aware_low_rank import SetAwareLowRankGaussianCopula
from simcast.evaluation.plots import plot_training_history
from simcast.fm.cache import PITLibrary, load_pit_library
from simcast.fm.feature_builder import FeatureBuilder
from simcast.training import ConditionalTrainer, DependenceCollator, DependenceDataset

LOGGER = logging.getLogger(__name__)


def _cache_path(config: SimcastConfig, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    name = config.output.cache_name or f"liander2024_{config.data.entity_type}"
    return (Path(config.output.cache_dir).expanduser() / name).resolve()


def _run_directory(config: SimcastConfig, method: str, override: str | Path | None) -> Path:
    if override is not None:
        path = Path(override).expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
        path = (Path(config.output.root_dir).expanduser() / f"{stamp}_{method}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_run_metadata(run_dir: Path, config: SimcastConfig, cache_path: Path, entity_ids: list[str]) -> None:
    resolved = config.model_dump(mode="json")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    packages = {
        name: version(name) for name in ("numpy", "pandas", "torch", "scikit-learn", "xarray", "zarr", "transformers")
    }
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "packages": packages,
        "cache_path": str(cache_path),
        "entity_ids": entity_ids,
        "seed": config.seed,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _split_indices(library: PITLibrary, label: str) -> np.ndarray:
    return np.flatnonzero(np.asarray(library.dataset["split"].values) == label)


def _complete_origin_indices(library: PITLibrary, indices: np.ndarray, limit: int) -> np.ndarray:
    scores = np.asarray(library.dataset["pit_z"].isel(origin=indices).values)
    has_complete_vector = np.isfinite(scores).all(axis=1).any(axis=1)
    return indices[has_complete_vector][:limit]


def _locations(library: PITLibrary) -> torch.Tensor | None:
    dataset = library.dataset
    if "latitude" not in dataset or "longitude" not in dataset:
        return None
    return torch.tensor(np.stack((dataset["latitude"].values, dataset["longitude"].values), axis=-1))


def _feature_builder(config: SimcastConfig, library: PITLibrary) -> FeatureBuilder:
    patch_size = int(library.dataset.attrs["output_patch_size"])
    settings = config.features
    if settings.use_entity_id_embedding:
        raise NotImplementedError(
            "entity-ID embeddings are an ablation and are intentionally disabled in this proof of concept"
        )
    return FeatureBuilder(
        patch_size,
        shape_eps=settings.shape_eps,
        standardize_scalar_features=settings.standardize_scalar_features,
        use_forecast_embedding=settings.use_forecast_embedding,
        use_quantile_shape=settings.use_quantile_shape,
        use_median=settings.use_median,
        use_log_spread=settings.use_log_spread,
        use_within_patch_position=settings.use_within_patch_position,
        use_location=settings.use_location,
    )


def _arrays(library: PITLibrary, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = library.dataset.isel(origin=indices)
    return (
        torch.tensor(dataset["forecast_embedding"].values, dtype=torch.float32),
        torch.tensor(dataset["quantile_prediction"].values, dtype=torch.float32),
        torch.tensor(dataset["pit_z"].values, dtype=torch.float32),
    )


def _train_conditional(
    config: SimcastConfig,
    library: PITLibrary,
    run_dir: Path,
    entity_ids: list[str],
    method: str,
) -> None:
    train_indices = _split_indices(library, "train")
    validation_indices = _split_indices(library, "validation")
    if method == "conditional_kernel":
        smoke = config.dependence.conditional_kernel
        if not smoke.smoke_only:
            raise ValueError("M4 is intentionally bounded; dependence.conditional_kernel.smoke_only must remain true")
        train_indices = _complete_origin_indices(library, train_indices, smoke.smoke_max_origins)
        validation_indices = _complete_origin_indices(library, validation_indices, smoke.smoke_max_origins)
    if not train_indices.size or not validation_indices.size:
        raise ValueError("conditional training requires non-empty chronological train and validation partitions")
    train_embeddings, train_predictions, train_z = _arrays(library, train_indices)
    validation_embeddings, validation_predictions, validation_z = _arrays(library, validation_indices)
    levels = torch.tensor(library.dataset["quantile"].values, dtype=torch.float32)
    locations = _locations(library)
    builder = _feature_builder(config, library)
    train_features = builder.fit_transform(train_embeddings, train_predictions, levels, locations)
    validation_features = builder.transform(validation_embeddings, validation_predictions, levels, locations)
    if method == "conditional_low_rank":
        m2_config = config.dependence.conditional_low_rank
        conditional_model = ConditionalLowRankGaussianCopula(
            input_dim=train_features.shape[-1],
            latent_rank=m2_config.latent_rank,
            hidden_dims=m2_config.hidden_dims,
            dropout=m2_config.dropout,
            layer_norm=config.features.layer_normalize_embedding,
            sigma_floor=m2_config.sigma_floor,
            jitter=m2_config.jitter,
        )
        model: torch.nn.Module = conditional_model
        model_kwargs = conditional_model.model_kwargs
        model_jitter = m2_config.jitter
    elif method == "set_aware_low_rank":
        m3_config = config.dependence.set_aware_low_rank
        set_model = SetAwareLowRankGaussianCopula(
            input_dim=train_features.shape[-1],
            model_dim=m3_config.model_dim,
            num_layers=m3_config.num_layers,
            num_heads=m3_config.num_heads,
            latent_rank=m3_config.latent_rank,
            dropout=m3_config.dropout,
            sigma_floor=m3_config.sigma_floor,
            jitter=m3_config.jitter,
        )
        model = set_model
        model_kwargs = set_model.model_kwargs
        model_jitter = m3_config.jitter
    elif method == "conditional_kernel":
        kernel_config = config.dependence.conditional_kernel
        kernel_model = ConditionalKernelGaussianCopula(
            input_dim=train_features.shape[-1],
            hidden_dims=kernel_config.hidden_dims,
            embedding_dim=kernel_config.embedding_dim,
            dropout=kernel_config.dropout,
            initial_length_scale=kernel_config.initial_length_scale,
            nugget=kernel_config.nugget,
            jitter=kernel_config.jitter,
        )
        model = kernel_model
        model_kwargs = kernel_model.model_kwargs
        model_jitter = kernel_config.jitter
    else:
        raise ValueError(f"unsupported conditional method: {method}")
    device = config.chronos.device if torch.cuda.is_available() else "cpu"
    trainer = ConditionalTrainer(
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        gradient_clip_norm=config.training.gradient_clip_norm,
        patience=config.training.patience,
        jitter=model_jitter,
        seed=config.seed,
        device=device,
    )
    subset = config.subset_training
    collator = DependenceCollator(
        subset_enabled=subset.enabled,
        min_entities=subset.min_entities,
        full_group_probability=subset.full_group_probability,
        generator=torch.Generator().manual_seed(config.seed),
    )

    def checkpoint_payload() -> dict[str, object]:
        return {
            "schema_version": 1,
            "method": method,
            "model_kwargs": model_kwargs,
            "feature_builder": builder.state_dict(),
            "entity_ids": entity_ids,
        }

    result = trainer.fit(
        model,
        DependenceDataset(train_features, train_z),
        DependenceDataset(validation_features, validation_z),
        output_dir=run_dir,
        training_collator=collator,
        checkpoint_payload=checkpoint_payload,
    )
    plot_training_history(
        [item.epoch for item in result.history],
        [item.training_nll for item in result.history],
        [item.validation_nll for item in result.history],
        run_dir / "training_curve.png",
    )
    LOGGER.info(
        "%s best validation pseudo-NLL %.6f at epoch %d",
        method,
        result.best_validation_nll,
        result.best_epoch,
    )


def train_from_config(
    config: SimcastConfig,
    *,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Fit M0 through M3 without opening sealed test labels."""

    _seed_everything(config.seed, config.runtime.deterministic)
    cache_path = _cache_path(config, cache_dir)
    library = load_pit_library(cache_path, access="training")
    entity_ids = [str(value) for value in library.dataset["entity_id"].values]
    method = config.dependence.method
    run_dir = _run_directory(config, method, output_dir)
    _write_run_metadata(run_dir, config, cache_path, entity_ids)
    train_indices = _split_indices(library, "train")
    train_z = np.asarray(library.dataset["pit_z"].isel(origin=train_indices).values)
    if method == "independent":
        IndependentCopula().fit(train_z, entity_ids).save(run_dir / "model.npz")
    elif method == "static_gaussian":
        settings = config.dependence.static_gaussian
        model = StaticGaussianCopula(
            shrinkage=settings.shrinkage,
            share_across_leads=settings.share_across_leads,
            jitter=settings.jitter,
        ).fit(train_z, entity_ids)
        model.save(run_dir / "model.npz")
    elif method in {"conditional_low_rank", "set_aware_low_rank", "conditional_kernel"}:
        _train_conditional(config, library, run_dir, entity_ids, method)
    else:
        raise ValueError(f"unknown dependence method: {method}")
    return run_dir


def train_dependence(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    return train_from_config(load_config(config_path, overrides=overrides), cache_dir=cache_dir, output_dir=output_dir)


def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False, readable=True)],
    override: Annotated[
        list[str] | None,
        typer.Option("--set", help="Configuration override dotted.path=value"),
    ] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir", file_okay=False)] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False)] = None,
) -> None:
    resolved = load_config(config, overrides=override or ())
    logging.basicConfig(level=getattr(logging, resolved.runtime.log_level))
    typer.echo(train_from_config(resolved, cache_dir=cache_dir, output_dir=output_dir))


if __name__ == "__main__":
    typer.run(main)
