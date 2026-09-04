"""Run the complete frozen-marginal Simcast experiment."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]

from simcast.cli.build_cache import build_cache_from_config
from simcast.cli.evaluate import CORE_METHODS, evaluate_from_config
from simcast.cli.train_dependence import _cache_path, _git_commit, train_from_config
from simcast.config import SimcastConfig, load_config


def _experiment_directory(config: SimcastConfig, override: str | Path | None) -> Path:
    if override is None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
        name = config.output.experiment_name or config.data.entity_type
        path = (Path(config.output.root_dir).expanduser() / f"{stamp}_{name}").resolve()
    else:
        path = Path(override).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _with_method(config: SimcastConfig, method: str) -> SimcastConfig:
    dependence = config.dependence.model_copy(update={"method": method})
    if method != "conditional_kernel":
        return config.model_copy(update={"dependence": dependence})
    training = config.training.model_copy(
        update={
            "epochs": min(config.training.epochs, 5),
            "patience": min(config.training.patience, 2, config.training.epochs),
        }
    )
    subset = config.subset_training.model_copy(update={"enabled": False})
    return config.model_copy(update={"dependence": dependence, "training": training, "subset_training": subset})


def run_experiment_from_config(
    config: SimcastConfig,
    *,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    rebuild_cache: bool = False,
    include_kernel_smoke: bool = False,
) -> Path:
    """Build/reuse the PIT library, train M0--M3, and evaluate once."""

    experiment = _experiment_directory(config, output_dir)
    cache = _cache_path(config, cache_dir)
    if rebuild_cache or not cache.is_dir():
        build_cache_from_config(config, output_dir=cache, overwrite=rebuild_cache)

    methods = list(CORE_METHODS)
    if include_kernel_smoke:
        methods.append("conditional_kernel")
    run_paths: dict[str, Path] = {}
    for method in methods:
        run_paths[method] = train_from_config(
            _with_method(config, method),
            cache_dir=cache,
            output_dir=experiment / method,
        )
    evaluation = evaluate_from_config(
        config,
        methods=methods,
        method_runs=run_paths,
        cache_dir=cache,
        output_dir=experiment / "evaluation",
    )
    group_metadata = json.loads((next(iter(run_paths.values())) / "run_metadata.json").read_text(encoding="utf-8"))[
        "group"
    ]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "cache_path": str(cache),
        "method_runs": {name: str(path) for name, path in run_paths.items()},
        "evaluation_path": str(evaluation),
        "methods": methods,
        "kernel_is_bounded_smoke": include_kernel_smoke,
        "group": group_metadata,
        "experimental_protocol": {
            "name": config.protocol.name,
            "full_group_only": config.protocol.full_group_only,
            "subset_training": config.subset_training.enabled,
            "entity_selection_augmentation_enabled": bool(config.evaluation.variable_k_sizes),
        },
        "git_commit": _git_commit(),
    }
    (experiment / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if config.output.save_resolved_config:
        (experiment / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
    return experiment


def run_experiment(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    rebuild_cache: bool = False,
    include_kernel_smoke: bool = False,
) -> Path:
    return run_experiment_from_config(
        load_config(config_path, overrides=overrides),
        cache_dir=cache_dir,
        output_dir=output_dir,
        rebuild_cache=rebuild_cache,
        include_kernel_smoke=include_kernel_smoke,
    )


def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False, readable=True)],
    override: Annotated[list[str] | None, typer.Option("--set")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir", file_okay=False)] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False)] = None,
    rebuild_cache: Annotated[bool, typer.Option("--rebuild-cache")] = False,
    include_kernel_smoke: Annotated[bool, typer.Option("--include-kernel-smoke")] = False,
) -> None:
    resolved = load_config(config, overrides=override or ())
    logging.basicConfig(
        level=getattr(logging, resolved.runtime.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    typer.echo(
        run_experiment_from_config(
            resolved,
            cache_dir=cache_dir,
            output_dir=output_dir,
            rebuild_cache=rebuild_cache,
            include_kernel_smoke=include_kernel_smoke,
        )
    )


if __name__ == "__main__":
    typer.run(main)
