"""Execute an explicit venue-scoped collection of scientific experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import matplotlib
import pandas as pd
import typer

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from simcast.cli.build_cache import build_cache_from_config
from simcast.cli.evaluate import evaluate_from_config
from simcast.cli.train_dependence import train_from_config
from simcast.config import (
    BaseExperimentConfig,
    CompositeExperimentConfig,
    ConditionalKernelMethodConfig,
    ConditionalLowRankMethodConfig,
    MethodConfig,
    SetAwareLowRankMethodConfig,
    base_fingerprint,
    load_base_config,
    load_composite_config,
    load_method_config,
    resolve_run_config,
)
from simcast.evaluation.uncertainty import paired_moving_block_bootstrap
from simcast.experiments import (
    locate_compatible_cache,
    update_base,
    update_method,
    write_yaml,
)
from simcast.reproducibility import canonical_json_hash, environment_metadata, git_commit, utc_run_id

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExpandedExperiment:
    base: BaseExperimentConfig
    base_entry_id: str
    experiment_id: str
    method: MethodConfig
    seed: int | None


def _resolve_path(source: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (source.parent / value).resolve()


def _is_optimized(method: MethodConfig) -> bool:
    return isinstance(
        method,
        (ConditionalLowRankMethodConfig, SetAwareLowRankMethodConfig, ConditionalKernelMethodConfig),
    )


def expand_composite(source: Path, config: CompositeExperimentConfig) -> list[ExpandedExperiment]:
    bases = {
        entry.id: update_base(load_base_config(_resolve_path(source, entry.config)), entry.overrides)
        for entry in config.bases
    }
    expanded: list[ExpandedExperiment] = []
    for entry in config.experiments:
        method = update_method(load_method_config(_resolve_path(source, entry.method)), entry.overrides)
        if entry.seeds and not _is_optimized(method):
            raise ValueError(f"deterministic experiment {entry.id!r} must not declare repeated seeds")
        if isinstance(
            method,
            (ConditionalLowRankMethodConfig, SetAwareLowRankMethodConfig, ConditionalKernelMethodConfig),
        ):
            seeds: list[int | None] = list(entry.seeds) or [int(method.optimization.seed)]
        else:
            seeds = [None]
        selected_bases = entry.base_ids or list(bases)
        expanded.extend(
            ExpandedExperiment(bases[base_id], base_id, entry.id, method, seed)
            for base_id in selected_bases
            for seed in seeds
        )
    return expanded


def _fit_is_complete(path: Path, family: str) -> bool:
    conditional = {"conditional_low_rank", "set_aware_low_rank", "conditional_kernel"}
    model = "best.pt" if family in conditional else "model.npz"
    return (path / model).is_file() and (path / "run_metadata.json").is_file()


def _cell_is_complete(cell: dict[str, object]) -> bool:
    return (
        cell.get("status") == "complete"
        and _fit_is_complete(Path(str(cell["fit_path"])), str(cell["method_family"]))
        and (Path(str(cell["evaluation_path"])) / "evaluation_manifest.json").is_file()
    )


def _reject_partial(path: Path, *, expected_file: str) -> None:
    if path.exists() and not (path / expected_file).is_file():
        raise RuntimeError(f"refusing to overwrite partial output: {path}")


def _configure_log(path: Path) -> None:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)


def _write_report(
    report_dir: Path,
    cells: list[dict[str, object]],
    config: CompositeExperimentConfig,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    origin_frames: list[pd.DataFrame] = []
    effect_rows: list[dict[str, object]] = []
    comparisons: set[tuple[str, str, str, str]] = set()
    for cell in cells:
        evaluation = Path(str(cell["evaluation_path"]))
        frame = pd.read_parquet(evaluation / "per_origin_metrics.parquet")
        frame["base_id"] = str(cell["base_id"])
        frame["experiment_id"] = str(cell["experiment_id"])
        frame["configured_seed"] = "deterministic" if cell["seed"] is None else str(cell["seed"])
        origin_frames.append(frame)
        primary_family = str(cell["method_family"])
        reference_family = str(cell["reference_family"])
        if primary_family == reference_family:
            continue
        comparisons.add(
            (str(cell["base_id"]), str(cell["experiment_id"]), primary_family, reference_family)
        )
    per_origin = pd.concat(origin_frames, ignore_index=True)
    per_origin.to_parquet(report_dir / "per_origin_metrics.parquet", index=False)
    for base_id, experiment_id, primary_family, reference_family in sorted(comparisons):
        comparison = per_origin[
            (per_origin["base_id"] == base_id) & (per_origin["experiment_id"] == experiment_id)
        ]
        # Averaging here treats optimization seeds as repeated fits, not as
        # additional test observations. Repeated deterministic reference rows
        # collapse to their common value in the same operation.
        averaged = comparison.groupby(["method", "origin"], as_index=False)["mean_pinball"].mean()
        wide = averaged.pivot(index="origin", columns="method", values="mean_pinball").dropna()
        if primary_family not in wide or reference_family not in wide:
            continue
        for block_length in [config.analysis.primary_block_length, *config.analysis.sensitivity_block_lengths]:
            if len(wide) < block_length:
                continue
            effect = paired_moving_block_bootstrap(
                wide[primary_family].to_numpy(),
                wide[reference_family].to_numpy(),
                block_length=block_length,
                bootstrap_replicates=config.analysis.bootstrap_replicates,
                seed=2027,
            )
            effect_rows.append(
                {
                    "base_id": base_id,
                    "experiment_id": experiment_id,
                    "method": primary_family,
                    "reference": reference_family,
                    **asdict(effect),
                }
            )
    summary = per_origin.groupby(
        ["base_id", "experiment_id", "method", "configured_seed"],
        dropna=False,
        as_index=False,
    ).agg(
        primary_metric=("mean_pinball", "mean")
    )
    summary.to_csv(report_dir / "method_summary.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(report_dir / "paired_effects.csv", index=False)
    figure_data = summary.groupby(["experiment_id", "method"], as_index=False)["primary_metric"].mean()
    figure, axis = plt.subplots(figsize=(max(7.0, 0.7 * len(figure_data)), 4.5))
    labels = [f"{row.experiment_id}\n{row.method}" for row in figure_data.itertuples()]
    axis.bar(labels, figure_data["primary_metric"])
    axis.set_ylabel("mean aggregate pinball loss")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(report_dir / "method_comparison.png", dpi=180)
    figure.savefig(report_dir / "method_comparison.pdf")
    plt.close(figure)


def run_composite(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    resume: bool = False,
    rebuild_cache: bool = False,
) -> Path:
    source = Path(config_path).expanduser().resolve()
    config = load_composite_config(source)
    expanded = expand_composite(source, config)
    reference_entry = next(entry for entry in config.experiments if entry.id == config.analysis.reference)
    reference = update_method(
        load_method_config(_resolve_path(source, reference_entry.method)), reference_entry.overrides
    )
    if _is_optimized(reference) or reference_entry.seeds:
        raise ValueError("the composite reference must be a deterministic M0 or M1 experiment")

    identifier = run_id or utc_run_id()
    if not identifier or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in identifier):
        raise ValueError("run_id must be a lowercase safe slug")
    run_root = (Path("runs") / config.venue / config.name / identifier).resolve()
    report_root = (Path("reports") / config.venue / config.name / identifier).resolve()
    dependency_payload = [
        {
            "base": item.base.model_dump(mode="json"),
            "base_sha256": canonical_json_hash(item.base.model_dump(mode="json")),
            "marginal_sha256": base_fingerprint(item.base),
            "base_entry_id": item.base_entry_id,
            "experiment_id": item.experiment_id,
            "method": item.method.model_dump(mode="json"),
            "method_sha256": canonical_json_hash(item.method.model_dump(mode="json")),
            "seed": item.seed,
        }
        for item in expanded
    ]
    configuration_hash = canonical_json_hash(
        {"composite": config.model_dump(mode="json"), "expanded": dependency_payload}
    )
    manifest_path = run_root / "composite_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"cannot resume without {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration_sha256") != configuration_hash:
            raise ValueError("refusing to resume: resolved composite configuration has changed")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema": "simcast.composite.v2",
            "venue": config.venue,
            "name": config.name,
            "run_id": identifier,
            "configuration_sha256": configuration_hash,
            "git_commit": git_commit(),
            "status": "running",
            "cells": [],
        }
        write_yaml(run_root / "resolved_composite.yaml", config)
        (run_root / "expansion_manifest.json").write_text(
            json.dumps(dependency_payload, indent=2) + "\n", encoding="utf-8"
        )
        environment = environment_metadata()
        (run_root / "environment.json").write_text(
            json.dumps(environment, indent=2) + "\n", encoding="utf-8"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _configure_log(run_root / "composite.log")
    recorded_complete = {
        str(cell["cell_id"]): cell for cell in manifest["cells"] if cell.get("status") == "complete"
    }
    invalid = [cell_id for cell_id, cell in recorded_complete.items() if not _cell_is_complete(cell)]
    if invalid:
        raise RuntimeError(f"recorded completed cells have missing or partial outputs: {invalid}")
    completed = recorded_complete
    cells: list[dict[str, object]] = list(completed.values())
    shared_fits: dict[tuple[str, str], Path] = {}
    rebuilt_caches: set[Path] = set()

    for item in expanded:
        seed_label = "deterministic" if item.seed is None else f"seed_{item.seed}"
        cell_id = f"{item.base_entry_id}/{item.experiment_id}/{seed_label}"
        if cell_id in completed:
            continue
        cache = locate_compatible_cache(item.base)
        runtime = resolve_run_config(item.base, item.method, seed=item.seed)
        if rebuild_cache and cache not in rebuilt_caches:
            build_cache_from_config(runtime, output_dir=cache, overwrite=True)
            rebuilt_caches.add(cache)
        elif not cache.is_dir():
            build_cache_from_config(runtime, output_dir=cache, overwrite=False)
        method_hash = canonical_json_hash(item.method.model_dump(mode="json"))[:12]
        if _is_optimized(item.method):
            fit_dir = run_root / item.base_entry_id / item.experiment_id / seed_label / "fit"
        else:
            fit_dir = run_root / item.base_entry_id / "shared" / f"{item.method.id}-{method_hash}"
        fit_key = (item.base_entry_id, f"{method_hash}:{item.seed}")
        if fit_key not in shared_fits:
            if fit_dir.exists() and not _fit_is_complete(fit_dir, item.method.family):
                raise RuntimeError(f"refusing to overwrite partial dependence fit: {fit_dir}")
            if not _fit_is_complete(fit_dir, item.method.family):
                train_from_config(runtime, cache_dir=cache, output_dir=fit_dir)
            shared_fits[fit_key] = fit_dir

        reference_runtime = resolve_run_config(item.base, reference)
        reference_hash = canonical_json_hash(reference.model_dump(mode="json"))[:12]
        reference_dir = run_root / item.base_entry_id / "shared" / f"{reference.id}-{reference_hash}"
        reference_key = (item.base_entry_id, reference_hash)
        if reference_key not in shared_fits:
            if reference_dir.exists() and not _fit_is_complete(reference_dir, reference.family):
                raise RuntimeError(f"refusing to overwrite partial dependence fit: {reference_dir}")
            if not _fit_is_complete(reference_dir, reference.family):
                train_from_config(reference_runtime, cache_dir=cache, output_dir=reference_dir)
            shared_fits[reference_key] = reference_dir

        cell_dir = run_root / item.base_entry_id / item.experiment_id / seed_label
        evaluation_dir = cell_dir / "evaluation"
        cell_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(cell_dir / "resolved_base.yaml", item.base)
        write_yaml(cell_dir / "resolved_method.yaml", item.method)
        methods = [item.method.family]
        method_runs: dict[str, Path] = {item.method.family: fit_dir}
        if reference.family != item.method.family:
            methods.append(reference.family)
            method_runs[reference.family] = reference_dir
        _reject_partial(evaluation_dir, expected_file="evaluation_manifest.json")
        if not (evaluation_dir / "evaluation_manifest.json").is_file():
            evaluate_from_config(
                runtime,
                methods=methods,
                method_runs=method_runs,
                cache_dir=cache,
                output_dir=evaluation_dir,
            )
        cell: dict[str, object] = {
            "cell_id": cell_id,
            "base_id": item.base_entry_id,
            "scientific_base_id": item.base.id,
            "experiment_id": item.experiment_id,
            "method_id": item.method.id,
            "method_family": item.method.family,
            "base_sha256": canonical_json_hash(item.base.model_dump(mode="json")),
            "marginal_sha256": base_fingerprint(item.base),
            "method_sha256": canonical_json_hash(item.method.model_dump(mode="json")),
            "reference_family": reference.family,
            "seed": item.seed,
            "cache_path": str(cache),
            "fit_path": str(fit_dir),
            "evaluation_path": str(evaluation_dir),
            "status": "complete",
        }
        cells.append(cell)
        manifest["cells"] = cells
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("completed %s", cell_id)

    _write_report(report_root, cells, config)
    manifest["status"] = "complete"
    manifest["report_path"] = str(report_root)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return run_root


def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False, readable=True)],
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    rebuild_cache: Annotated[bool, typer.Option("--rebuild-cache")] = False,
) -> None:
    logging.basicConfig(level=logging.INFO)
    typer.echo(run_composite(config, run_id=run_id, resume=resume, rebuild_cache=rebuild_cache))


if __name__ == "__main__":
    typer.run(main)
