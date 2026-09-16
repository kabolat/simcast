"""Execute an explicit venue-scoped collection of scientific experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

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
from simcast.experiments import (
    locate_compatible_cache,
    update_base,
    update_method,
    write_yaml,
)
from simcast.reporting.composite_report import write_composite_report
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

    write_composite_report(report_root, cells, config.analysis)
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
