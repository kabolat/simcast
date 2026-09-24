"""Execute an explicit venue-scoped collection of scientific experiments.

A composite run only fits declared methods. Evaluations, if declared, run
immediately afterward reusing those same fits. Reports are never generated
here; run ``simcast.cli.report_composite`` explicitly against a completed run.
"""

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
    load_evaluation_config,
    load_method_config,
    resolve_run_config,
)
from simcast.experiments import (
    locate_compatible_cache,
    update_base,
    update_method,
    write_yaml,
)
from simcast.reproducibility import canonical_json_hash, environment_metadata, git_commit, utc_run_id

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExpandedFit:
    base: BaseExperimentConfig
    base_entry_id: str
    method_id: str
    method: MethodConfig
    seed: int | None


def _resolve_path(source: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (source.parent / value).resolve()


def _is_optimized(method: MethodConfig) -> bool:
    return isinstance(
        method,
        (ConditionalLowRankMethodConfig, SetAwareLowRankMethodConfig, ConditionalKernelMethodConfig),
    )


def _load_bases(source: Path, config: CompositeExperimentConfig) -> dict[str, BaseExperimentConfig]:
    return {
        entry.id: update_base(load_base_config(_resolve_path(source, entry.config)), entry.overrides)
        for entry in config.bases
    }


def expand_methods(source: Path, config: CompositeExperimentConfig) -> list[ExpandedFit]:
    """Expand every declared base x method (x seed) fit cell.

    Independent of any declared evaluation: fitting never depends on
    sampling or scoring settings.
    """

    bases = _load_bases(source, config)
    expanded: list[ExpandedFit] = []
    for entry in config.methods:
        method = update_method(load_method_config(_resolve_path(source, entry.method)), entry.overrides)
        if entry.seeds and not _is_optimized(method):
            raise ValueError(f"deterministic method {entry.id!r} must not declare repeated seeds")
        if isinstance(
            method,
            (ConditionalLowRankMethodConfig, SetAwareLowRankMethodConfig, ConditionalKernelMethodConfig),
        ):
            seeds: list[int | None] = list(entry.seeds) or [int(method.optimization.seed)]
        else:
            seeds = [None]
        selected_bases = entry.base_ids or list(bases)
        expanded.extend(
            ExpandedFit(bases[base_id], base_id, entry.id, method, seed)
            for base_id in selected_bases
            for seed in seeds
        )
    return expanded


def _seed_label(seed: int | None) -> str:
    return "deterministic" if seed is None else f"seed_{seed}"


def _fit_is_complete(path: Path, family: str) -> bool:
    conditional = {"conditional_low_rank", "set_aware_low_rank", "conditional_kernel"}
    model = "best.pt" if family in conditional else "model.npz"
    return (path / model).is_file() and (path / "run_metadata.json").is_file()


def _reject_partial(path: Path, *, expected_file: str) -> None:
    if path.exists() and not (path / expected_file).is_file():
        raise RuntimeError(f"refusing to overwrite partial output: {path}")


def _configure_log(path: Path) -> None:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)


def _run_fits(
    run_root: Path,
    expanded: list[ExpandedFit],
    *,
    rebuild_cache: bool,
    fits_by_id: dict[str, dict[str, object]],
) -> None:
    rebuilt_caches: set[Path] = set()
    for item in expanded:
        fit_id = f"{item.base_entry_id}/{item.method_id}/{_seed_label(item.seed)}"
        if fit_id in fits_by_id:
            continue
        cache = locate_compatible_cache(item.base)
        runtime = resolve_run_config(item.base, item.method, seed=item.seed)
        if rebuild_cache and cache not in rebuilt_caches:
            build_cache_from_config(runtime, output_dir=cache, overwrite=True)
            rebuilt_caches.add(cache)
        elif not cache.is_dir():
            build_cache_from_config(runtime, output_dir=cache, overwrite=False)
        fit_dir = run_root / item.base_entry_id / item.method_id / _seed_label(item.seed)
        if fit_dir.exists() and not _fit_is_complete(fit_dir, item.method.family):
            raise RuntimeError(f"refusing to overwrite partial dependence fit: {fit_dir}")
        if not _fit_is_complete(fit_dir, item.method.family):
            train_from_config(runtime, cache_dir=cache, output_dir=fit_dir)
        fits_by_id[fit_id] = {
            "fit_id": fit_id,
            "base_id": item.base_entry_id,
            "method_id": item.method_id,
            "seed": item.seed,
            "method_family": item.method.family,
            "base_sha256": canonical_json_hash(item.base.model_dump(mode="json")),
            "marginal_sha256": base_fingerprint(item.base),
            "method_sha256": canonical_json_hash(item.method.model_dump(mode="json")),
            "cache_path": str(cache),
            "fit_path": str(fit_dir),
            "status": "complete",
        }
        LOGGER.info("fitted %s", fit_id)


def _run_evaluations(
    run_root: Path,
    source: Path,
    config: CompositeExperimentConfig,
    fits_by_id: dict[str, dict[str, object]],
    evaluations_by_id: dict[str, list[dict[str, object]]],
) -> None:
    fits_by_base: dict[str, list[dict[str, object]]] = {}
    for fit in fits_by_id.values():
        fits_by_base.setdefault(str(fit["base_id"]), []).append(fit)
    base_entries = {entry.id: entry for entry in config.bases}
    method_entries = {entry.id: entry for entry in config.methods}

    for entry in config.evaluations:
        document = load_evaluation_config(_resolve_path(source, entry.config))
        if document.reference not in method_entries:
            raise ValueError(f"evaluation {entry.id!r} references unknown method {document.reference!r}")
        cells = evaluations_by_id.setdefault(entry.id, [])
        completed = {(cell["base_id"], cell["method_id"], cell["seed"]) for cell in cells}
        base_ids = entry.base_ids or list(fits_by_base)
        for base_id in base_ids:
            base_fits = fits_by_base.get(base_id, [])
            reference_fits = [fit for fit in base_fits if fit["method_id"] == document.reference]
            if len(reference_fits) != 1 or reference_fits[0]["seed"] is not None:
                raise ValueError(
                    f"evaluation {entry.id!r} reference {document.reference!r} must be a single "
                    f"deterministic fit for base {base_id!r}"
                )
            reference_fit = reference_fits[0]
            selected = [
                fit
                for fit in base_fits
                if fit["method_id"] != document.reference
                and (not entry.method_ids or fit["method_id"] in entry.method_ids)
            ]
            for fit in selected:
                key = (fit["base_id"], fit["method_id"], fit["seed"])
                if key in completed:
                    continue
                fit_seed = fit["seed"] if isinstance(fit["seed"], int) else None
                evaluation_dir = (
                    run_root / "evaluations" / entry.id / base_id / str(fit["method_id"]) / _seed_label(fit_seed)
                )
                _reject_partial(evaluation_dir, expected_file="evaluation_manifest.json")
                methods = [str(fit["method_family"])]
                method_runs: dict[str, Path] = {str(fit["method_family"]): Path(str(fit["fit_path"]))}
                if reference_fit["method_family"] != fit["method_family"]:
                    methods.append(str(reference_fit["method_family"]))
                    method_runs[str(reference_fit["method_family"])] = Path(str(reference_fit["fit_path"]))
                if not (evaluation_dir / "evaluation_manifest.json").is_file():
                    base_entry = base_entries[base_id]
                    base = update_base(
                        load_base_config(_resolve_path(source, base_entry.config)), base_entry.overrides
                    )
                    method_entry = method_entries[str(fit["method_id"])]
                    method = update_method(
                        load_method_config(_resolve_path(source, method_entry.method)), method_entry.overrides
                    )
                    seed = fit_seed
                    runtime = resolve_run_config(base, method, seed=seed)
                    runtime = runtime.model_copy(
                        update={"sampling": document.sampling, "evaluation": document.evaluation}
                    )
                    evaluate_from_config(
                        runtime,
                        methods=methods,
                        method_runs=method_runs,
                        cache_dir=Path(str(fit["cache_path"])),
                        output_dir=evaluation_dir,
                    )
                cells.append(
                    {
                        "base_id": fit["base_id"],
                        "method_id": fit["method_id"],
                        "seed": fit["seed"],
                        "method_family": fit["method_family"],
                        "reference_method_id": document.reference,
                        "reference_family": reference_fit["method_family"],
                        "evaluation_path": str(evaluation_dir),
                        "status": "complete",
                    }
                )
                completed.add(key)
                LOGGER.info("evaluated %s under %s", key, entry.id)


def run_composite(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    resume: bool = False,
    rebuild_cache: bool = False,
) -> Path:
    source = Path(config_path).expanduser().resolve()
    config = load_composite_config(source)
    expanded = expand_methods(source, config)

    identifier = run_id or utc_run_id()
    if not identifier or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in identifier):
        raise ValueError("run_id must be a lowercase safe slug")
    run_root = (Path("runs") / config.venue / config.name / identifier).resolve()
    configuration_hash = canonical_json_hash({"composite": config.model_dump(mode="json")})
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
            "schema": "simcast.composite.v3",
            "venue": config.venue,
            "name": config.name,
            "run_id": identifier,
            "configuration_sha256": configuration_hash,
            "git_commit": git_commit(),
            "status": "running",
            "fits": [],
            "evaluations": {},
        }
        write_yaml(run_root / "resolved_composite.yaml", config)
        environment = environment_metadata()
        (run_root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _configure_log(run_root / "composite.log")

    fits_by_id = {str(fit["fit_id"]): fit for fit in manifest.get("fits", []) if fit.get("status") == "complete"}
    invalid = [
        fit_id
        for fit_id, fit in fits_by_id.items()
        if not _fit_is_complete(Path(str(fit["fit_path"])), str(fit["method_family"]))
    ]
    if invalid:
        raise RuntimeError(f"recorded completed fits have missing or partial outputs: {invalid}")

    _run_fits(run_root, expanded, rebuild_cache=rebuild_cache, fits_by_id=fits_by_id)
    manifest["fits"] = list(fits_by_id.values())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    evaluations_by_id: dict[str, list[dict[str, object]]] = dict(manifest.get("evaluations", {}))
    if config.evaluations:
        _run_evaluations(run_root, source, config, fits_by_id, evaluations_by_id)
        manifest["evaluations"] = evaluations_by_id
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    manifest["status"] = "complete"
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


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)

