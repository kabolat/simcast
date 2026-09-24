"""Add a new evaluation to an existing composite run without retraining.

This never calls Chronos or a dependence fitter: it only reads fitted method
directories already recorded in an existing composite_manifest.json and runs
the evaluator with a (possibly different) evaluation design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]

from simcast.cli.evaluate import evaluate_from_config
from simcast.config import (
    EvaluationDocumentConfig,
    ResolvedExperimentConfig,
    load_composite_config,
    load_evaluation_config,
)


def _resolve_path(source: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (source.parent / value).resolve()


def _resolve_composite_run(*, venue: str, name: str, run_id: str | None = None) -> Path:
    run_dir = Path("runs") / venue / name
    if run_id is not None:
        if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in run_id):
            raise ValueError("run_id must be a lowercase safe slug")
        selected = run_dir / run_id
        if not (selected / "composite_manifest.json").is_file():
            raise FileNotFoundError(f"no completed composite run found at {selected}")
        return selected.resolve()
    candidates = sorted(
        path for path in run_dir.iterdir() if path.is_dir() and (path / "composite_manifest.json").is_file()
    ) if run_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no completed composite run found under {run_dir}")
    if len(candidates) > 1:
        raise ValueError(f"multiple composite runs found under {run_dir}; use a single run directory")
    return candidates[0].resolve()


def evaluate_composite(
    config_path: str | Path | None = None,
    *,
    composite_config_path: str | Path | None = None,
    evaluation_id: str | None = None,
    run_root: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    if (config_path is None) == (composite_config_path is None):
        raise ValueError("provide exactly one of --config or --composite-config")
    if composite_config_path is not None:
        if run_root is not None:
            raise ValueError("--run-root is not valid with --composite-config")
        source = Path(composite_config_path).expanduser().resolve()
        composite = load_composite_config(source)
        resolved_run_root = _resolve_composite_run(venue=composite.venue, name=composite.name, run_id=run_id)
        entries = composite.evaluations
        if evaluation_id is not None:
            entry = next((item for item in entries if item.id == evaluation_id), None)
            if entry is None:
                expected = [item.id for item in entries]
                raise ValueError(f"unknown evaluation {evaluation_id!r}; expected one of {expected}")
            entries = [entry]
        if not entries:
            raise ValueError("composite has no evaluations to run")
        destinations: list[Path] = []
        for entry in entries:
            document = load_evaluation_config(_resolve_path(source, entry.config))
            destinations.append(
                _evaluate_document(
                    document,
                    entry.id,
                    entry.base_ids,
                    entry.method_ids,
                    resolved_run_root,
                )
            )
        return destinations[0] if len(destinations) == 1 else resolved_run_root / "evaluations"
    else:
        if evaluation_id is not None:
            raise ValueError("--evaluation-id is valid only with --composite-config")
        if run_id is not None:
            raise ValueError("--run-id is valid only with --composite-config")
        if config_path is None:
            raise ValueError("--config is required in standalone mode")
        source = Path(config_path).expanduser().resolve()
        if run_root is None:
            raise ValueError("--run-root is required with --config")
        document = load_evaluation_config(source)
        base_ids, method_ids = document.base_ids, document.method_ids
        resolved_run_root = Path(run_root).expanduser().resolve()
        evaluation_id = document.id

    return _evaluate_document(document, evaluation_id, base_ids, method_ids, resolved_run_root)


def _evaluate_document(
    document: EvaluationDocumentConfig,
    evaluation_id: str,
    base_ids: list[str],
    method_ids: list[str],
    resolved_run_root: Path,
) -> Path:
    manifest_path = resolved_run_root / "composite_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no composite_manifest.json under {resolved_run_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_fits = [fit for fit in manifest.get("fits", []) if fit.get("status") == "complete"]
    if base_ids:
        all_fits = [fit for fit in all_fits if fit["base_id"] in base_ids]
    if not all_fits:
        raise ValueError(f"{manifest_path} records no complete fits matching the requested selection")

    fits_by_base: dict[str, list[dict[str, object]]] = {}
    for fit in all_fits:
        fits_by_base.setdefault(str(fit["base_id"]), []).append(fit)

    evaluations_by_id: dict[str, list[dict[str, object]]] = dict(manifest.get("evaluations", {}))
    cells = evaluations_by_id.setdefault(evaluation_id, [])
    completed = {(cell["base_id"], cell["method_id"], cell["seed"]) for cell in cells}

    for base_id, base_fits in fits_by_base.items():
        selected = [
            fit
            for fit in base_fits
            if not method_ids or fit["method_id"] in method_ids
        ]
        for fit in selected:
            key = (fit["base_id"], fit["method_id"], fit["seed"])
            if key in completed:
                continue
            seed_label = "deterministic" if fit["seed"] is None else f"seed_{fit['seed']}"
            evaluation_dir = (
                resolved_run_root / "evaluations" / evaluation_id / base_id / str(fit["method_id"]) / seed_label
            )
            if evaluation_dir.exists() and not (evaluation_dir / "evaluation_manifest.json").is_file():
                raise RuntimeError(f"refusing to overwrite partial evaluation output: {evaluation_dir}")
            methods = [str(fit["method_family"])]
            method_runs: dict[str, Path] = {str(fit["method_family"]): Path(str(fit["fit_path"]))}
            if not (evaluation_dir / "evaluation_manifest.json").is_file():
                fit_resolved_config = Path(str(fit["fit_path"])) / "resolved_config.yaml"
                runtime = ResolvedExperimentConfig.model_validate(
                    yaml.safe_load(fit_resolved_config.read_text(encoding="utf-8"))
                )
                runtime = runtime.model_copy(update={"sampling": document.sampling, "evaluation": document.evaluation})
                evaluate_from_config(
                    runtime,
                    methods=methods,
                    metrics=document.metrics,
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
                    "evaluation_path": str(evaluation_dir),
                    "status": "complete",
                }
            )
            completed.add(key)

    manifest["evaluations"] = evaluations_by_id
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return resolved_run_root / "evaluations" / evaluation_id


def main(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Standalone evaluation YAML; requires --run-root.",
        ),
    ] = None,
    composite_config: Annotated[
        Path | None,
        typer.Option(
            "--composite-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Composite YAML; its single matching run is resolved automatically.",
        ),
    ] = None,
    evaluation_id: Annotated[
        str | None,
            typer.Option(
                "--evaluation-id",
                help="Evaluation ID to run from --composite-config; omit to run all evaluations.",
            ),
    ] = None,
    run_root: Annotated[
        Path | None,
        typer.Option(
            "--run-root",
            file_okay=False,
            help="Completed composite run to reuse with --config.",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Run ID to select with --composite-config when multiple runs exist."),
    ] = None,
) -> None:
    typer.echo(
        evaluate_composite(
            config,
            composite_config_path=composite_config,
            evaluation_id=evaluation_id,
            run_root=run_root,
            run_id=run_id,
        )
    )


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
