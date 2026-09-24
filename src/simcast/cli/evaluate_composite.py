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


def _load_document(
    source: Path, *, evaluation_id: str | None
) -> tuple[EvaluationDocumentConfig, list[str], list[str]]:
    kind = yaml.safe_load(source.read_text(encoding="utf-8")).get("kind")
    if kind == "composite":
        if evaluation_id is None:
            raise ValueError("--evaluation is required when --config names a composite")
        composite = load_composite_config(source)
        entry = next((item for item in composite.evaluations if item.id == evaluation_id), None)
        if entry is None:
            expected = [item.id for item in composite.evaluations]
            raise ValueError(f"unknown evaluation {evaluation_id!r}; expected one of {expected}")
        document = load_evaluation_config(_resolve_path(source, entry.config))
        return document, entry.base_ids, entry.method_ids
    if evaluation_id is not None:
        raise ValueError("--evaluation is valid only when --config names a composite")
    document = load_evaluation_config(source)
    return document, document.base_ids, document.method_ids


def evaluate_composite(
    config_path: str | Path,
    *,
    evaluation_id: str | None = None,
    run_root: str | Path | None = None,
) -> Path:
    source = Path(config_path).expanduser().resolve()
    document, base_ids, method_ids = _load_document(source, evaluation_id=evaluation_id)
    evaluation_id = document.id if evaluation_id is None else evaluation_id
    resolved_run_root = Path(run_root).expanduser().resolve() if run_root is not None else document.run_root
    if resolved_run_root is None:
        raise ValueError("run_root must be given via --run-root or the evaluation document")
    resolved_run_root = Path(resolved_run_root).expanduser().resolve()

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
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Standalone evaluation YAML or composite YAML containing the evaluation entry.",
        ),
    ],
    evaluation_id: Annotated[
        str | None,
        typer.Option("--evaluation", help="Evaluation ID to select when --config is a composite YAML."),
    ] = None,
    run_root: Annotated[
        Path | None,
        typer.Option(
            "--run-root",
            file_okay=False,
            help="Completed composite run to reuse; overrides run_root in a standalone evaluation YAML.",
        ),
    ] = None,
) -> None:
    typer.echo(evaluate_composite(config, evaluation_id=evaluation_id, run_root=run_root))


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
