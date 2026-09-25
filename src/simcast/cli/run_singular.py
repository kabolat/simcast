"""Fit and evaluate one explicitly selected dependence method."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from simcast.cli.build_cache import build_cache_from_config
from simcast.cli.evaluate import evaluate_from_config
from simcast.cli.train_dependence import train_from_config
from simcast.config import base_fingerprint, load_base_config, load_method_config, resolve_run_config
from simcast.experiments import locate_compatible_cache, write_yaml
from simcast.reproducibility import canonical_json_hash, git_commit, utc_run_id


def run_singular(
    base_path: str | Path,
    method_path: str | Path,
    *,
    reference_method_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    rebuild_cache: bool = False,
) -> Path:
    base = load_base_config(base_path)
    primary = load_method_config(method_path)
    reference = load_method_config(reference_method_path) if reference_method_path is not None else None
    if reference is not None and reference.family == primary.family:
        raise ValueError("primary and reference methods must have different families")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (Path("runs/singular") / f"{utc_run_id()}_{base.id}_{primary.id}").resolve()
    )
    destination.mkdir(parents=True, exist_ok=False)
    cache = locate_compatible_cache(base)
    primary_runtime = resolve_run_config(base, primary)
    if rebuild_cache or not cache.is_dir():
        build_cache_from_config(primary_runtime, output_dir=cache, overwrite=rebuild_cache)

    methods = [primary, *(() if reference is None else (reference,))]
    run_paths: dict[str, Path] = {}
    for method in methods:
        runtime = resolve_run_config(base, method)
        run_paths[method.family] = train_from_config(
            runtime,
            cache_dir=cache,
            output_dir=destination / "methods" / method.id,
        )
    evaluation = evaluate_from_config(
        primary_runtime,
        methods=tuple(run_paths),
        method_runs=run_paths,
        cache_dir=cache,
        output_dir=destination / "evaluation",
        base_figures_dir=destination / "figures",
    )
    write_yaml(destination / "resolved_base.yaml", base)
    write_yaml(destination / "resolved_method.yaml", primary)
    if reference is not None:
        write_yaml(destination / "resolved_reference_method.yaml", reference)
    manifest = {
        "schema": "simcast.singular.v2",
        "base_id": base.id,
        "primary_method": primary.id,
        "reference_method": None if reference is None else reference.id,
        "cache_path": str(cache),
        "method_runs": {name: str(path) for name, path in run_paths.items()},
        "evaluation_path": str(evaluation),
        "base_sha256": canonical_json_hash(base.model_dump(mode="json")),
        "marginal_sha256": base_fingerprint(base),
        "method_sha256": canonical_json_hash(primary.model_dump(mode="json")),
        "configuration_sha256": canonical_json_hash(
            {"base": base.model_dump(mode="json"), "methods": [item.model_dump(mode="json") for item in methods]}
        ),
        "git_commit": git_commit(),
    }
    (destination / "singular_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def main(
    base: Annotated[
        Path,
        typer.Option(
            "--base",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Base YAML defining the frozen marginal experiment.",
        ),
    ],
    method: Annotated[
        Path,
        typer.Option(
            "--method",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Method YAML defining the dependence model to fit and evaluate.",
        ),
    ],
    reference_method: Annotated[
        Path | None,
        typer.Option(
            "--reference-method",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional method YAML to evaluate alongside the selected method.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", file_okay=False, help="Explicit singular run directory."),
    ] = None,
    rebuild_cache: Annotated[
        bool,
        typer.Option("--rebuild-cache", help="Rebuild the compatible marginal cache before fitting."),
    ] = False,
) -> None:
    logging.basicConfig(level=logging.INFO)
    typer.echo(
        run_singular(
            base,
            method,
            reference_method_path=reference_method,
            output_dir=output_dir,
            rebuild_cache=rebuild_cache,
        )
    )


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
