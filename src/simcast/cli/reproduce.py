"""Re-run training or evaluation from a saved run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from simcast.cli.evaluate import evaluate_from_config
from simcast.cli.train_dependence import train_from_config
from simcast.config import load_config


def reproduce_run(run_dir: str | Path, *, output_dir: str | Path | None = None) -> Path:
    """Re-run a saved training or evaluation configuration."""

    source = Path(run_dir).expanduser().resolve()
    config_path = source / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"saved resolved configuration not found: {config_path}")
    config = load_config(config_path)
    evaluation_manifest = source / "evaluation_manifest.json"
    if evaluation_manifest.is_file():
        payload: dict[str, Any] = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
        methods = payload.get("methods")
        method_runs = payload.get("method_runs")
        if not isinstance(methods, list) or not all(isinstance(item, str) for item in methods):
            raise ValueError("evaluation manifest has invalid methods")
        if not isinstance(method_runs, dict):
            raise ValueError("evaluation manifest has invalid method runs")
        return evaluate_from_config(
            config,
            methods=methods,
            method_runs={str(name): Path(path) for name, path in method_runs.items()},
            cache_dir=Path(payload["cache_path"]),
            output_dir=output_dir,
        )

    metadata_path = source / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"saved run metadata not found: {metadata_path}")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return train_from_config(config, cache_dir=Path(metadata["cache_path"]), output_dir=output_dir)


def main(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False)] = None,
) -> None:
    typer.echo(reproduce_run(run_dir, output_dir=output_dir))


if __name__ == "__main__":
    typer.run(main)
