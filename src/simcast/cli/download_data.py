"""Download the revision-pinned Liander2024 files used by an experiment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import typer


def allow_patterns(entity_type: str) -> list[str]:
    """Return the minimum snapshot paths needed by the configured entity type."""

    return [
        "liander2024_targets.yaml",
        f"load_measurements/{entity_type}/*.parquet",
        f"weather_measurements/{entity_type}/*.parquet",
        f"weather_forecasts_versioned/{entity_type}/*.parquet",
    ]


def download_data(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
    snapshot_download_fn: Callable[..., str] | None = None,
) -> Path:
    """Load configuration only when invoked, then fetch its pinned dataset files."""

    from simcast.config import load_base_config

    config = load_base_config(Path(config_path), overrides=overrides)
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download

    data = config.data
    if not data.revision:
        raise ValueError("data.revision must pin a dataset revision")
    local_dir = Path(data.local_dir).expanduser().resolve()
    downloaded = snapshot_download_fn(
        repo_id=data.dataset_id,
        repo_type="dataset",
        revision=data.revision,
        local_dir=local_dir,
        allow_patterns=allow_patterns(data.entity_type),
    )
    return Path(downloaded)


def main(
    base: Annotated[
        Path,
        typer.Option(
            "--base",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Base YAML identifying the data and physical group to download.",
        ),
    ],
    override: Annotated[
        list[str] | None,
        typer.Option("--set", help="Configuration override as dotted.path=value"),
    ] = None,
) -> None:
    """Download data selected by a Simcast YAML configuration."""

    path = download_data(base, overrides=override or ())
    typer.echo(path)


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
