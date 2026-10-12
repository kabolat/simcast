"""Download the revision-pinned dataset files selected by a base configuration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from simcast.config import DataConfig


def allow_patterns(data: str | DataConfig) -> list[str]:
    """Snapshot paths the configured adapter needs; a bare string is a Liander entity type."""

    from simcast.config import DataConfig
    from simcast.data.registry import adapter_spec

    if isinstance(data, str):
        data = DataConfig(entity_type=data)
    return adapter_spec(data.adapter).allow_patterns(data)


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
        allow_patterns=allow_patterns(data),
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
