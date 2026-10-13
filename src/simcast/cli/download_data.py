"""Download the revision-pinned dataset files selected by a base configuration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from simcast.config import DataConfig


def allow_patterns(data: str | DataConfig, *, entity_ids: Sequence[str] = ()) -> list[str]:
    """Snapshot paths the configured adapter needs; a bare string is a Liander entity type."""

    from simcast.config import DataConfig
    from simcast.data.registry import adapter_spec

    if isinstance(data, str):
        data = DataConfig(entity_type=data)
    return adapter_spec(data.adapter).allow_patterns(data, entity_ids=entity_ids)


def download_data(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
    snapshot_download_fn: Callable[..., str] | None = None,
    all_feeders: bool = False,
) -> Path:
    """Load configuration only when invoked, then fetch its pinned dataset files."""

    from simcast.config import DatasetDownloadConfig, load_download_config
    from simcast.data.registry import adapter_spec

    config = load_download_config(Path(config_path), overrides=overrides)
    adapter = adapter_spec(config.data.adapter)
    if isinstance(config, DatasetDownloadConfig):
        if adapter.download is None:
            raise ValueError(f"adapter {config.data.adapter!r} does not support dataset-wide download profiles")
        return Path(adapter.download(config, snapshot_download_fn=snapshot_download_fn))
    if adapter.download is not None:
        return Path(
            adapter.download(
                config,
                snapshot_download_fn=snapshot_download_fn,
                all_feeders=all_feeders,
            )
        )
    if all_feeders:
        raise ValueError("--all-feeders is supported only by the FeederBW adapter")
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
        allow_patterns=allow_patterns(data, entity_ids=config.protocol.ordered_entity_ids),
    )
    return Path(downloaded)


def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "--base",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Experiment base or dataset-wide download profile.",
        ),
    ],
    override: Annotated[
        list[str] | None,
        typer.Option("--set", help="Configuration override as dotted.path=value"),
    ] = None,
    all_feeders: Annotated[
        bool,
        typer.Option(
            "--all-feeders",
            help="For FeederBW, download all five measurement batches instead of only the configured group.",
        ),
    ] = False,
) -> None:
    """Download data selected by a Simcast YAML configuration."""

    path = download_data(config_path, overrides=override or (), all_feeders=all_feeders)
    typer.echo(path)


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
