"""Small shared helpers for the interactive documentation notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from simcast.config import SimcastConfig, load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def resolve_config(config_file: str, overrides: Sequence[str] = ()) -> SimcastConfig:
    """Load a repository YAML configuration exactly as the CLI does."""

    return load_config(REPOSITORY_ROOT / config_file, overrides=overrides)


def cache_path(config: SimcastConfig) -> Path:
    """Return the same default cache location used by the training and evaluation CLI."""

    name = config.output.cache_name or config.data.entity_type
    return (Path(config.output.cache_dir).expanduser() / name).resolve()
