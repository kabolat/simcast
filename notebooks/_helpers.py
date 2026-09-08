"""Small shared helpers for the interactive documentation notebooks."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from simcast.config import SimcastConfig, load_config  # noqa: E402


def resolve_config(config_file: str, overrides: Sequence[str] = ()) -> SimcastConfig:
    """Load a configuration with the repository as the CLI-equivalent working directory."""

    os.chdir(REPOSITORY_ROOT)
    return load_config(REPOSITORY_ROOT / config_file, overrides=overrides)


def cache_path(config: SimcastConfig) -> Path:
    """Return the same default cache location used by the training and evaluation CLI."""

    name = config.output.cache_name or f"liander2024_{config.data.entity_type}"
    return (Path(config.output.cache_dir).expanduser() / name).resolve()
