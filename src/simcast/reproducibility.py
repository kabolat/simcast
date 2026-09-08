"""Small, deterministic hashes for experiment provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from simcast.config import SimcastConfig


def sha256_file(path: str | Path) -> str:
    """Hash a saved checkpoint or artifact without loading it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: SimcastConfig) -> str:
    """Hash the canonical resolved configuration."""

    payload: Any = config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
