"""Small, deterministic hashes for experiment provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

PACKAGES = ("numpy", "pandas", "torch", "scikit-learn", "xarray", "zarr", "transformers")


def utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in PACKAGES},
        "git_commit": git_commit(),
    }


def sha256_file(path: str | Path) -> str:
    """Hash a saved checkpoint or artifact without loading it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: Any) -> str:
    """Hash the canonical resolved configuration."""

    return canonical_json_hash(config.model_dump(mode="json"))
