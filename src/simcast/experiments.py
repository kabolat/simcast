"""Shared mechanics for role-specific scientific experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import TypeAdapter

from simcast.config import (
    BaseExperimentConfig,
    MethodConfig,
    base_fingerprint,
    deep_merge,
    marginal_fingerprint,
)


def update_base(base: BaseExperimentConfig, overrides: Mapping[str, Any]) -> BaseExperimentConfig:
    return BaseExperimentConfig.model_validate(deep_merge(base.model_dump(mode="python"), overrides))


def update_method(method: MethodConfig, overrides: Mapping[str, Any]) -> MethodConfig:
    values = deep_merge(method.model_dump(mode="python"), overrides)
    return TypeAdapter(MethodConfig).validate_python(values)


def write_yaml(path: Path, value: Any) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _cache_fingerprint(path: Path) -> str | None:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    resolved = metadata.get("base_config") or metadata.get("resolved_config")
    if not isinstance(resolved, dict):
        return None
    protocol = resolved.get("protocol")
    if isinstance(protocol, dict) and not protocol.get("ordered_entity_ids"):
        group = metadata.get("entity_group", {})
        resolved = deep_merge(
            resolved,
            {
                "protocol": {
                    "full_group_only": True,
                    "ordered_entity_ids": group.get("entity_ids", []),
                    "entity_count": len(group.get("entity_ids", [])),
                }
            },
        )
    try:
        return marginal_fingerprint(resolved)
    except (KeyError, TypeError):
        return None


def locate_compatible_cache(base: BaseExperimentConfig) -> Path:
    """Return the canonical or unique metadata-compatible marginal cache path."""

    root = Path(base.output.cache_dir).expanduser().resolve()
    expected = base_fingerprint(base)
    canonical = root / f"{base.id}-{expected[:12]}"
    if canonical.exists():
        if _cache_fingerprint(canonical) != expected:
            raise ValueError(f"canonical cache exists with incompatible metadata: {canonical}")
        return canonical
    compatible = (
        [path for path in sorted(root.iterdir()) if path.is_dir() and _cache_fingerprint(path) == expected]
        if root.is_dir()
        else []
    )
    if len(compatible) > 1:
        raise ValueError(f"multiple compatible caches found for {base.id}: {compatible}")
    return compatible[0] if compatible else canonical
