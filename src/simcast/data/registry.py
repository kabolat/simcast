"""Registry mapping ``data.adapter`` identifiers to dataset adapters.

Adding a dataset means writing one module that exposes the functions below and adding one line here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simcast.config import DataConfig, ResolvedExperimentConfig
    from simcast.data.adapter import DatasetAdapter


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    load: Callable[[ResolvedExperimentConfig], DatasetAdapter]
    allow_patterns: Callable[..., list[str]]
    validate_data_config: Callable[[DataConfig], None]
    default_cache_name: Callable[[DataConfig], str]
    download: Callable[..., Path] | None


def adapter_spec(name: str) -> AdapterSpec:
    from simcast.data import feederbw, gefcom2014, liander2024

    specs = {
        liander2024.ADAPTER_NAME: liander2024,
        gefcom2014.ADAPTER_NAME: gefcom2014,
        feederbw.ADAPTER_NAME: feederbw,
    }
    if name not in specs:
        raise ValueError(f"unknown data.adapter {name!r}; expected one of: {', '.join(sorted(specs))}")
    module = specs[name]
    return AdapterSpec(
        load=module.load_adapter,
        allow_patterns=module.allow_patterns,
        validate_data_config=module.validate_data_config,
        default_cache_name=module.default_cache_name,
        download=getattr(module, "download_data", None),
    )
