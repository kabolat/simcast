"""Read current and historical evaluation manifests without rewriting them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationArtifact:
    """An evaluation directory and its unmodified manifest payload."""

    path: Path
    manifest: dict[str, Any]


def read_evaluation_artifacts(root: str | Path) -> list[EvaluationArtifact]:
    """Discover readable evaluation manifests beneath ``root``.

    The reader intentionally performs no schema migration. This keeps historical
    artifacts inspectable while guaranteeing that opening them cannot alter them.
    """

    source = Path(root).expanduser().resolve()
    artifacts: list[EvaluationArtifact] = []
    for manifest_path in sorted(source.rglob("evaluation_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            artifacts.append(EvaluationArtifact(manifest_path.parent, payload))
    return artifacts
