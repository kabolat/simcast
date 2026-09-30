"""Shared path resolution for composite CLI workflows."""

from pathlib import Path


def _resolve_path(source: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (source.parent / value).resolve()


def _resolve_composite_run(*, venue: str, name: str, run_id: str | None = None) -> Path:
    run_dir = Path("runs") / venue / name
    if run_id is not None:
        if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in run_id):
            raise ValueError("run_id must be a lowercase safe slug")
        selected = run_dir / run_id
        if not (selected / "composite_manifest.json").is_file():
            raise FileNotFoundError(f"no completed composite run found at {selected}")
        return selected.resolve()
    candidates = sorted(
        path for path in run_dir.iterdir() if path.is_dir() and (path / "composite_manifest.json").is_file()
    ) if run_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no completed composite run found under {run_dir}")
    if len(candidates) > 1:
        raise ValueError(f"multiple composite runs found under {run_dir}; use a single run directory")
    return candidates[0].resolve()