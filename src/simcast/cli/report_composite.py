"""Regenerate a composite report from an existing runs/ directory.

This never fits or evaluates anything: it only reads an existing
``composite_manifest.json`` and re-runs the reporting stage, so different
metrics or bootstrap settings can be iterated on without touching ``runs/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]

from simcast.config import load_composite_config, load_report_config
from simcast.reporting.composite_report import write_composite_report


def _default_output_dir(run_root: Path, config_path: Path) -> Path:
    parts = run_root.parts
    base = Path(*("reports", *parts[parts.index("runs") + 1 :])) if "runs" in parts else run_root / "report"
    return base / config_path.stem


def report_composite(
    config_path: str | Path,
    *,
    report_id: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    source = Path(config_path).expanduser().resolve()
    kind = yaml.safe_load(source.read_text(encoding="utf-8")).get("kind")
    selected_evaluations: set[str] | None = None
    if kind == "composite":
        if report_id is None:
            raise ValueError("--report is required when --config names a composite")
        composite = load_composite_config(source)
        entry = next((item for item in composite.reports if item.id == report_id), None)
        if entry is None:
            expected = [item.id for item in composite.reports]
            raise ValueError(f"unknown report {report_id!r}; expected one of {expected}")
        report_path = (source.parent / entry.config).resolve()
        config = load_report_config(report_path)
        selected_evaluations = set(entry.evaluation_ids)
    else:
        if report_id is not None:
            raise ValueError("--report is valid only when --config names a composite")
        config = load_report_config(source)
    run_root = Path(config.run_root).expanduser().resolve()
    manifest_path = run_root / "composite_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no composite_manifest.json under {run_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = [
        cell
        for cell in manifest.get("cells", [])
        if cell.get("status") == "complete"
        and (selected_evaluations is None or cell.get("evaluation_id") in selected_evaluations)
    ]
    if not cells:
        raise ValueError(f"{manifest_path} records no complete cells to report on")

    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
    elif config.output_dir is not None:
        destination = Path(config.output_dir).expanduser().resolve()
    else:
        destination = _default_output_dir(run_root, Path(config_path).expanduser().resolve()).resolve()
    write_composite_report(destination, cells, config.analysis, metrics=config.metrics)
    return destination


def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False, readable=True)],
    report_id: Annotated[str | None, typer.Option("--report")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False)] = None,
) -> None:
    typer.echo(report_composite(config, report_id=report_id, output_dir=output_dir))


if __name__ == "__main__":
    typer.run(main)
