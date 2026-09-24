"""Regenerate a composite report from an existing runs/ directory.

This never fits or evaluates anything: it only reads already-computed
evaluation cells recorded in composite_manifest.json, so different metrics or
bootstrap settings can be iterated on without touching runs/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from simcast.config import load_composite_config, load_report_config
from simcast.reporting.composite_report import write_composite_report


def _resolve_composite_run(*, venue: str, name: str) -> Path:
    run_dir = Path("runs") / venue / name
    candidates = sorted(
        path for path in run_dir.iterdir() if path.is_dir() and (path / "composite_manifest.json").is_file()
    ) if run_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no completed composite run found under {run_dir}")
    if len(candidates) > 1:
        raise ValueError(f"multiple composite runs found under {run_dir}; use a single run directory")
    return candidates[0].resolve()


def report_composite(
    config_path: str | Path | None = None,
    *,
    composite_config_path: str | Path | None = None,
    report_id: str | None = None,
    run_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    if (config_path is None) == (composite_config_path is None):
        raise ValueError("provide exactly one of --config or --composite-config")
    if composite_config_path is not None:
        if run_root is not None:
            raise ValueError("--run-root is not valid with --composite-config")
        source = Path(composite_config_path).expanduser().resolve()
        composite = load_composite_config(source)
        if report_id is None:
            raise ValueError("--report-id is required with --composite-config")
        entry = next((item for item in composite.reports if item.id == report_id), None)
        if entry is None:
            expected = [item.id for item in composite.reports]
            raise ValueError(f"unknown report {report_id!r}; expected one of {expected}")
        config = load_report_config((source.parent / entry.config).resolve())
        evaluation_ids = entry.evaluation_ids
        resolved_run_root = _resolve_composite_run(venue=composite.venue, name=composite.name)
    else:
        if report_id is not None:
            raise ValueError("--report-id is valid only with --composite-config")
        if config_path is None:
            raise ValueError("--config is required in standalone mode")
        source = Path(config_path).expanduser().resolve()
        config = load_report_config(source)
        report_id = None
        evaluation_ids = config.evaluation_ids
        if run_root is None:
            raise ValueError("--run-root is required with --config")
        resolved_run_root = Path(run_root).expanduser().resolve()

    manifest_path = resolved_run_root / "composite_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no composite_manifest.json under {resolved_run_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluations = manifest.get("evaluations", {})
    selected_ids = evaluation_ids or list(evaluations)
    cells = [
        cell
        for evaluation_id in selected_ids
        for cell in evaluations.get(evaluation_id, [])
        if cell.get("status") == "complete"
    ]
    if not cells:
        raise ValueError(f"{manifest_path} records no complete evaluation cells to report on")

    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
    elif config.output_dir is not None:
        destination = Path(config.output_dir).expanduser().resolve()
    else:
        destination = resolved_run_root / "reports" / (report_id or source.stem)
    write_composite_report(
        destination,
        cells,
        config.analysis,
        reference=config.reference,
        metrics=config.metrics,
    )
    return destination


def main(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Standalone report YAML; requires --run-root.",
        ),
    ] = None,
    composite_config: Annotated[
        Path | None,
        typer.Option(
            "--composite-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Composite YAML; its single matching run is resolved automatically.",
        ),
    ] = None,
    report_id: Annotated[
        str | None,
        typer.Option("--report-id", help="Report ID to select from --composite-config."),
    ] = None,
    run_root: Annotated[
        Path | None,
        typer.Option(
            "--run-root",
            file_okay=False,
            help="Completed composite run to read with --config.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Destination for the report; defaults to <run_root>/reports/<report-id>.",
        ),
    ] = None,
) -> None:
    typer.echo(
        report_composite(
            config,
            composite_config_path=composite_config,
            report_id=report_id,
            run_root=run_root,
            output_dir=output_dir,
        )
    )


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)

