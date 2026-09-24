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
import yaml  # type: ignore[import-untyped]

from simcast.config import load_composite_config, load_report_config
from simcast.reporting.composite_report import write_composite_report


def report_composite(
    config_path: str | Path,
    *,
    report_id: str | None = None,
    run_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    source = Path(config_path).expanduser().resolve()
    kind = yaml.safe_load(source.read_text(encoding="utf-8")).get("kind")
    if kind == "composite":
        if report_id is None:
            raise ValueError("--report is required when --config names a composite")
        composite = load_composite_config(source)
        entry = next((item for item in composite.reports if item.id == report_id), None)
        if entry is None:
            expected = [item.id for item in composite.reports]
            raise ValueError(f"unknown report {report_id!r}; expected one of {expected}")
        config = load_report_config((source.parent / entry.config).resolve())
        evaluation_ids = entry.evaluation_ids
    else:
        if report_id is not None:
            raise ValueError("--report is valid only when --config names a composite")
        config = load_report_config(source)
        report_id = None
        evaluation_ids = config.evaluation_ids

    resolved_run_root = Path(run_root).expanduser().resolve() if run_root is not None else config.run_root
    if resolved_run_root is None:
        raise ValueError("run_root must be given via --run-root or the report document")
    resolved_run_root = Path(resolved_run_root).expanduser().resolve()

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
        destination = resolved_run_root / "reports" / (report_id or Path(config_path).stem)
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
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Standalone report YAML or composite YAML containing the report entry.",
        ),
    ],
    report_id: Annotated[
        str | None,
        typer.Option("--report", help="Report ID to select when --config is a composite YAML."),
    ] = None,
    run_root: Annotated[
        Path | None,
        typer.Option(
            "--run-root",
            file_okay=False,
            help="Completed composite run to read; overrides run_root in the report YAML.",
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
    typer.echo(report_composite(config, report_id=report_id, run_root=run_root, output_dir=output_dir))


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)

