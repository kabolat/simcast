"""Regenerate a composite report from an existing runs/ directory.

This never fits or evaluates anything: it only reads already-computed
evaluation cells recorded in composite_manifest.json, so different metrics or
bootstrap settings can be iterated on without touching runs/.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer

from simcast.config import ReportConfig, load_composite_config, load_report_config
from simcast.reporting.composite_report import write_composite_report

LOGGER = logging.getLogger(__name__)


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


def report_composite(
    config_path: str | Path | None = None,
    *,
    composite_config_path: str | Path | None = None,
    report_id: str | None = None,
    evaluation_id: str | None = None,
    run_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> Path:
    if (config_path is None) == (composite_config_path is None):
        raise ValueError("provide exactly one of --config or --composite-config")
    if composite_config_path is not None:
        if run_root is not None:
            raise ValueError("--run-root is not valid with --composite-config")
        source = Path(composite_config_path).expanduser().resolve()
        composite = load_composite_config(source)
        resolved_run_root = _resolve_composite_run(venue=composite.venue, name=composite.name, run_id=run_id)
        entries = composite.reports
        if report_id is not None:
            entry = next((item for item in entries if item.id == report_id), None)
            if entry is None:
                expected = [item.id for item in entries]
                raise ValueError(f"unknown report {report_id!r}; expected one of {expected}")
            entries = [entry]
        if not entries:
            raise ValueError("composite has no reports to run")
        destinations: list[Path] = []
        for entry in entries:
            config = load_report_config((source.parent / entry.config).resolve())
            selected_evaluation_ids = entry.evaluation_ids
            if evaluation_id is not None:
                if evaluation_id not in selected_evaluation_ids:
                    raise ValueError(
                        f"unknown evaluation {evaluation_id!r} for report {entry.id!r}; "
                        f"expected one of {selected_evaluation_ids}"
                    )
                selected_evaluation_ids = [evaluation_id]
            report_root = (
                (Path(output_dir) / entry.id if output_dir is not None else None)
                if len(entries) > 1
                else output_dir
            )
            destinations.extend(
                _write_report(config, evaluation_id, entry.id, resolved_run_root, report_root, force=force)
                for evaluation_id in selected_evaluation_ids
            )
        return destinations[0] if len(destinations) == 1 else resolved_run_root / "reports"
    else:
        if report_id is not None:
            raise ValueError("--report-id is valid only with --composite-config")
        if run_id is not None:
            raise ValueError("--run-id is valid only with --composite-config")
        if config_path is None:
            raise ValueError("--config is required in standalone mode")
        source = Path(config_path).expanduser().resolve()
        config = load_report_config(source)
        report_id = None
        evaluation_ids = [evaluation_id] if evaluation_id is not None else config.evaluation_ids
        if run_root is None:
            raise ValueError("--run-root is required with --config")
        resolved_run_root = Path(run_root).expanduser().resolve()

    selected_ids = evaluation_ids or _manifest_evaluation_ids(resolved_run_root)
    if not selected_ids:
        raise ValueError("run contains no evaluations to report on")
    destinations = [
        _write_report(config, evaluation_id, source.stem, resolved_run_root, output_dir, force=force)
        for evaluation_id in selected_ids
    ]
    return destinations[0] if len(destinations) == 1 else resolved_run_root / "reports" / source.stem


def _write_report(
    config: ReportConfig,
    evaluation_id: str,
    report_id: str,
    resolved_run_root: Path,
    output_dir: str | Path | None,
    *,
    force: bool = False,
) -> Path:
    manifest_path = resolved_run_root / "composite_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no composite_manifest.json under {resolved_run_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluations = manifest.get("evaluations", {})
    cells = [
        cell
        for cell in evaluations.get(evaluation_id, [])
        if cell.get("status") == "complete"
    ]
    if not cells:
        raise ValueError(f"{manifest_path} records no complete evaluation cells to report on")
    declared_metric_sets = [set(_cell_metrics(cell)) for cell in cells]
    declared_metrics = set.intersection(*declared_metric_sets)
    if config.metrics:
        if not set(config.metrics).issubset(declared_metrics):
            missing = sorted(set(config.metrics) - declared_metrics)
            raise ValueError(f"report metrics were not declared by every selected evaluation cell: {missing}")
        metrics = config.metrics
    else:
        metrics = sorted(declared_metrics)
        if not metrics:
            raise ValueError("selected evaluations do not record declared metrics")
    LOGGER.info(
        "Reporting %s/%s from %d evaluation cells with metrics: %s",
        report_id,
        evaluation_id,
        len(cells),
        ", ".join(metrics),
    )

    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve() / evaluation_id
    elif config.output_dir is not None:
        destination = Path(config.output_dir).expanduser().resolve() / evaluation_id
    else:
        destination = resolved_run_root / "reports" / report_id / evaluation_id
    if force and destination.exists():
        shutil.rmtree(destination)
    write_composite_report(
        destination,
        cells,
        config.analysis,
        reference=config.reference,
        metrics=metrics,
    )
    LOGGER.info("Report complete: %s", destination)
    return destination


def _manifest_evaluation_ids(run_root: Path) -> list[str]:
    manifest_path = run_root / "composite_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no composite_manifest.json under {run_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [str(evaluation_id) for evaluation_id in manifest.get("evaluations", {})]


def _cell_metrics(cell: dict[str, object]) -> list[str]:
    path = Path(str(cell["evaluation_path"])) / "evaluation_manifest.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(metric) for metric in payload.get("metrics", [])]


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
        typer.Option(
            "--report-id", help="Report ID to run from --composite-config; omit to run all reports."
        ),
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
    evaluation_id: Annotated[
        str | None,
        typer.Option("--evaluation-id", help="Evaluation ID to include in the report."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Run ID to select with --composite-config when multiple runs exist."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace the selected report directory before regenerating it."),
    ] = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    typer.echo(
        report_composite(
            config,
            composite_config_path=composite_config,
            report_id=report_id,
            run_root=run_root,
            output_dir=output_dir,
            evaluation_id=evaluation_id,
            run_id=run_id,
            force=force,
        )
    )


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)

