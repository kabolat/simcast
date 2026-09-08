"""Build the PowerTech 2027 report strictly from saved experiment artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from simcast.reporting.powertech import build_powertech_report


def main(
    input_root: Annotated[Path, typer.Option("--input-root", exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)] = Path("reports/powertech2027"),
) -> None:
    typer.echo(build_powertech_report(input_root, output_dir))


if __name__ == "__main__":
    typer.run(main)
