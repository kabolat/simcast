"""Regression checks for the executable interactive documentation."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).parents[1] / "notebooks"
EXPECTED = {
    "00_configuration_and_reproducibility.ipynb": "resolve_config",
    "01_eda_and_preprocessing.ipynb": "_load_frames",
    "02_chronos_cache_and_features.ipynb": "build_cache_from_config",
    "03_dependence_models.ipynb": "train_from_config",
    "04_evaluation_reporting_and_visualisation.ipynb": "run_experiment_from_config",
}


def test_interactive_notebooks_are_valid_and_use_pipeline_functions() -> None:
    for filename, pipeline_function in EXPECTED.items():
        path = NOTEBOOKS / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert pipeline_function in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
