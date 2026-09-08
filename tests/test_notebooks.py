"""Regression checks for the executable interactive documentation."""

from __future__ import annotations

import json
import os
import sys
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


def test_notebook_examples_match_the_cache_and_group_interfaces() -> None:
    eda_source = (NOTEBOOKS / "01_eda_and_preprocessing.ipynb").read_text(encoding="utf-8")
    cache_source = (NOTEBOOKS / "02_chronos_cache_and_features.ipynb").read_text(encoding="utf-8")
    assert "entity.model_dump()" not in eda_source
    assert "entity.group_id" not in eda_source
    assert "group.group_id" in eda_source
    assert "asdict(entity)" in eda_source
    assert "ds.attrs['output_patch_size']" in cache_source


def test_notebook_helper_makes_the_local_source_package_importable() -> None:
    sys.path.insert(0, str(NOTEBOOKS))
    from _helpers import REPOSITORY_ROOT, SOURCE_ROOT, resolve_config

    assert str(SOURCE_ROOT) in sys.path
    original_directory = Path.cwd()
    try:
        os.chdir(NOTEBOOKS)
        config = resolve_config("configs/base.yaml", ("data.local_dir=data/notebook-check",))
        assert Path(config.data.local_dir).resolve() == REPOSITORY_ROOT / "data" / "notebook-check"
    finally:
        os.chdir(original_directory)


def test_notebook_cache_path_matches_cli_fallback() -> None:
    sys.path.insert(0, str(NOTEBOOKS))
    from _helpers import REPOSITORY_ROOT, cache_path, resolve_config

    config = resolve_config("configs/base.yaml", ("data.entity_type=transformer",))
    assert cache_path(config) == REPOSITORY_ROOT / "artifacts/cache/liander2024_transformer"

    named = resolve_config("configs/base.yaml", ("output.cache_name=custom_cache",))
    assert cache_path(named) == REPOSITORY_ROOT / "artifacts/cache/custom_cache"
