"""Regression checks for the executable scientific notebooks."""

from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOKS = Path(__file__).parents[1] / "notebooks"
WORKFLOW_EXPECTED = {
    "00_configuration_and_reproducibility.ipynb": "load_base_config",
    "01_eda_and_preprocessing.ipynb": "build_entity_group",
    "02_chronos_cache_and_features.ipynb": "locate_compatible_cache",
    "03_dependence_models.ipynb": "conditional_kernel",
    "04_evaluation_reporting_and_visualisation.ipynb": "load_composite_config",
}


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_workflow_notebooks_are_valid_and_use_role_specific_interfaces() -> None:
    for filename, expected in WORKFLOW_EXPECTED.items():
        assert expected in _source(NOTEBOOKS / filename)


def test_each_method_notebook_accepts_one_base_and_one_method() -> None:
    paths = sorted((NOTEBOOKS / "dependency_methods").glob("*.ipynb"))
    assert len(paths) == 5
    for path in paths:
        source = _source(path)
        assert "BASE_FILE" in source and "METHOD_FILE" in source
        assert "load_base_config" in source and "load_method_config" in source
        assert "resolve_run_config" in source and "locate_compatible_cache" in source
        assert "train_from_config" in source and "evaluate_from_config" in source
        assert "smoke" not in source.lower()


def test_notebook_examples_match_cache_and_group_interfaces() -> None:
    eda_source = _source(NOTEBOOKS / "01_eda_and_preprocessing.ipynb")
    cache_source = _source(NOTEBOOKS / "02_chronos_cache_and_features.ipynb")
    assert "entity.model_dump()" not in eda_source
    assert "entity.group_id" not in eda_source
    assert "group.group_id" in eda_source
    assert "ds.attrs['output_patch_size']" in cache_source


def test_local_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)")
    for path in [*NOTEBOOKS.glob("*.ipynb"), *(NOTEBOOKS / "dependency_methods").glob("*.ipynb")]:
        for target in link_pattern.findall(_source(path)):
            if "://" not in target:
                assert (path.parent / target).resolve().exists(), f"broken link in {path}: {target}"
