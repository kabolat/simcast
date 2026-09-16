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
        compact_source = source.replace(" ", "")
        assert "BASE_FILE" in source and "METHOD_FILE" in source
        assert "load_base_config" in source and "load_method_config" in source
        assert "resolve_run_config" in source and "locate_compatible_cache" in source
        assert "train_from_config" in source and "evaluate_from_config" in source
        assert "CACHE_ROOT=PROJECT_ROOT/'artifacts'/'cache'" in compact_source
        assert "output.cache_dir={CACHE_ROOT}" in source
        assert "FIT_IF_MISSING" in source and "RUN_EVALUATION" in source
        assert "TRAIN_IF_MISSING" not in source
        assert "smoke" not in source.lower()


def test_method_notebooks_trace_dependence_into_scenarios() -> None:
    for path in sorted((NOTEBOOKS / "dependency_methods").glob("0[1-4]_*.ipynb")):
        source = _source(path)
        assert "dependence_pit_scores" in source
        assert "sample_gaussian_uniforms" in source
        assert "generate_scenarios" in source
        assert "aggregate_samples" in source


def test_notebook_examples_match_cache_and_group_interfaces() -> None:
    eda_source = _source(NOTEBOOKS / "01_eda_and_preprocessing.ipynb")
    cache_source = _source(NOTEBOOKS / "02_chronos_cache_and_features.ipynb")
    assert "entity.model_dump()" not in eda_source
    assert "entity.group_id" not in eda_source
    assert "group.group_id" in eda_source
    assert "ds.attrs['output_patch_size']" in cache_source


def test_report_walkthrough_notebook_reads_without_regenerating_by_default() -> None:
    source = _source(NOTEBOOKS / "reporting/01_report_walkthrough.ipynb")
    assert "report_composite" in source
    assert "REPORT_DIR" in source and "report_summary.md" in source
    assert "REGENERATE = False" in source


def test_local_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)")
    for path in [
        *NOTEBOOKS.glob("*.ipynb"),
        *(NOTEBOOKS / "dependency_methods").glob("*.ipynb"),
        *(NOTEBOOKS / "reporting").glob("*.ipynb"),
    ]:
        for target in link_pattern.findall(_source(path)):
            if "://" not in target:
                assert (path.parent / target).resolve().exists(), f"broken link in {path}: {target}"
