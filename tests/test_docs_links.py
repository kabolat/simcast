"""Verify local markdown links in docs/README point at files that exist.

Anchor-only fragments (``#section``) are not resolved against heading slugs;
only the file part of a link is checked, which is enough to catch dangling
references to renamed or deleted files without reimplementing a slugifier.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).parents[1]
DOC_FILES = sorted(
    [ROOT / "README.md", ROOT / "notebooks" / "README.md"] + list((ROOT / "docs").rglob("*.md"))
)
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _local_link_targets(markdown_path: Path) -> list[str]:
    text = markdown_path.read_text(encoding="utf-8")
    return [
        target
        for target in LINK_PATTERN.findall(text)
        if not target.startswith(("http://", "https://", "mailto:"))
    ]


@pytest.mark.parametrize("markdown_path", DOC_FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_markdown_links_resolve_to_existing_files(markdown_path: Path) -> None:
    missing = []
    for target in _local_link_targets(markdown_path):
        file_part = unquote(target.split("#", 1)[0])
        if not file_part:
            continue  # a "#anchor-in-this-file" link; not checked here
        if not (markdown_path.parent / file_part).resolve().is_file():
            missing.append(target)
    assert not missing, f"{markdown_path.relative_to(ROOT)} links to missing files: {missing}"


def test_scientific_math_uses_github_compatible_commands() -> None:
    sources = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").rglob("*.md")),
        *sorted((ROOT / "notebooks").rglob("*.ipynb")),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "\\!" not in text
    assert "\\operatorname" not in text
    assert "\\left" not in text
    assert "\\right" not in text
    assert "\\begin{cases}" not in text
    assert "\\end{cases}" not in text
    assert "\t" not in text
    assert not any(ord(character) < 32 and character not in "\n\r\t" for character in text)
