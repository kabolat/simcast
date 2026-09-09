import json
from pathlib import Path

import pytest

from simcast.config import base_fingerprint, load_base_config
from simcast.experiments import locate_compatible_cache

CONFIGS = Path(__file__).parents[1] / "configs"


def _base_with_cache_root(root: Path):
    base = load_base_config(CONFIGS / "bases/liander2024/transformer.yaml")
    return base.model_copy(update={"output": base.output.model_copy(update={"cache_dir": root})})


def test_cache_reuse_depends_on_metadata_not_directory_name(tmp_path: Path) -> None:
    base = _base_with_cache_root(tmp_path)
    compatible = tmp_path / "an_unrelated_human_name"
    compatible.mkdir()
    (compatible / "metadata.json").write_text(
        json.dumps({"base_config": base.model_dump(mode="json")}), encoding="utf-8"
    )
    incompatible = tmp_path / "looks_plausible"
    incompatible.mkdir()
    other = base.model_copy(update={"pit": base.pit.model_copy(update={"eps": 1.0e-6})})
    (incompatible / "metadata.json").write_text(
        json.dumps({"base_config": other.model_dump(mode="json")}), encoding="utf-8"
    )

    assert locate_compatible_cache(base) == compatible


def test_canonical_cache_with_wrong_metadata_is_never_trusted(tmp_path: Path) -> None:
    base = _base_with_cache_root(tmp_path)
    canonical = tmp_path / f"{base.id}-{base_fingerprint(base)[:12]}"
    canonical.mkdir()
    (canonical / "metadata.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible metadata"):
        locate_compatible_cache(base)


def test_ambiguous_compatible_caches_are_rejected(tmp_path: Path) -> None:
    base = _base_with_cache_root(tmp_path)
    payload = json.dumps({"base_config": base.model_dump(mode="json")})
    for name in ("first", "second"):
        path = tmp_path / name
        path.mkdir()
        (path / "metadata.json").write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="multiple compatible caches"):
        locate_compatible_cache(base)
