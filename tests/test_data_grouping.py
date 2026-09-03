from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from simcast.data.grouping import build_entity_group, load_target_metadata
from simcast.data.liander2024 import read_liander_parquet, summarize_load_measurements


def _write_targets(path: Path, *, duplicate: bool = False) -> None:
    second_group = "transformer" if duplicate else "station_installation"
    path.write_text(
        f"""
- name: Shared name
  group_name: transformer
  latitude: 52.0
  longitude: 5.0
  benchmark_start: 2024-03-01T00:00:00Z
  benchmark_end: 2024-12-31T23:59:59Z
  train_start: 2024-01-01T00:00:00Z
- name: Shared name
  group_name: {second_group}
  latitude: 53.0
  longitude: 6.0
""".strip(),
        encoding="utf-8",
    )


def test_canonical_ids_disambiguate_equal_names_and_group_is_homogeneous(tmp_path: Path) -> None:
    metadata_path = tmp_path / "liander2024_targets.yaml"
    _write_targets(metadata_path)

    metadata = load_target_metadata(metadata_path)
    assert [entity.entity_id for entity in metadata] == [
        "transformer::Shared name",
        "station_installation::Shared name",
    ]

    group = build_entity_group(metadata_path, "transformer")
    assert group.entity_ids == ["transformer::Shared name"]
    assert group.metadata == {"entity_type": "transformer", "entity_count": 1}
    assert all(entity.group_name == "transformer" for entity in group.entities)


def test_duplicate_canonical_id_is_rejected(tmp_path: Path) -> None:
    metadata_path = tmp_path / "liander2024_targets.yaml"
    _write_targets(metadata_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate canonical"):
        load_target_metadata(metadata_path)


def test_parquet_schema_and_missingness_are_normalized(tmp_path: Path) -> None:
    metadata_path = tmp_path / "liander2024_targets.yaml"
    metadata_path.write_text(
        """
- name: A
  group_name: transformer
  latitude: 52.0
  longitude: 5.0
- name: B
  group_name: transformer
  latitude: 53.0
  longitude: 6.0
""".strip(),
        encoding="utf-8",
    )
    timestamps = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    target_dir = tmp_path / "load_measurements" / "transformer"
    target_dir.mkdir(parents=True)
    for name, values in (("A", [1.0, float("nan"), 3.0]), ("B", [4.0, 5.0, 6.0])):
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "load": values,
                "available_at": timestamps,
                "__index_level_0__": range(3),
            }
        ).to_parquet(target_dir / f"{name}.parquet")

    frame = read_liander_parquet(target_dir / "A.parquet")
    assert frame.index.equals(pd.DatetimeIndex(timestamps, name="timestamp"))
    assert "timestamp" not in frame.columns
    assert "__index_level_0__" not in frame.columns

    stats = summarize_load_measurements(tmp_path, build_entity_group(metadata_path))
    assert stats.entity_count == 2
    assert stats.row_count == 6
    assert stats.missing_count == 1
    assert stats.missing_percent == pytest.approx(100 / 6)
    assert stats.start == timestamps[0]
    assert stats.end == timestamps[-1]
