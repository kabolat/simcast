from __future__ import annotations

import json
from pathlib import Path

import pytest
from _liander_fixture import digest_windows, liander_fixture_config, write_liander_fixture

from simcast.cli.build_cache import build_cache_from_config
from simcast.config import ProtocolConfig, base_fingerprint, load_base_config, marginal_fingerprint
from simcast.data.adapter import IneligibleWindow
from simcast.data.liander2024 import load_adapter
from simcast.data.windows import build_aligned_group_windows

GOLDEN_INPUT_DIGEST = "38105c3547bbd9de1c7f0eb7dd9c8fa2ce6d053643e25cdeb65cf726ca353a79"


def test_liander_adapter_preserves_resolved_window_inputs(tmp_path: Path) -> None:
    write_liander_fixture(tmp_path)
    config = liander_fixture_config(tmp_path)
    adapter = load_adapter(config)
    start, end = adapter.coverage()
    forecast = config.forecast
    windows = build_aligned_group_windows(
        start,
        end,
        entity_ids=adapter.group.entity_ids,
        lookback_steps=forecast.lookback_steps,
        horizon_steps=forecast.horizon_steps,
        frequency_minutes=forecast.frequency_minutes,
        origin_stride_steps=forecast.origin_stride_steps,
        origin_time=forecast.origin_time,
    )

    def inputs_for(window):
        try:
            prepared = adapter.prepare_window(window)
        except IneligibleWindow as error:
            return str(error)
        return prepared.inputs, prepared.true_y, prepared.history_missing

    assert adapter.group.entity_ids == ["transformer::A", "transformer::B", "transformer::C"]
    assert len(windows) == 69
    assert sum(isinstance(inputs_for(window), str) for window in windows) == 6
    assert digest_windows(windows, inputs_for) == GOLDEN_INPUT_DIGEST


def test_existing_liander_cache_fingerprints_are_unchanged() -> None:
    repo = Path(__file__).parents[1]
    golden = json.loads((Path(__file__).parent / "_liander_fingerprints.json").read_text(encoding="utf-8"))
    current = {}
    for path in sorted((repo / "configs/bases/liander2024").glob("*.yaml")):
        if path.name == "common.yaml":
            continue
        base = load_base_config(path)
        current[base.id] = base_fingerprint(base)
    from simcast.experiments import update_base

    base = load_base_config(repo / "configs/bases/liander2024/transformer.yaml")
    current["transformer_long"] = base_fingerprint(update_base(base, {"forecast": {"lookback_steps": 2688}}))
    assert current == golden


def test_legacy_resolved_config_without_adapter_fields_keeps_fingerprint() -> None:
    repo = Path(__file__).parents[1]
    base = load_base_config(repo / "configs/bases/liander2024/transformer.yaml")
    legacy = base.model_dump(mode="json")
    legacy["data"].pop("adapter")
    legacy["data"].pop("options")
    assert marginal_fingerprint(legacy) == base_fingerprint(base)


def test_gefcom_fingerprint_includes_adapter_and_options() -> None:
    repo = Path(__file__).parents[1]
    base_path = repo / "configs/bases/gefcom2014/wind.yaml"
    base = load_base_config(base_path)
    changed_run = load_base_config(base_path, overrides=["data.options.nwp_run_length_hours=12"])
    changed_adapter = base.model_copy(
        deep=True,
        update={"data": base.data.model_copy(update={"adapter": "liander2024"})},
    )
    assert base_fingerprint(base) != base_fingerprint(changed_run)
    assert base_fingerprint(base) != base_fingerprint(changed_adapter)


def test_adapter_rejects_protocol_entity_order_mismatch(tmp_path: Path) -> None:
    write_liander_fixture(tmp_path)
    config = liander_fixture_config(tmp_path)
    mismatched = config.model_copy(
        update={
            "protocol": ProtocolConfig(
                ordered_entity_ids=["transformer::C", "transformer::B", "transformer::A"],
                entity_count=3,
            )
        }
    )
    with pytest.raises(ValueError, match="does not match the adapter's entity group"):
        build_cache_from_config(mismatched, output_dir=tmp_path / "cache")
