from pathlib import Path

import pytest
from pydantic import ValidationError

from simcast.config import (
    CHRONOS_MODEL_REVISION,
    CHRONOS_SOURCE_REVISION,
    LIANDER2024_REVISION,
    SimcastConfig,
    deep_merge,
    load_config,
    parse_overrides,
)

CONFIGS = Path(__file__).parents[1] / "configs"


def test_base_protocol_and_pins() -> None:
    config = load_config(CONFIGS / "base.yaml")

    assert config.data.revision == LIANDER2024_REVISION
    assert config.chronos.source_revision == CHRONOS_SOURCE_REVISION
    assert config.chronos.model_revision == CHRONOS_MODEL_REVISION
    assert (config.forecast.lookback_steps, config.forecast.horizon_steps) == (672, 96)
    assert config.forecast.origin_time.isoformat(timespec="minutes") == "23:45"
    assert config.features.use_entity_id_embedding is False
    assert config.sampling.num_samples == 4096


@pytest.mark.parametrize(
    ("filename", "method"),
    [
        ("method_independent.yaml", "independent"),
        ("method_static_gaussian.yaml", "static_gaussian"),
        ("method_conditional_low_rank.yaml", "conditional_low_rank"),
        ("method_set_aware_low_rank.yaml", "set_aware_low_rank"),
        ("method_conditional_kernel.yaml", "conditional_kernel"),
        ("method_conditional_kernel_smoke.yaml", "conditional_kernel"),
    ],
)
def test_method_configs_resolve(filename: str, method: str) -> None:
    config = load_config(CONFIGS / filename)

    assert config.dependence.method == method
    assert config.data.entity_type == "transformer"
    assert config.covariates.weather[0] == "temperature_2m"


@pytest.mark.parametrize(
    ("filename", "entity_type", "repair"),
    [
        ("liander2024_transformer.yaml", "transformer", "none"),
        ("liander2024_solar_park.yaml", "solar_park", "isotonic"),
        ("liander2024_wind_park.yaml", "wind_park", "none"),
        ("liander2024_mv_feeder.yaml", "mv_feeder", "none"),
        ("liander2024_station_installation.yaml", "station_installation", "none"),
    ],
)
def test_entity_type_configs_resolve(filename: str, entity_type: str, repair: str) -> None:
    config = load_config(CONFIGS / filename)

    assert config.data.entity_type == entity_type
    assert config.pit.monotone_repair == repair
    assert config.output.cache_name is not None


def test_kernel_config_is_bounded_smoke() -> None:
    config = load_config(CONFIGS / "method_conditional_kernel_smoke.yaml")

    assert config.dependence.conditional_kernel.smoke_only
    assert config.dependence.conditional_kernel.smoke_max_origins == 32
    assert config.training.epochs == 5
    assert not config.subset_training.enabled


def test_full_kernel_config_uses_core_training_protocol() -> None:
    config = load_config(CONFIGS / "method_conditional_kernel.yaml")

    assert not config.dependence.conditional_kernel.smoke_only
    assert config.training.epochs == 100
    assert config.training.patience == 12
    assert config.subset_training.enabled


def test_recursive_inheritance_environment_and_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "data:\n  local_dir: ${TEST_SIMCAST_DATA:-fallback}\n"
        "forecast:\n  lookback_steps: 100\n"
        "covariates:\n  weather: [old]\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: base.yaml\nforecast:\n  horizon_steps: 12\ncovariates:\n  weather: [new]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_SIMCAST_DATA", "/tmp/simcast-data")

    config = load_config(child, ["forecast.origin_stride_steps=4", "chronos.device=cpu"])

    assert config.data.local_dir == Path("/tmp/simcast-data")
    assert config.forecast.lookback_steps == 100
    assert config.forecast.horizon_steps == 12
    assert config.forecast.origin_stride_steps == 4
    assert config.covariates.weather == ["new"]
    assert config.chronos.device == "cpu"


def test_multiple_parents_merge_left_to_right(tmp_path: Path) -> None:
    (tmp_path / "one.yaml").write_text("training:\n  epochs: 20\n  patience: 5\n", encoding="utf-8")
    (tmp_path / "two.yaml").write_text("training:\n  epochs: 10\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: [one.yaml, two.yaml]\ntraining:\n  patience: 3\n", encoding="utf-8")

    config = load_config(child)

    assert config.training.epochs == 10
    assert config.training.patience == 3


def test_cycle_detection(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inheritance cycle"):
        load_config(first)


def test_override_parser_and_deep_merge() -> None:
    overrides = parse_overrides(
        ["training.epochs=7", "covariates.weather=[temperature_2m]", "runtime.deterministic=false"]
    )

    assert overrides == {
        "training": {"epochs": 7},
        "covariates": {"weather": ["temperature_2m"]},
        "runtime": {"deterministic": False},
    }
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


@pytest.mark.parametrize(
    "values",
    [
        {"split": {"tune_fraction": 1.0}},
        {"data": {"entity_type": "mixed"}},
        {"forecast": {"lookback_steps": 0}},
        {"dependence": {"conditional_low_rank": {"latent_rank": 0}}},
        {"dependence": {"set_aware_low_rank": {"model_dim": 127, "num_heads": 4}}},
        {"evaluation": {"quantile_levels": [0.5, 0.1]}},
        {"chronos": {"cross_learning": True}},
    ],
)
def test_invalid_config_is_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SimcastConfig.model_validate(values)


def test_invalid_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="dotted.key=value"):
        parse_overrides(["training.epochs"])
