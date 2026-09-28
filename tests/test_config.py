from pathlib import Path

import pytest
from pydantic import ValidationError

from simcast.config import (
    CHRONOS_MODEL_REVISION,
    CHRONOS_SOURCE_REVISION,
    LIANDER2024_REVISION,
    ConditionalKernelMethodConfig,
    ConditionalLowRankMethodConfig,
    IndependentMethodConfig,
    PitConfig,
    SetAwareLowRankMethodConfig,
    StaticGaussianMethodConfig,
    base_fingerprint,
    deep_merge,
    load_base_config,
    load_composite_config,
    load_evaluation_config,
    load_method_config,
    parse_overrides,
    resolve_run_config,
)

CONFIGS = Path(__file__).parents[1] / "configs"
BASES = CONFIGS / "bases" / "liander2024"
METHODS = CONFIGS / "methods"


def test_base_contains_only_the_common_scientific_design() -> None:
    base = load_base_config(BASES / "transformer.yaml")

    assert base.data.revision == LIANDER2024_REVISION
    assert base.chronos.source_revision == CHRONOS_SOURCE_REVISION
    assert base.chronos.model_revision == CHRONOS_MODEL_REVISION
    assert (base.forecast.lookback_steps, base.forecast.horizon_steps) == (672, 96)
    assert base.protocol.entity_count == len(base.protocol.ordered_entity_ids) == 15
    assert base.covariates.future_weather_source == "vintage"
    assert base.covariates.calendar.include_is_weekend
    assert not hasattr(base, "features")
    assert not hasattr(base, "training")
    assert not hasattr(base, "dependence")


def test_method_and_evaluation_ids_default_to_filename(tmp_path: Path) -> None:
    method_path = tmp_path / "filename_method.yaml"
    method_path.write_text("kind: method\nfamily: independent\n", encoding="utf-8")
    evaluation_path = tmp_path / "filename_evaluation.yaml"
    evaluation_path.write_text("kind: evaluation\n", encoding="utf-8")

    assert load_method_config(method_path).id == "filename_method"
    assert load_evaluation_config(evaluation_path).id == "filename_evaluation"


def test_evaluation_cross_entity_statistic_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "absolute_sum.yaml"
    for statistic in ("absolute_sum", "max", "absolute_max"):
        path.write_text(
            f"kind: evaluation\nevaluation: {{cross_entity_statistic: {statistic}}}\n", encoding="utf-8"
        )
        assert load_evaluation_config(path).evaluation.cross_entity_statistic == statistic

    path.write_text("kind: evaluation\nevaluation: {cross_entity_statistic: maximum}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="cross_entity_statistic"):
        load_evaluation_config(path)


def test_composite_name_defaults_to_filename(tmp_path: Path) -> None:
    venue_dir = tmp_path / "configs" / "venues" / "lab"
    venue_dir.mkdir(parents=True)
    path = venue_dir / "filename_composite.yaml"
    path.write_text(
        "kind: composite\nvenue: lab\n"
        "bases: [{id: base, config: base.yaml}]\n"
        "methods: [{id: m0, method: m0.yaml}]\n",
        encoding="utf-8",
    )

    assert load_composite_config(path).name == "filename_composite"


def test_linear_interpolation_is_a_valid_consistent_pit_mode() -> None:
    pit = PitConfig(mode="linear_interpolation")
    assert pit.mode == "linear_interpolation"
    with pytest.raises(ValidationError, match="training_frequency is defined only for discretized PIT cells"):
        PitConfig(mode="linear_interpolation", dependence_transform="training_frequency")


@pytest.mark.parametrize(
    ("filename", "expected_type", "allowed_fields"),
    [
        ("m0_independent.yaml", IndependentMethodConfig, {"kind", "id", "family"}),
        ("m1_static_gaussian.yaml", StaticGaussianMethodConfig, {"kind", "id", "family", "model"}),
        (
            "m2_conditional_low_rank.yaml",
            ConditionalLowRankMethodConfig,
            {"kind", "id", "family", "features", "model", "optimization"},
        ),
        (
            "m3_set_aware_low_rank.yaml",
            SetAwareLowRankMethodConfig,
            {"kind", "id", "family", "features", "model", "optimization"},
        ),
        (
            "m4_conditional_kernel.yaml",
            ConditionalKernelMethodConfig,
            {"kind", "id", "family", "features", "model", "optimization"},
        ),
    ],
)
def test_method_files_are_strictly_role_specific(
    filename: str, expected_type: type, allowed_fields: set[str]
) -> None:
    method = load_method_config(METHODS / filename)
    assert isinstance(method, expected_type)
    assert set(method.model_fields_set | {"kind"}) == allowed_fields


@pytest.mark.parametrize(
    ("filename", "entity_type", "entity_count", "repair"),
    [
        ("transformer.yaml", "transformer", 15, "isotonic"),
        ("solar_park.yaml", "solar_park", 5, "isotonic"),
        ("wind_park.yaml", "wind_park", 5, "none"),
        ("mv_feeder.yaml", "mv_feeder", 15, "none"),
        ("station_installation.yaml", "station_installation", 15, "none"),
    ],
)
def test_bases_predeclare_complete_ordered_groups(
    filename: str, entity_type: str, entity_count: int, repair: str
) -> None:
    base = load_base_config(BASES / filename)
    assert base.data.entity_type == entity_type
    assert base.protocol.entity_count == entity_count
    assert len(base.protocol.ordered_entity_ids) == entity_count
    assert all(value.startswith(f"{entity_type}::") for value in base.protocol.ordered_entity_ids)
    assert base.pit.monotone_repair == repair


def test_method_validation_rejects_cross_family_fields(tmp_path: Path) -> None:
    invalid = tmp_path / "m0.yaml"
    invalid.write_text("kind: method\nid: m0\nfamily: independent\nfeatures: {}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="features"):
        load_method_config(invalid)

    invalid.write_text(
        "kind: method\nid: m4\nfamily: conditional_kernel\nfeatures: {}\n"
        "model: {latent_rank: 4}\noptimization: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="latent_rank"):
        load_method_config(invalid)


def test_conditional_methods_require_features_model_and_optimization(tmp_path: Path) -> None:
    invalid = tmp_path / "m2.yaml"
    invalid.write_text("kind: method\nid: m2\nfamily: conditional_low_rank\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="features|model|optimization"):
        load_method_config(invalid)


def test_method_inheritance_and_cycle_detection(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text((METHODS / "m4_conditional_kernel.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    child.write_text("extends: parent.yaml\noptimization:\n  epochs: 3\n  patience: 2\n", encoding="utf-8")
    method = load_method_config(child)
    assert isinstance(method, ConditionalKernelMethodConfig)
    assert method.optimization.epochs == 3
    assert method.optimization.patience == 2

    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inheritance cycle"):
        load_method_config(first)


def test_multiple_method_parents_merge_deterministically_left_to_right(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    child = tmp_path / "child.yaml"
    first.write_text(
        (METHODS / "m4_conditional_kernel.yaml").read_text(encoding="utf-8").replace(
            "epochs: 100", "epochs: 20"
        ),
        encoding="utf-8",
    )
    second.write_text("optimization:\n  epochs: 10\n", encoding="utf-8")
    child.write_text(
        "extends: [first.yaml, second.yaml]\noptimization:\n  patience: 3\n",
        encoding="utf-8",
    )

    method = load_method_config(child)

    assert isinstance(method, ConditionalKernelMethodConfig)
    assert method.optimization.epochs == 10
    assert method.optimization.patience == 3


def test_composite_venue_must_match_its_directory(tmp_path: Path) -> None:
    venue_dir = tmp_path / "configs" / "venues" / "lab"
    venue_dir.mkdir(parents=True)
    path = venue_dir / "bad.yaml"
    path.write_text(
        "kind: composite\nname: x\nvenue: paper\n"
        "bases: [{id: x, config: x.yaml}]\n"
        "methods: [{id: m0, method: m0.yaml}]\n"
        "evaluations: [{id: standard, config: evaluation.yaml}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        load_composite_config(path)


def test_composite_requires_only_bases_and_methods(tmp_path: Path) -> None:
    venue_dir = tmp_path / "configs" / "venues" / "lab"
    venue_dir.mkdir(parents=True)
    path = venue_dir / "study.yaml"
    path.write_text(
        "kind: composite\nname: study\nvenue: lab\n"
        "bases: [{id: x, config: x.yaml}]\n"
        "methods: [{id: m0, method: m0.yaml}]\n",
        encoding="utf-8",
    )
    composite = load_composite_config(path)
    assert composite.evaluations == []
    assert composite.reports == []


def test_composite_reports_require_at_least_one_evaluation(tmp_path: Path) -> None:
    venue_dir = tmp_path / "configs" / "venues" / "lab"
    venue_dir.mkdir(parents=True)
    path = venue_dir / "study.yaml"
    path.write_text(
        "kind: composite\nname: study\nvenue: lab\n"
        "bases: [{id: x, config: x.yaml}]\n"
        "methods: [{id: m0, method: m0.yaml}]\n"
        "reports: [{id: standard_report, config: report.yaml, evaluation_ids: [standard]}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="reports require at least one evaluation"):
        load_composite_config(path)



def test_runtime_adapter_gives_m4_its_full_budget() -> None:
    base = load_base_config(BASES / "transformer.yaml")
    method = load_method_config(METHODS / "m4_conditional_kernel.yaml")
    runtime = resolve_run_config(base, method)
    assert runtime.training.epochs == 100
    assert runtime.training.patience == 12
    assert runtime.dependence.method == "conditional_kernel"
    assert runtime.dependence.model is not None
    assert set(runtime.dependence.model.model_dump()) == {
        "hidden_dims",
        "embedding_dim",
        "dropout",
        "initial_length_scale",
        "nugget",
        "jitter",
    }


def test_cache_fingerprint_ignores_runtime_but_not_marginal_design() -> None:
    base = load_base_config(BASES / "transformer.yaml")
    operational = base.model_copy(
        update={
            "runtime": base.runtime.model_copy(update={"log_level": "DEBUG"}),
            "chronos": base.chronos.model_copy(update={"device": "cpu", "batch_size": 1}),
        }
    )
    changed_pit = base.model_copy(update={"pit": base.pit.model_copy(update={"eps": 1.0e-6})})
    changed_mode = base.model_copy(
        update={"pit": base.pit.model_copy(update={"mode": "linear_interpolation"})}
    )
    changed_dependence_transform = base.model_copy(
        update={"pit": base.pit.model_copy(update={"dependence_transform": "nominal_cells"})}
    )
    assert base_fingerprint(base) == base_fingerprint(operational)
    assert base_fingerprint(base) == base_fingerprint(changed_dependence_transform)
    assert base_fingerprint(base) != base_fingerprint(changed_pit)
    assert base_fingerprint(base) != base_fingerprint(changed_mode)


def test_override_parser_and_deep_merge() -> None:
    overrides = parse_overrides(["optimization.epochs=7", "features.use_location=false"])
    assert overrides == {"optimization": {"epochs": 7}, "features": {"use_location": False}}
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_invalid_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="dotted.key=value"):
        parse_overrides(["optimization.epochs"])
