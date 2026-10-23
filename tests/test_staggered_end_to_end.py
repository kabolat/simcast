"""End-to-end variable-cardinality regression for staggered entity lifetimes.

Canonical universe K_max = 5 on a 160-step timeline (origins on odd steps 3..157):

* e0 exists throughout;
* e1 starts at step 80, after the training period, so it has no training PIT;
* e2 ends at step 110, during the test period;
* e3 exists throughout but has target gaps (30-37 in train, 120-127 in test);
* e4 exists only on the central interval 60-100.

No real Chronos inference is run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from _staggered_fixture import (
    START,
    STEP,
    RecordingForecaster,
    expected_forecastable,
    inject_adapter,
    staggered_adapter,
    staggered_config,
    timestamp,
)

import simcast.cli.build_cache as cache_cli
import simcast.cli.train_dependence as train_cli
from simcast.cli.evaluate import PreparedMethod, _sample_and_evaluate, evaluate_from_config
from simcast.config import ResolvedExperimentConfig
from simcast.fm.cache import load_pit_library
from simcast.training.dataset import DependenceBatch, DependenceCollator
from simcast.training.losses import gaussian_copula_pseudo_nll

PERIODS = 160
LIFETIMES = ((0, 159), (80, 159), (0, 110), (0, 159), (60, 100))
GAPS = {3: ((30, 37), (120, 127))}
EMPTY_HISTORY_GAP_ORIGINS = (33, 35, 37, 123, 125, 127)
NEURAL_MODELS = {
    "conditional_low_rank": {"hidden_dims": [8], "latent_rank": 2, "dropout": 0.0},
    "set_aware_low_rank": {"model_dim": 8, "num_layers": 1, "num_heads": 2, "latent_rank": 2, "dropout": 0.0},
    "conditional_kernel": {"hidden_dims": [8], "embedding_dim": 3, "dropout": 0.0},
}
METHODS = ("independent", "static_gaussian", *NEURAL_MODELS)


def _config(root: Path, mode: str, *, dependence: dict[str, object] | None = None) -> ResolvedExperimentConfig:
    base = staggered_config(
        root,
        cardinality_mode=mode,
        entity_count=len(LIFETIMES),
        tune_fraction=0.6,
        validation_fraction_within_tune=0.25,
        pit={"monotone_repair": "none", "dependence_transform": "training_frequency"},
    )
    payload = base.model_dump(mode="json")
    payload["chronos"]["device"] = "cpu"
    payload["sampling"] = {"num_samples": 64}
    payload["evaluation"] = {
        "quantile_levels": [0.1, 0.5, 0.9],
        "interval_levels": [0.8],
        "scenario_batch_size": 4,
        "joint_score_num_samples": 16,
    }
    if dependence is not None:
        payload["dependence"] = dependence
    if dependence is not None and dependence["method"] in NEURAL_MODELS:
        payload["features"] = {"use_location": True}
        payload["training"] = {"batch_size": 8, "epochs": 2, "patience": 2}
    return ResolvedExperimentConfig.model_validate(payload)


def _origin_indices(values: np.ndarray) -> list[int]:
    return [int((pd.Timestamp(value, tz="UTC") - START) / STEP) for value in values]


def _build(monkeypatch: pytest.MonkeyPatch, root: Path, mode: str) -> tuple[Path, RecordingForecaster, list[str]]:
    config = _config(root, mode)
    adapter = staggered_adapter(config, LIFETIMES, periods=PERIODS, gaps=GAPS)
    inject_adapter(monkeypatch, adapter)
    forecaster = RecordingForecaster()
    destination = cache_cli.build_cache_from_config(config, forecaster=forecaster, output_dir=root / f"cache-{mode}")
    return destination, forecaster, adapter.group.entity_ids


@dataclass
class Pipeline:
    root: Path
    cache: Path
    forecaster: RecordingForecaster
    entity_ids: list[str]
    runs: dict[str, Path]
    batches: dict[str, list[DependenceBatch]]
    evaluation: Path


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pipeline]:
    root = tmp_path_factory.mktemp("staggered-e2e")
    batches: dict[str, list[DependenceBatch]] = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        cache, forecaster, entity_ids = _build(monkeypatch, root, "variable")
        runs = {
            "static_gaussian": train_cli.train_from_config(
                _config(root, "variable", dependence={"method": "static_gaussian", "model": {}}),
                cache_dir=cache,
                output_dir=root / "static_gaussian",
            )
        }
        for method, model in NEURAL_MODELS.items():
            recorded: list[DependenceBatch] = []
            batches[method] = recorded

            class RecordingCollator(DependenceCollator):
                def __call__(self, samples, _recorded=recorded):  # type: ignore[no-untyped-def]
                    batch = super().__call__(samples)
                    _recorded.append(batch)
                    return batch

            monkeypatch.setattr(train_cli, "DependenceCollator", RecordingCollator)
            runs[method] = train_cli.train_from_config(
                _config(root, "variable", dependence={"method": method, "model": model}),
                cache_dir=cache,
                output_dir=root / method,
            )
    evaluation = evaluate_from_config(
        _config(root, "variable"),
        methods=METHODS,
        method_runs=runs,
        cache_dir=cache,
        output_dir=root / "evaluation",
    )
    yield Pipeline(root, cache, forecaster, entity_ids, runs, batches, evaluation)


def _expected_forecastable(origins: list[int]) -> np.ndarray:
    expected = np.asarray([[expected_forecastable(origin, life) for life in LIFETIMES] for origin in origins])
    for origin in EMPTY_HISTORY_GAP_ORIGINS:
        if origin in origins:
            expected[origins.index(origin), 3] = False
    return expected


def test_union_coverage_entry_exit_and_canonical_indexing(pipeline: Pipeline) -> None:
    library = load_pit_library(pipeline.cache)
    protocol = library.metadata["forecast_protocol"]
    coverage = (protocol["coverage_start"], protocol["coverage_end"])
    assert coverage == (timestamp(0).isoformat(), timestamp(159).isoformat())
    origins = _origin_indices(library.dataset["origin_timestamp"].values)
    assert origins == list(range(3, 158, 2))
    assert library.dataset["entity_id"].values.tolist() == pipeline.entity_ids
    assert library.metadata["entity_group"]["entity_ids"] == pipeline.entity_ids

    forecastable = library.dataset["forecastable_entity"].values
    np.testing.assert_array_equal(forecastable, _expected_forecastable(origins))
    first = {k: origins[int(np.argmax(forecastable[:, k]))] for k in range(5)}
    last = {k: origins[len(origins) - 1 - int(np.argmax(forecastable[::-1, k]))] for k in range(5)}
    assert first == {0: 3, 1: 83, 2: 3, 3: 3, 4: 63}
    assert last == {0: 157, 1: 157, 2: 107, 3: 157, 4: 97}

    missingness = library.metadata["missingness"]
    assert missingness["history_rule"] == "at_least_one_finite_lookback_target"
    assert missingness["empty_history_non_forecastable_count_by_entity"]["transformer::e3"] == 6
    assert missingness["empty_history_rejected_origin_count"] == 0

    # Chronos sees only compact, canonically ordered, forecastable inputs with usable history.
    assert sum(len(call) for call in pipeline.forecaster.calls) == int(forecastable.sum())
    for call in pipeline.forecaster.calls:
        assert call == [entity for entity in pipeline.entity_ids if entity in call]
    assert min(min(counts) for counts in pipeline.forecaster.finite_history_counts) >= 1


def test_sealed_entity_valid_and_pit_retention(pipeline: Pipeline) -> None:
    training = load_pit_library(pipeline.cache)
    split = training.dataset["split"].values
    assert np.isnan(training.dataset["entity_valid"].values[split == "test"]).all()
    assert np.isnan(training.dataset["pit_u"].values[split == "test"]).all()
    with pytest.raises(PermissionError):
        training.test_data()

    full = load_pit_library(pipeline.cache, access="evaluation").dataset
    forecastable = full["forecastable_entity"].values
    entity_valid = full["entity_valid"].values.astype(bool)
    truth = full["true_y"].values
    pit_u = full["pit_u"].values
    np.testing.assert_array_equal(entity_valid, forecastable[:, :, None] & np.isfinite(truth))
    assert np.isfinite(pit_u[entity_valid]).all()
    assert np.isnan(pit_u[~entity_valid]).all()
    origins = _origin_indices(full["origin_timestamp"].values)
    # Forecastable entity with a missing future target: e3 at origin 119 (steps 120-121 are a gap).
    row = origins.index(119)
    assert forecastable[row, 3] and not entity_valid[row, 3].any()
    # The e3 train gap leaves partially observed histories forecastable.
    assert forecastable[origins.index(31), 3] and forecastable[origins.index(39), 3]


def test_training_frequency_fallback_is_exactly_the_late_entity(pipeline: Pipeline) -> None:
    expected = np.zeros((5, 2), dtype=bool)
    expected[1] = True
    for name, run in pipeline.runs.items():
        with np.load(run / "pit_training_frequency_map.npz") as saved:
            np.testing.assert_array_equal(saved["fallback_mask"], expected, err_msg=name)
            nominal = [0.025, 0.075, 0.175, 0.375, 0.625, 0.825, 0.925, 0.975]
            np.testing.assert_allclose(saved["midpoints"][1, 0], nominal)
        summary = json.loads((run / "run_metadata.json").read_text())["pit_training_frequency_fallback"]
        assert summary["fallback_count"] == 2
        assert summary["fallback_count_by_entity"]["transformer::e1"] == 2
    manifest = json.loads((pipeline.evaluation / "evaluation_manifest.json").read_text())
    assert manifest["pit_training_frequency_fallback"]["fallback_count_by_entity"] == {
        "transformer::e0": 0,
        "transformer::e1": 2,
        "transformer::e2": 0,
        "transformer::e3": 0,
        "transformer::e4": 0,
    }


def test_m1_pairwise_fit_keeps_entities_without_overlap_independent(pipeline: Pipeline) -> None:
    run = pipeline.runs["static_gaussian"]
    diagnostics = json.loads((run / "m1_diagnostics.json").read_text())
    assert diagnostics["estimator_mode"] == "pairwise_complete"
    overlaps = np.asarray(diagnostics["pairwise_overlap_counts"])
    assert overlaps.shape == (2, 5, 5)
    assert (overlaps[:, 1, :] == 0).all()
    assert (overlaps[:, 0, 4] > 0).all() and (overlaps[:, 0, 4] < overlaps[:, 0, 0]).all()
    with np.load(run / "model.npz") as saved:
        correlations = saved["correlations"]
    np.testing.assert_allclose(correlations[:, 1, [0, 2, 3, 4]], 0.0, atol=1e-6)
    assert (correlations[:, 0, 2] > 0.2).all()


def test_neural_training_batches_are_compact_active_sets(pipeline: Pipeline) -> None:
    for method, recorded in pipeline.batches.items():
        assert recorded, method
        counts = torch.cat([batch.active_counts for batch in recorded])
        assert counts.min() >= 2 and counts.max() <= 4
        assert set(counts.tolist()) >= {2, 3, 4}
        for batch in recorded:
            assert batch.features.shape[1] == int(batch.active_counts.max())
            for row in range(batch.z.shape[0]):
                indices = batch.canonical_entity_indices[row][batch.active_mask[row]]
                assert torch.equal(indices, indices.sort().values)
                assert 1 not in indices.tolist()
                assert (batch.canonical_entity_indices[row][~batch.active_mask[row]] == -1).all()


def test_evaluation_scores_active_subsets_and_reports_active_k(pipeline: Pipeline) -> None:
    output = pipeline.evaluation
    full = load_pit_library(pipeline.cache, access="evaluation")
    test = full.test_data()
    origins = _origin_indices(test["origin_timestamp"].values)
    forecastable = test["forecastable_entity"].values.astype(bool)
    expected_active = forecastable[:, :, None] & test["entity_valid"].values.astype(bool)
    expected_counts = expected_active.sum(axis=1)
    with np.load(output / "independent_aggregate_predictions.npz") as arrays:
        active = arrays["active_entity_mask"]
        counts = arrays["active_entity_count"]
        valid = arrays["valid"]
    np.testing.assert_array_equal(active, expected_active)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(valid, expected_counts >= 2)
    assert counts[origins.index(95)].tolist() == [5, 5]
    assert counts[origins.index(125)].tolist() == [2, 2]
    for method in METHODS:
        with np.load(output / f"{method}_aggregate_predictions.npz") as arrays:
            np.testing.assert_array_equal(arrays["active_entity_mask"], active)
            assert np.isfinite(arrays["quantile_predictions"][valid]).all()
            assert np.isfinite(arrays["pseudo_nll"][valid]).all()
            assert np.isnan(arrays["pseudo_nll"][~valid]).all()
            if method in NEURAL_MODELS:
                inactive_rows = arrays["correlations"][~active.transpose(0, 2, 1)]
                np.testing.assert_allclose(np.abs(inactive_rows).sum(axis=-1), 1.0, atol=1e-6)

    # Active-submatrix pseudo-NLL of M1 recomputed from the saved matrix and the train-only map.
    config = _config(pipeline.root, "variable")
    dependence_z, _, _ = train_cli._dependence_scores(full, config)
    test_rows = np.flatnonzero(full.dataset["split"].values == "test")
    with np.load(pipeline.runs["static_gaussian"] / "model.npz") as saved:
        static = torch.tensor(saved["correlations"], dtype=torch.float32)
    with np.load(output / "static_gaussian_aggregate_predictions.npz") as arrays:
        nll = arrays["pseudo_nll"]
    for origin_row, lead in zip(*np.nonzero(valid), strict=True):
        entities = np.flatnonzero(active[origin_row, :, lead])
        z = torch.tensor(dependence_z[test_rows[origin_row], entities, lead], dtype=torch.float32)
        expected = gaussian_copula_pseudo_nll(z, static[lead][np.ix_(entities, entities)], reduction="none")
        assert nll[origin_row, lead] == pytest.approx(float(expected), rel=1e-4, abs=1e-5)

    metrics = pd.read_parquet(output / "per_origin_lead_metrics.parquet")
    truth = test["true_y"].values
    for origin_row, lead in zip(*np.nonzero(valid), strict=True):
        case = metrics[
            (metrics["method"] == "static_gaussian")
            & (metrics["origin_index"] == test_rows[origin_row])
            & (metrics["lead"] == lead + 1)
        ]
        assert len(case) == 1
        assert case["active_entity_count"].iloc[0] == counts[origin_row, lead]
        assert case["observed_aggregate"].iloc[0] == pytest.approx(
            truth[origin_row, active[origin_row, :, lead], lead].sum(), rel=1e-5
        )
    assert (metrics["K"] == 5).all()

    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    assert manifest["cardinality_mode"] == "variable"
    assert manifest["group"]["entity_count"] == 5
    assert manifest["common_random_numbers"] is True
    assert (manifest["active_entity_count"]["min"], manifest["active_entity_count"]["max"]) == (2, 5)
    summary = json.loads((output / "metrics.json").read_text())
    for method in METHODS:
        assert summary[method]["min_active_entity_count"] == 2
        assert summary[method]["max_active_entity_count"] == 5


def test_common_random_numbers_are_shared_across_methods_and_active_sets(pipeline: Pipeline) -> None:
    full = load_pit_library(pipeline.cache, access="evaluation")
    test = full.test_data()
    truth = torch.tensor(test["true_y"].values, dtype=torch.float32)
    predictions = torch.tensor(test["quantile_prediction"].values, dtype=torch.float32)
    levels = torch.tensor(test["quantile"].values, dtype=torch.float32)
    with np.load(pipeline.evaluation / "independent_aggregate_predictions.npz") as arrays:
        active = torch.tensor(arrays["active_entity_mask"])
        valid = torch.tensor(arrays["valid"])
    n_origin, n_entity, horizon = truth.shape
    identity = torch.eye(n_entity).expand(n_origin, horizon, n_entity, n_entity)

    def run(name: str, crn: bool) -> torch.Tensor:
        payload = _config(pipeline.root, "variable").model_dump(mode="json")
        payload["sampling"]["common_random_numbers"] = crn
        config = ResolvedExperimentConfig.model_validate(payload)
        result = _sample_and_evaluate(
            PreparedMethod(name, identity),
            truth,
            predictions,
            levels,
            valid,
            config,
            active_entity_mask=active,
            metrics=("mean_pinball",),
        )
        return result.aggregate.quantile_predictions

    torch.testing.assert_close(run("method_a", True), run("method_b", True), equal_nan=True)
    assert not torch.allclose(run("method_a", False)[valid], run("method_b", False)[valid])


def test_fixed_mode_reduces_the_same_source_to_the_common_full_group(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        cache, forecaster, entity_ids = _build(monkeypatch, tmp_path, "fixed")
    library = load_pit_library(cache)
    protocol = library.metadata["forecast_protocol"]
    coverage = (protocol["coverage_start"], protocol["coverage_end"])
    assert coverage == (timestamp(80).isoformat(), timestamp(100).isoformat())
    origins = _origin_indices(library.dataset["origin_timestamp"].values)
    assert origins == list(range(83, 98, 2))
    assert library.dataset["forecastable_entity"].values.all()
    assert forecaster.calls and all(call == entity_ids for call in forecaster.calls)
    assert library.metadata["missingness"]["empty_history_non_forecastable_count"] == 0

    static = train_cli.train_from_config(
        _config(tmp_path, "fixed", dependence={"method": "static_gaussian", "model": {}}),
        cache_dir=cache,
        output_dir=tmp_path / "static",
    )
    assert json.loads((static / "m1_diagnostics.json").read_text())["estimator_mode"] == "ledoit_wolf"
    output = evaluate_from_config(
        _config(tmp_path, "fixed"),
        methods=("independent", "static_gaussian"),
        method_runs={"static_gaussian": static},
        cache_dir=cache,
        output_dir=tmp_path / "evaluation",
    )
    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    assert manifest["cardinality_mode"] == "fixed"
    assert (manifest["active_entity_count"]["min"], manifest["active_entity_count"]["max"]) == (5, 5)
    with np.load(output / "static_gaussian_aggregate_predictions.npz") as arrays:
        assert arrays["active_entity_mask"].all()
        assert arrays["valid"].all()
