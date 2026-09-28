import json
from math import erf
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from simcast.cli.evaluate import PreparedMethod, _base_normal_draws, _sample_and_evaluate, evaluate_from_config
from simcast.cli.train_dependence import train_from_config
from simcast.config import DependenceConfig, ResolvedExperimentConfig, StaticGaussianConfig
from simcast.fm.cache import build_cache_dataset, load_pit_library, save_pit_library


def _cache(path: Path, *, signed: bool = False) -> Path:
    rng = np.random.default_rng(8)
    n_origin, n_entity, horizon = 10, 4, 3
    truth = rng.normal(10.0, 1.0, size=(n_origin, n_entity, horizon)).astype(np.float32)
    if signed:
        truth[:, 0] *= -1
    predictions = np.stack((truth - 1.0, truth, truth + 1.0), axis=-1)
    pit_z = rng.normal(size=(n_origin, n_entity, horizon)).astype(np.float32)
    pit_u = (0.5 * (1.0 + np.vectorize(erf)(pit_z / np.sqrt(2.0)))).astype(np.float32)
    dataset = build_cache_dataset(
        origin_timestamps=np.arange(n_origin).astype("datetime64[D]"),
        entity_ids=[f"transformer::entity-{index}" for index in range(n_entity)],
        true_y=truth,
        quantile_predictions=predictions,
        pit_u=pit_u,
        pit_z=pit_z,
        forecast_embeddings=rng.normal(size=(n_origin, n_entity, 2, 6)).astype(np.float16),
        quantile_levels=[0.1, 0.5, 0.9],
        split=["train"] * 6 + ["validation"] * 2 + ["test"] * 2,
        latitude=np.arange(n_entity),
        longitude=np.arange(n_entity) + 4,
    )
    dataset.attrs["output_patch_size"] = 2
    return save_pit_library(path, dataset, {"fixture": True})


def test_cross_entity_statistic_changes_samples_and_observations(tmp_path: Path) -> None:
    truth = torch.tensor([[[-2.0], [1.0]]])
    predictions = torch.stack((truth, truth, truth), dim=-1)
    prepared = PreparedMethod("independent", torch.eye(2).reshape(1, 1, 2, 2))
    valid = torch.ones((1, 1), dtype=torch.bool)
    config = _config(tmp_path)

    simple = _sample_and_evaluate(
        prepared, truth, predictions, torch.tensor([0.1, 0.5, 0.9]), valid, config, metrics=("mean_pinball",)
    )
    absolute_config = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"cross_entity_statistic": "absolute_sum"})}
    )
    absolute = _sample_and_evaluate(
        prepared,
        truth,
        predictions,
        torch.tensor([0.1, 0.5, 0.9]),
        valid,
        absolute_config,
        metrics=("mean_pinball",),
    )

    assert simple.aggregate.quantile_predictions[0, 0, 1].item() == pytest.approx(-1.0)
    assert absolute.aggregate.quantile_predictions[0, 0, 1].item() == pytest.approx(3.0)
    assert simple.aggregate.overall["mean_pinball"] == pytest.approx(0.0)
    assert absolute.aggregate.overall["mean_pinball"] == pytest.approx(0.0)
    for statistic, expected in (("max", 1.0), ("absolute_max", 2.0)):
        chosen = config.model_copy(
            update={"evaluation": config.evaluation.model_copy(update={"cross_entity_statistic": statistic})}
        )
        result = _sample_and_evaluate(
            prepared, truth, predictions, torch.tensor([0.1, 0.5, 0.9]), valid, chosen, metrics=("mean_pinball",)
        )
        assert result.aggregate.quantile_predictions[0, 0, 1].item() == pytest.approx(expected)
        assert result.aggregate.overall["mean_pinball"] == pytest.approx(0.0)


@pytest.mark.parametrize("statistic", ["absolute_sum", "max", "absolute_max"])
def test_statistic_is_retained_in_evaluation_artifacts(tmp_path: Path, statistic: str) -> None:
    cache = _cache(tmp_path / "cache", signed=True)
    config = _config(tmp_path)
    config = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"cross_entity_statistic": statistic})}
    )

    output = evaluate_from_config(
        config,
        methods=("independent",),
        cache_dir=cache,
        output_dir=tmp_path / "evaluation",
    )

    truth = load_pit_library(cache, access="evaluation").test_data()["true_y"].values
    selected = np.abs(truth) if statistic in {"absolute_sum", "absolute_max"} else truth
    expected = (selected.max(axis=1) if statistic in {"max", "absolute_max"} else selected.sum(axis=1)).reshape(-1)
    cases = pd.read_parquet(output / "per_origin_lead_metrics.parquet")
    assert np.allclose(cases["observed_aggregate"], expected)
    assert not np.allclose(expected, truth.sum(axis=1).reshape(-1))
    assert json.loads((output / "evaluation_manifest.json").read_text())["cross_entity_statistic"] == statistic


def _config(tmp_path: Path) -> ResolvedExperimentConfig:
    return ResolvedExperimentConfig.model_validate(
        {
            "chronos": {"device": "cpu"},
            "sampling": {"num_samples": 32},
            "evaluation": {
                "quantile_levels": [0.1, 0.5, 0.9],
                "interval_levels": [0.8],
                "scenario_batch_size": 2,
                "joint_score_num_samples": 8,
            },
        }
    )


def test_full_group_evaluation_uses_only_the_complete_group(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    config = _config(tmp_path)
    static_config = config.model_copy(
        update={"dependence": DependenceConfig(method="static_gaussian", model=StaticGaussianConfig())}
    )
    static_run = train_from_config(static_config, cache_dir=cache, output_dir=tmp_path / "static")

    output = evaluate_from_config(
        config,
        methods=("independent", "static_gaussian"),
        method_runs={"static_gaussian": static_run},
        cache_dir=cache,
        output_dir=tmp_path / "evaluation",
    )

    assert (output / "metrics.json").is_file()
    assert (output / "metrics_by_lead.csv").is_file()
    assert (output / "per_origin_lead_metrics.parquet").is_file()
    assert (output / "per_origin_metrics.parquet").is_file()
    assert (output / "evaluation_manifest.json").is_file()
    assert (output / "resolved_config.yaml").is_file()
    assert (output / "figures" / "dataset_locations.png").is_file()
    assert (output / "figures" / "marginal_pit.png").is_file()
    assert (output / "figures" / "independent" / "summary_by_lead_mean_pinball.png").is_file()
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["group"]["entity_count"] == 4


def test_evaluation_persists_only_declared_metrics(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    output = evaluate_from_config(
        _config(tmp_path),
        methods=("independent",),
        metrics=("mean_pinball",),
        cache_dir=cache,
        output_dir=tmp_path / "evaluation",
    )

    summary = json.loads((output / "metrics.json").read_text(encoding="utf-8"))["independent"]
    assert "mean_pinball" in summary
    assert "crps" not in summary
    assert "weighted_interval_score" not in summary
    assert "energy_score" not in summary
    assert "variogram_score" not in summary
    assert "test_pseudo_nll" not in summary
    case_columns = set(pd.read_parquet(output / "per_origin_metrics.parquet").columns)
    assert {"mean_pinball", "pinball_q0.1", "pinball_q0.5", "pinball_q0.9"} <= case_columns
    assert not {"crps", "weighted_interval_score", "energy_score", "variogram_score", "test_pseudo_nll"} & case_columns
    with np.load(output / "independent_aggregate_predictions.npz") as arrays:
        assert set(arrays.files) == {"quantile_predictions", "correlations", "valid"}


def test_evaluation_rejects_a_shrunken_correlation_matrix(tmp_path: Path) -> None:
    config = _config(tmp_path)
    truth = torch.ones(2, 4, 3)
    predictions = torch.stack((truth - 1, truth, truth + 1), dim=-1)
    prepared = PreparedMethod("independent", torch.eye(3).expand(2, 3, 3, 3))

    with pytest.raises(ValueError, match="shape"):
        _sample_and_evaluate(
            prepared,
            truth,
            predictions,
            torch.tensor([0.1, 0.5, 0.9]),
            torch.ones(2, 3, dtype=torch.bool),
            config,
        )


def test_common_random_normals_are_keyed_by_evaluation_seed_and_case() -> None:
    positions = torch.tensor([2, 9])
    first = _base_normal_draws(positions, 16, 4, seed=2027, device=torch.device("cpu"))
    second = _base_normal_draws(positions, 16, 4, seed=2027, device=torch.device("cpu"))
    reversed_draws = _base_normal_draws(positions.flip(0), 16, 4, seed=2027, device=torch.device("cpu"))

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, reversed_draws.flip(0))
    assert not torch.equal(first[0], first[1])
