import json
from math import erf
from pathlib import Path

import numpy as np
import pytest
import torch

from simcast.cli.evaluate import PreparedMethod, _base_normal_draws, _sample_and_evaluate, evaluate_from_config
from simcast.cli.train_dependence import train_from_config
from simcast.config import SimcastConfig
from simcast.fm.cache import build_cache_dataset, save_pit_library


def _cache(path: Path) -> Path:
    rng = np.random.default_rng(8)
    n_origin, n_entity, horizon = 10, 4, 3
    truth = rng.normal(10.0, 1.0, size=(n_origin, n_entity, horizon)).astype(np.float32)
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


def _config(tmp_path: Path) -> SimcastConfig:
    return SimcastConfig.model_validate(
        {
            "protocol": {"name": "full_group", "full_group_only": True},
            "chronos": {"device": "cpu"},
            "sampling": {"num_samples": 32},
            "evaluation": {
                "quantile_levels": [0.1, 0.5, 0.9],
                "interval_levels": [0.8],
                "scenario_batch_size": 2,
                "joint_score_num_samples": 8,
            },
            "output": {"root_dir": tmp_path / "runs"},
        }
    )


def test_full_group_evaluation_uses_only_the_complete_group(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    config = _config(tmp_path)
    static_config = config.model_copy(
        update={"dependence": config.dependence.model_copy(update={"method": "static_gaussian"})}
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
    assert not (output / "variable_k.csv").exists()
    assert (output / "scientific_summary.json").is_file()
    assert (output / "evaluation_manifest.json").is_file()
    assert (output / "resolved_config.yaml").is_file()
    assert (output / "figures" / "summary_coverage.png").is_file()
    assert not (output / "figures" / "variable_cardinality.png").exists()
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["group"]["entity_count"] == 4
    assert manifest["experimental_protocol"] == {
        "name": "full_group",
        "full_group_only": True,
        "subset_training": False,
        "entity_selection_augmentation_enabled": False,
    }
    assert "variable_k_entity_ids" not in manifest


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
