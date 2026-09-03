from math import erf
from pathlib import Path

import numpy as np

from simcast.cli.evaluate import evaluate_from_config
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
            "chronos": {"device": "cpu"},
            "sampling": {"num_samples": 32},
            "evaluation": {
                "quantile_levels": [0.1, 0.5, 0.9],
                "interval_levels": [0.8],
                "scenario_batch_size": 2,
                "joint_score_num_samples": 8,
                "variable_k_sizes": [3, 4],
            },
            "output": {"root_dir": tmp_path / "runs"},
        }
    )


def test_final_evaluation_writes_tables_figures_and_summary(tmp_path: Path) -> None:
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
    assert (output / "variable_k.csv").is_file()
    assert (output / "scientific_summary.json").is_file()
    assert (output / "evaluation_manifest.json").is_file()
    assert (output / "resolved_config.yaml").is_file()
    assert (output / "figures" / "summary_coverage.png").is_file()
    assert (output / "figures" / "variable_cardinality.png").is_file()
    variable_rows = (output / "variable_k.csv").read_text(encoding="utf-8").splitlines()
    assert len(variable_rows) == 1 + 2 * 2
