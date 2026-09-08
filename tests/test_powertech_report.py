from __future__ import annotations

import json
from math import erf
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from simcast.config import load_config
from simcast.fm.cache import build_cache_dataset, save_pit_library
from simcast.reporting.powertech import build_powertech_report

CONFIGS = Path(__file__).parents[1] / "configs"


def _synthetic_cache(path: Path) -> Path:
    rng = np.random.default_rng(5)
    origins, entities, leads = 18, 4, 2
    z = rng.normal(size=(origins, entities, leads)).astype(np.float32)
    u = (0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))).astype(np.float32)
    truth = rng.normal(size=(origins, entities, leads)).astype(np.float32)
    quantiles = np.stack((truth - 1, truth, truth + 1), axis=-1)
    dataset = build_cache_dataset(
        origin_timestamps=np.arange(origins).astype("datetime64[D]"),
        entity_ids=[f"transformer::entity-{index}" for index in range(entities)],
        true_y=truth,
        quantile_predictions=quantiles,
        pit_u=u,
        pit_z=z,
        forecast_embeddings=rng.normal(size=(origins, entities, 1, 3)).astype(np.float16),
        quantile_levels=[0.1, 0.5, 0.9],
        split=["train"] * 12 + ["validation"] * 3 + ["test"] * 3,
    )
    dataset["pit_valid"] = (("origin", "lead"), np.ones((origins, leads), dtype=bool))
    return save_pit_library(
        path,
        dataset,
        {"pit": {"crossing_frequency": 0.0}},
    )


def _synthetic_evaluation(root: Path, cache: Path) -> None:
    evaluation = root / "main" / "seed_11" / "evaluation"
    evaluation.mkdir(parents=True)
    config = load_config(CONFIGS / "powertech2027" / "transformer.yaml").model_dump(mode="json")
    config["confirmatory"]["bootstrap_replicates"] = 50
    config["confirmatory"]["primary_block_length"] = 3
    config["confirmatory"]["sensitivity_block_lengths"] = [2]
    (evaluation / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    timestamps = pd.date_range("2024-01-01", periods=14, tz="UTC")
    method_scores = {
        "independent": 10.0,
        "static_gaussian": 9.0,
        "conditional_low_rank": 8.5,
        "set_aware_low_rank": 8.0,
    }
    case_rows: list[dict[str, object]] = []
    origin_rows: list[dict[str, object]] = []
    for method, score in method_scores.items():
        seed = 11 if method.startswith(("conditional", "set_aware")) else None
        for origin_index, timestamp in enumerate(timestamps):
            origin_rows.append(
                {
                    "group": "transformer",
                    "method": method,
                    "neural_seed": seed,
                    "origin_index": origin_index,
                    "origin": timestamp,
                    "K": 4,
                    "mean_pinball": score + origin_index / 100,
                    "crps": score * 2,
                    "weighted_interval_score": score * 3,
                    "coverage_0.9": 0.9,
                    "interval_width_0.9": score * 4,
                    "interval_score_0.9": score * 5,
                    "energy_score": score * 2,
                    "variogram_score": score,
                    "test_pseudo_nll": score / 10,
                    "valid_leads": 2,
                }
            )
            for lead in (1, 2):
                case_rows.append(
                    {
                        "group": "transformer",
                        "method": method,
                        "neural_seed": seed,
                        "origin_index": origin_index,
                        "origin": timestamp,
                        "lead": lead,
                        "K": 4,
                        "valid": True,
                        "observed_aggregate": 20.0 + origin_index,
                        "aggregate_q0.05": 18.0 + origin_index,
                        "aggregate_q0.5": 20.0 + origin_index,
                        "aggregate_q0.95": 22.0 + origin_index,
                        "mean_pinball": score + origin_index / 100,
                        "crps": score * 2,
                        "weighted_interval_score": score * 3,
                        "coverage_0.9": 0.9,
                        "interval_width_0.9": score * 4,
                        "interval_score_0.9": score * 5,
                        "energy_score": score * 2,
                        "variogram_score": score,
                        "test_pseudo_nll": score / 10,
                    }
                )
        correlations = np.eye(4)[None, None].repeat(14, axis=0).repeat(2, axis=1)
        np.savez_compressed(evaluation / f"{method}_aggregate_predictions.npz", correlations=correlations)
    pd.DataFrame(case_rows).to_parquet(evaluation / "per_origin_lead_metrics.parquet", index=False)
    pd.DataFrame(origin_rows).to_parquet(evaluation / "per_origin_metrics.parquet", index=False)
    manifest = {
        "cache_path": str(cache),
        "group": {"name": "transformer", "entity_count": 4, "entity_ids": [f"e{i}" for i in range(4)]},
        "experimental_protocol": {"name": "powertech2027", "full_group_only": True},
        "confirmatory": config["confirmatory"],
        "evaluation_seed": 2027,
        "git_commit": "deadbeef",
        "config_sha256": "abc",
    }
    (evaluation / "evaluation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_report_is_generated_from_saved_artifacts_without_training(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path / "cache")
    _synthetic_evaluation(tmp_path / "runs", cache)

    report = build_powertech_report(tmp_path / "runs", tmp_path / "report")

    assert (report / "results" / "per_origin_lead_metrics.parquet").is_file()
    assert (report / "results" / "method_seed_summary.csv").is_file()
    assert (report / "main_results.csv").is_file()
    assert (report / "main_results.tex").is_file()
    assert "\\textbf" in (report / "main_results.tex").read_text(encoding="utf-8")
    assert (report / "paired_effects.csv").is_file()
    assert (report / "figures" / "figure_A_method_pipeline.pdf").is_file()
    assert (report / "figures" / "figure_B_paired_effect_forest.svg").is_file()
    assert (report / "scientific_summary.md").is_file()
    assert (report / "protocol.json").is_file()
