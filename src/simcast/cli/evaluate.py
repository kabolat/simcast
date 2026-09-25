"""Final test evaluation for fixed-marginal spatial dependence methods."""

from __future__ import annotations

import json
import logging
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import torch
import yaml  # type: ignore[import-untyped]

from simcast.cli.train_dependence import _cache_path, _dependence_scores
from simcast.config import EvaluationFiguresConfig, ResolvedExperimentConfig
from simcast.dependence import IndependentCopula, StaticGaussianCopula
from simcast.evaluation.aggregate import AggregateEvaluation, evaluate_aggregate_ensemble
from simcast.evaluation.metrics import energy_score, variogram_score
from simcast.evaluation.plots import (
    plot_aggregate_fan,
    plot_correlation_heatmap,
    plot_dependence_dynamics,
    plot_eigenvalue_spectrum,
    plot_entity_load_traces,
    plot_entity_locations,
    plot_factor_parameters,
    plot_missingness,
    plot_pinball_by_quantile,
    plot_pit_histogram,
    plot_quantile_coverage,
    plot_score_by_lead,
)
from simcast.fm.cache import PITLibrary, load_pit_library
from simcast.reproducibility import config_sha256, git_commit, sha256_file, utc_run_id
from simcast.sampling.gaussian_copula import generate_scenarios
from simcast.training.checkpoint import LoadedConditionalModel, load_conditional_checkpoint
from simcast.training.losses import gaussian_copula_pseudo_nll

LOGGER = logging.getLogger(__name__)
CORE_METHODS = (
    "independent",
    "static_gaussian",
    "conditional_low_rank",
    "set_aware_low_rank",
    "conditional_kernel",
)


class _FactorModel(Protocol):
    def factor_parameters(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class PreparedMethod:
    name: str
    correlations: torch.Tensor
    conditional: LoadedConditionalModel | None = None
    features: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MethodEvaluation:
    name: str
    aggregate: AggregateEvaluation
    energy_score: torch.Tensor
    variogram_score: torch.Tensor
    pseudo_nll: torch.Tensor
    correlations: torch.Tensor


def _latest_run(root: Path, method: str) -> Path:
    candidates = sorted(path for path in root.glob(f"*_{method}") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no trained run matching '*_{method}' under {root}")
    return candidates[-1]


def _resolve_runs(
    config: ResolvedExperimentConfig,
    methods: Sequence[str],
    supplied: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    provided = {} if supplied is None else {name: Path(path).expanduser().resolve() for name, path in supplied.items()}
    root = Path(config.output.root_dir).expanduser().resolve()
    for method in methods:
        if method != "independent" and method not in provided:
            provided[method] = _latest_run(root, method)
    return provided


def _evaluation_directory(config: ResolvedExperimentConfig, override: str | Path | None) -> Path:
    if override is None:
        stamp = utc_run_id()
        path = (Path(config.output.root_dir).expanduser() / f"{stamp}_evaluation").resolve()
    else:
        path = Path(override).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    (path / "figures").mkdir()
    return path


def _entity_ids(library: PITLibrary, indices: Sequence[int] | None = None) -> list[str]:
    ids = [str(value) for value in library.dataset["entity_id"].values]
    return ids if indices is None else [ids[index] for index in indices]


def _locations(library: PITLibrary, indices: Sequence[int]) -> torch.Tensor | None:
    dataset = library.dataset
    if "latitude" not in dataset or "longitude" not in dataset:
        return None
    values = np.stack((dataset["latitude"].values, dataset["longitude"].values), axis=-1)
    return torch.tensor(values[np.asarray(indices)], dtype=torch.float32)


def _test_arrays(
    library: PITLibrary,
    entity_indices: Sequence[int],
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = library.test_data().isel(entity=list(entity_indices))
    truth = torch.tensor(dataset["true_y"].values, dtype=torch.float32)
    predictions = torch.tensor(dataset["quantile_prediction"].values, dtype=torch.float32)
    embeddings = torch.tensor(dataset["forecast_embedding"].values, dtype=torch.float32)
    split_origin_indices = np.asarray(dataset["origin"].values, dtype=np.int64)
    return split_origin_indices, truth, predictions, embeddings


def _prepare_method(
    name: str,
    run_paths: Mapping[str, Path],
    library: PITLibrary,
    config: ResolvedExperimentConfig,
    entity_indices: Sequence[int],
) -> PreparedMethod:
    _, _, predictions, embeddings = _test_arrays(library, entity_indices)
    n_origin, n_entity, horizon, _ = predictions.shape
    ids = _entity_ids(library, entity_indices)
    if name == "independent":
        matrix = IndependentCopula(ids, n_leads=horizon).correlation_matrix(1).to(torch.float32)
        return PreparedMethod(name, matrix.expand(n_origin, horizon, n_entity, n_entity).clone())
    if name == "static_gaussian":
        model = StaticGaussianCopula.load(run_paths[name] / "model.npz")
        if config.protocol.full_group_only and tuple(ids) != model.entity_ids:
            raise ValueError("full-group evaluation requires the exact ordered M1 entity set")
        matrices = torch.stack([model.correlation_matrix(lead, entity_ids=ids) for lead in range(1, horizon + 1)])
        return PreparedMethod(name, matrices[None].expand(n_origin, -1, -1, -1).to(torch.float32).clone())
    checkpoint = load_conditional_checkpoint(run_paths[name] / "best.pt", device="cpu")
    if checkpoint.method != name:
        raise ValueError(f"{name} run contains a {checkpoint.method} checkpoint")
    if config.protocol.full_group_only and tuple(ids) != checkpoint.entity_ids:
        raise ValueError(f"full-group evaluation requires the exact ordered {name} checkpoint entity set")
    if not set(ids).issubset(checkpoint.entity_ids):
        raise ValueError(f"{name} checkpoint entity IDs do not cover the evaluation group")
    levels = torch.tensor(library.dataset["quantile"].values, dtype=torch.float32)
    features = checkpoint.feature_builder.transform(
        embeddings, predictions, levels, _locations(library, entity_indices)
    )
    flat = features.permute(0, 2, 1, 3).reshape(-1, n_entity, features.shape[-1])
    device = torch.device(config.chronos.device if torch.cuda.is_available() else "cpu")
    checkpoint.model.to(device).eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], 256):
            correlation = checkpoint.model(flat[start : start + 256].to(device))
            if not isinstance(correlation, torch.Tensor):
                raise TypeError("conditional model did not return tensor correlations")
            chunks.append(correlation.to(device="cpu", dtype=torch.float32))
    checkpoint.model.to("cpu")
    matrices = torch.cat(chunks).reshape(n_origin, horizon, n_entity, n_entity)
    return PreparedMethod(name, matrices, conditional=checkpoint, features=features)


def _valid_pairs(truth: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(truth).all(dim=1) & torch.isfinite(predictions).all(dim=(1, 3))
    crossing = (predictions[..., :-1] > predictions[..., 1:]).any(dim=(1, 3))
    return finite & ~crossing


def _base_normal_draws(
    positions: torch.Tensor,
    num_samples: int,
    num_entities: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate draws keyed only by evaluation seed and flattened case index."""

    batches: list[torch.Tensor] = []
    for position in positions.tolist():
        generator = torch.Generator(device=device).manual_seed(seed + int(position))
        batches.append(torch.randn((num_samples, num_entities), generator=generator, device=device))
    return torch.stack(batches)


def _sample_and_evaluate(
    prepared: PreparedMethod,
    truth: torch.Tensor,
    predictions: torch.Tensor,
    levels: torch.Tensor,
    valid: torch.Tensor,
    config: ResolvedExperimentConfig,
    *,
    dependence_z: torch.Tensor | None = None,
    num_samples: int | None = None,
    compute_joint: bool = True,
) -> MethodEvaluation:
    sample_count = config.sampling.num_samples if num_samples is None else num_samples
    n_origin, n_entity, horizon = truth.shape
    expected_shape = (n_origin, horizon, n_entity, n_entity)
    if prepared.correlations.shape != expected_shape:
        raise ValueError(
            f"full group requires correlations with shape {expected_shape}, got {tuple(prepared.correlations.shape)}"
        )
    aggregate = torch.full((n_origin * horizon, sample_count), torch.nan)
    energy = torch.full((n_origin * horizon,), torch.nan)
    variogram = torch.full((n_origin * horizon,), torch.nan)
    pseudo_nll = torch.full((n_origin * horizon,), torch.nan)
    correlations = prepared.correlations.reshape(-1, n_entity, n_entity)
    marginal = predictions.permute(0, 2, 1, 3).reshape(-1, n_entity, predictions.shape[-1])
    realized = truth.permute(0, 2, 1).reshape(-1, n_entity)
    flattened_z = None if dependence_z is None else dependence_z.permute(0, 2, 1).reshape(-1, n_entity)
    case_indices = valid.reshape(-1).nonzero(as_tuple=False).flatten()
    device = torch.device(config.chronos.device if torch.cuda.is_available() else "cpu")
    for offset in range(0, case_indices.numel(), config.evaluation.scenario_batch_size):
        positions = case_indices[offset : offset + config.evaluation.scenario_batch_size]
        evaluation_seed = config.sampling.evaluation_seed
        if not config.sampling.common_random_numbers:
            evaluation_seed += zlib.crc32(prepared.name.encode())
        base_normals = _base_normal_draws(
            positions,
            sample_count,
            n_entity,
            seed=evaluation_seed,
            device=device,
        )
        scenarios = generate_scenarios(
            correlations.index_select(0, positions).to(device),
            marginal.index_select(0, positions).to(device),
            levels.to(device),
            num_samples=sample_count,
            base_normals=base_normals,
            marginal_mode=config.pit.mode,
        )
        aggregate[positions] = scenarios.aggregate_samples.cpu()
        if compute_joint:
            joint_count = min(sample_count, config.evaluation.joint_score_num_samples)
            joint_samples = scenarios.entity_samples[:, :joint_count]
            joint_truth = realized.index_select(0, positions).to(device)
            energy[positions] = energy_score(joint_samples, joint_truth, pair_chunk_size=128).cpu()
            variogram[positions] = variogram_score(
                joint_samples, joint_truth, power=config.evaluation.variogram_power
            ).cpu()
        if flattened_z is not None:
            pseudo_nll[positions] = gaussian_copula_pseudo_nll(
                flattened_z.index_select(0, positions).to(device),
                correlations.index_select(0, positions).to(device),
                reduction="none",
            ).cpu()
    aggregate = aggregate.reshape(n_origin, horizon, sample_count)
    energy = energy.reshape(n_origin, horizon)
    variogram = variogram.reshape(n_origin, horizon)
    pseudo_nll = pseudo_nll.reshape(n_origin, horizon)
    aggregate_truth = truth.sum(dim=1)
    report = evaluate_aggregate_ensemble(
        aggregate,
        aggregate_truth,
        quantile_levels=torch.tensor(config.evaluation.quantile_levels),
        interval_coverages=tuple(config.evaluation.interval_levels),
        valid_mask=valid,
    )
    return MethodEvaluation(prepared.name, report, energy, variogram, pseudo_nll, prepared.correlations)


def _serializable_metrics(result: MethodEvaluation, valid: torch.Tensor) -> dict[str, float | int]:
    metrics: dict[str, float | int] = dict(result.aggregate.overall)
    metrics["valid_origin_lead_count"] = int(valid.sum())
    metrics["dropped_origin_lead_count"] = int(valid.numel() - valid.sum())
    if torch.isfinite(result.energy_score).any():
        metrics["energy_score"] = float(result.energy_score[torch.isfinite(result.energy_score)].mean())
        metrics["variogram_score"] = float(result.variogram_score[torch.isfinite(result.variogram_score)].mean())
    if torch.isfinite(result.pseudo_nll).any():
        metrics["test_pseudo_nll"] = float(result.pseudo_nll[torch.isfinite(result.pseudo_nll)].mean())
    return metrics


def _save_method_result(run_dir: Path, result: MethodEvaluation, valid: torch.Tensor) -> dict[str, float | int]:
    metrics = _serializable_metrics(result, valid)
    np.savez_compressed(
        run_dir / f"{result.name}_aggregate_predictions.npz",
        quantile_predictions=result.aggregate.quantile_predictions.numpy(),
        correlations=result.correlations.numpy(),
        energy_score=result.energy_score.numpy(),
        variogram_score=result.variogram_score.numpy(),
        pseudo_nll=result.pseudo_nll.numpy(),
        valid=valid.numpy(),
    )
    return metrics


def _lead_table(result: MethodEvaluation) -> pd.DataFrame:
    table = result.aggregate.by_lead.copy()
    for name, values in (
        ("energy_score", result.energy_score),
        ("variogram_score", result.variogram_score),
        ("test_pseudo_nll", result.pseudo_nll),
    ):
        table[name] = [
            float(column[torch.isfinite(column)].mean()) if torch.isfinite(column).any() else np.nan
            for lead in table.index
            for column in (values[:, int(lead) - 1],)
        ]
    table.insert(0, "method", result.name)
    return table.reset_index()


def _method_seed(name: str, run_paths: Mapping[str, Path]) -> int | None:
    if name not in {"conditional_low_rank", "set_aware_low_rank", "conditional_kernel"}:
        return None
    metadata = json.loads((run_paths[name] / "run_metadata.json").read_text(encoding="utf-8"))
    return int(metadata["seed"])


def _case_tables(
    config: ResolvedExperimentConfig,
    library: PITLibrary,
    results: Mapping[str, MethodEvaluation],
    run_paths: Mapping[str, Path],
    valid: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = library.test_data()
    origins = pd.to_datetime(test["origin_timestamp"].values, utc=True)
    origin_indices = np.asarray(test["origin"].values, dtype=np.int64)
    observed_aggregate = np.asarray(test["true_y"].values).sum(axis=1)
    n_origin, horizon = valid.shape
    rows: list[pd.DataFrame] = []
    for name, result in results.items():
        table = pd.DataFrame(
            {
                "group": config.data.entity_type,
                "method": name,
                "neural_seed": _method_seed(name, run_paths),
                "origin_index": np.repeat(origin_indices, horizon),
                "origin": np.repeat(origins, horizon),
                "lead": np.tile(np.arange(1, horizon + 1), n_origin),
                "K": result.correlations.shape[-1],
                "valid": valid.numpy().reshape(-1),
                "observed_aggregate": observed_aggregate.reshape(-1),
            }
        )
        for quantile_index, level in enumerate(config.evaluation.quantile_levels):
            values = result.aggregate.quantile_predictions[..., quantile_index].numpy().reshape(-1)
            table[f"aggregate_q{level:g}"] = values
        metrics = {
            **result.aggregate.case_metrics,
            "energy_score": result.energy_score,
            "variogram_score": result.variogram_score,
            "test_pseudo_nll": result.pseudo_nll,
        }
        for metric, metric_values in metrics.items():
            array = metric_values.detach().cpu().numpy().reshape(-1)
            table[metric] = np.where(table["valid"], array, np.nan)
        rows.append(table)
    per_case = pd.concat(rows, ignore_index=True)
    metric_columns = [
        column
        for column in per_case.columns
        if column
        not in {
            "group",
            "method",
            "neural_seed",
            "origin_index",
            "origin",
            "lead",
            "K",
            "valid",
            "observed_aggregate",
            *(f"aggregate_q{level:g}" for level in config.evaluation.quantile_levels),
        }
    ]
    valid_rows = per_case[per_case["valid"]]
    per_origin = (
        valid_rows.groupby(
            ["group", "method", "neural_seed", "origin_index", "origin", "K"],
            dropna=False,
            sort=False,
        )[metric_columns]
        .mean()
        .reset_index()
    )
    valid_counts = (
        valid_rows.groupby(
            ["group", "method", "neural_seed", "origin_index", "origin", "K"],
            dropna=False,
            sort=False,
        )
        .size()
        .rename("valid_leads")
        .reset_index()
    )
    return per_case, per_origin.merge(
        valid_counts,
        on=["group", "method", "neural_seed", "origin_index", "origin", "K"],
        validate="one_to_one",
    )


def _training_correlations(library: PITLibrary) -> tuple[np.ndarray, np.ndarray]:
    dataset = library.dataset
    train = dataset.where(dataset["split"] == "train", drop=True)["pit_z"].values
    lead = train[:, :, 0]
    lead = lead[np.isfinite(lead).all(axis=1)]
    if lead.shape[0] < 2:
        raise ValueError("at least two complete training PIT vectors are required for diagnostics")
    empirical = np.corrcoef(lead, rowvar=False)
    empirical = np.nan_to_num(empirical, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(empirical, 1.0)
    off_diagonal = empirical[~np.eye(empirical.shape[0], dtype=bool)]
    return empirical, off_diagonal


def _plots(
    output: Path,
    library: PITLibrary,
    prepared: Mapping[str, PreparedMethod],
    results: Mapping[str, MethodEvaluation],
    truth: torch.Tensor,
    predictions: torch.Tensor,
    valid: torch.Tensor,
    config: ResolvedExperimentConfig,
    *,
    metrics: Sequence[str],
    figures: EvaluationFiguresConfig,
    base_figures_dir: Path,
) -> None:
    method_figures = output / "figures"
    base_figures_dir.mkdir(parents=True, exist_ok=True)
    ids = _entity_ids(library)
    dataset = library.dataset
    latitude = dataset["latitude"].values if "latitude" in dataset else np.arange(len(ids))
    longitude = dataset["longitude"].values if "longitude" in dataset else np.zeros(len(ids))
    plot_entity_locations(latitude, longitude, ids, base_figures_dir / "dataset_locations.png")
    plot_entity_load_traces(dataset["true_y"].values, ids, base_figures_dir / "dataset_load_traces.png")
    plot_missingness(dataset["true_y"].values, ids, base_figures_dir / "dataset_missingness.png")
    tune = dataset.where(dataset["split"] != "test", drop=True)
    plot_pit_histogram(
        tune["pit_u"].values,
        base_figures_dir / "marginal_pit.png",
        quantile_levels=dataset["quantile"].values,
        mode=config.pit.mode,
    )
    plot_quantile_coverage(
        tune["true_y"].values,
        tune["quantile_prediction"].values,
        dataset["quantile"].values,
        base_figures_dir / "marginal_quantile_coverage.png",
    )
    plot_pinball_by_quantile(
        tune["true_y"].values,
        tune["quantile_prediction"].values,
        dataset["quantile"].values,
        base_figures_dir / "marginal_pinball.png",
    )
    empirical, _ = _training_correlations(library)
    if "static_gaussian" in prepared:
        static = prepared["static_gaussian"].correlations[0, 0].numpy()
        plot_correlation_heatmap(
            empirical, ids, base_figures_dir / "pit_empirical_correlation.png", title="Training PIT correlation"
        )
        plot_correlation_heatmap(
            static, ids, base_figures_dir / "static_correlation.png", title="Ledoit-Wolf PIT correlation"
        )
        plot_eigenvalue_spectrum(
            {"Empirical": empirical, "Ledoit-Wolf": static}, base_figures_dir / "static_eigenvalues.png"
        )

    test_origins = pd.to_datetime(library.test_data()["origin_timestamp"].values, utc=True)
    aggregate_origin = _figure_origin_index(test_origins, figures.aggregate_origin)
    correlation_origin = _figure_origin_index(test_origins, figures.correlation_origin)
    correlation_lead = figures.correlation_lead - 1
    if correlation_lead >= truth.shape[-1]:
        raise ValueError("figures.correlation_lead exceeds the configured forecast horizon")
    if valid[aggregate_origin].any():
        aggregate_truth = truth[aggregate_origin].sum(dim=0).numpy()
        for _name, result in results.items():
            method_figures = output / "figures" if len(results) == 1 else output / "figures" / _name
            method_figures.mkdir(parents=True, exist_ok=True)
            plot_aggregate_fan(
                result.aggregate.quantile_predictions[aggregate_origin].numpy(),
                config.evaluation.quantile_levels,
                aggregate_truth,
                method_figures / "aggregate_fan.png",
                title="Aggregate forecast",
            )
    if valid[correlation_origin, correlation_lead]:
        for name, result in results.items():
            method_figures = output / "figures" if len(results) == 1 else output / "figures" / name
            method_figures.mkdir(parents=True, exist_ok=True)
            plot_correlation_heatmap(
                result.correlations[correlation_origin, correlation_lead].numpy(),
                ids,
                method_figures / "correlation.png",
                title=f"Origin {correlation_origin}, lead {correlation_lead + 1}",
            )
    if "static_gaussian" in prepared:
        static_by_origin = prepared["static_gaussian"].correlations[:, 0]
        for name in ("conditional_low_rank", "set_aware_low_rank"):
            item = prepared.get(name)
            if item is None:
                continue
            method_figures = output / "figures" if len(results) == 1 else output / "figures" / name
            method_figures.mkdir(parents=True, exist_ok=True)
            plot_dependence_dynamics(
                item.correlations[:, 0].numpy(),
                static_by_origin.numpy(),
                method_figures / f"dependence_dynamics_{name}.png",
            )
            if item.conditional is not None and item.features is not None:
                with torch.no_grad():
                    loadings, sigma = cast(_FactorModel, item.conditional.model).factor_parameters(
                        item.features[0, :, 0]
                    )
                plot_factor_parameters(loadings.numpy(), sigma.numpy(), ids, method_figures / f"factors_{name}.png")
    for metric in metrics:
        for name, result in results.items():
            method_figures = output / "figures" if len(results) == 1 else output / "figures" / name
            method_figures.mkdir(parents=True, exist_ok=True)
            if metric in result.aggregate.by_lead:
                values = result.aggregate.by_lead[metric].tolist()
            else:
                tensors = {
                    "energy_score": "energy_score",
                    "variogram_score": "variogram_score",
                    "test_pseudo_nll": "pseudo_nll",
                }
                attribute = tensors.get(metric)
                if attribute is None:
                    continue
                values = [
                    float(column[torch.isfinite(column)].mean())
                    for column in getattr(result, attribute).T
                ]
            plot_score_by_lead(
                {name: values}, method_figures / f"summary_by_lead_{metric}.png", metric=metric
            )


def _figure_origin_index(origins: pd.DatetimeIndex, requested: datetime | None) -> int:
    if requested is None:
        return 0
    target = pd.Timestamp(requested)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    matches = np.flatnonzero(origins == target)
    if not len(matches):
        raise ValueError(f"figure origin {target.isoformat()} is not a testing origin")
    return int(matches[0])


def _scientific_summary(
    metrics: Mapping[str, Mapping[str, float | int]],
    mean_abs_pit_correlation: float,
) -> dict[str, str]:
    def delta(left: str, right: str) -> str:
        if left not in metrics or right not in metrics:
            return "not evaluated"
        change = float(metrics[right]["mean_pinball"]) - float(metrics[left]["mean_pinball"])
        return f"mean pinball change ({right} - {left}) = {change:.6g}"

    widest_coverage = max(
        (
            float(key.removeprefix("coverage_"))
            for key in metrics.get("independent", {})
            if key.startswith("coverage_")
        ),
        default=None,
    )
    if widest_coverage is None:
        independent_calibration = "not evaluated"
    else:
        observed = float(metrics["independent"][f"coverage_{widest_coverage:g}"])
        independent_calibration = (
            f"independent sampling gives {observed:.3f} empirical coverage for the "
            f"nominal {widest_coverage:.3f} interval"
        )
    return {
        "question_1": f"Mean absolute off-diagonal training PIT correlation is {mean_abs_pit_correlation:.4f}.",
        "question_2": independent_calibration,
        "question_3": delta("independent", "static_gaussian"),
        "question_4": delta("static_gaussian", "conditional_low_rank"),
        "question_5": delta("conditional_low_rank", "set_aware_low_rank"),
        "question_6": (
            "Cross-group heterogeneity is assessed in the consolidated summary; "
            "this evaluation uses the complete static group only."
        ),
    }


def evaluate_from_config(
    config: ResolvedExperimentConfig,
    *,
    methods: Sequence[str] = CORE_METHODS,
    metrics: Sequence[str] | None = None,
    figures: EvaluationFiguresConfig | None = None,
    base_figures_dir: str | Path | None = None,
    method_runs: Mapping[str, str | Path] | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Open sealed test truth once and perform final evaluation."""

    unknown = set(methods) - set(CORE_METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be non-empty and unique")
    declared_metrics = list(metrics or [
        "mean_pinball",
        "crps",
        "weighted_interval_score",
        "energy_score",
        "variogram_score",
        "test_pseudo_nll",
    ])
    cache_path = _cache_path(config, cache_dir)
    library = load_pit_library(cache_path, access="evaluation")
    runs = _resolve_runs(config, methods, method_runs)
    output = _evaluation_directory(config, output_dir)
    all_entities = list(range(library.dataset.sizes["entity"]))
    entity_ids = _entity_ids(library)
    if config.protocol.ordered_entity_ids and entity_ids != config.protocol.ordered_entity_ids:
        raise ValueError("configured ordered entity IDs do not match the complete cached group")
    if config.protocol.entity_count is not None and len(entity_ids) != config.protocol.entity_count:
        raise ValueError("configured entity_count does not match the complete cached group")
    test_origin_indices, truth, predictions, _ = _test_arrays(library, all_entities)
    levels = torch.tensor(library.dataset["quantile"].values, dtype=torch.float32)
    dependence_z, frequency_mapping = _dependence_scores(library, config)
    test_z = torch.tensor(dependence_z[test_origin_indices], dtype=torch.float32)
    valid = _valid_pairs(truth, predictions) & torch.isfinite(test_z).all(dim=1)
    prepared = {name: _prepare_method(name, runs, library, config, all_entities) for name in methods}
    results = {
        name: _sample_and_evaluate(
            item,
            truth,
            predictions,
            levels,
            valid,
            config,
            dependence_z=test_z,
            compute_joint=bool({"energy_score", "variogram_score"} & set(declared_metrics)),
        )
        for name, item in prepared.items()
    }
    result_metrics = {name: _save_method_result(output, result, valid) for name, result in results.items()}
    (output / "metrics.json").write_text(
        json.dumps(result_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.concat([_lead_table(result) for result in results.values()], ignore_index=True).to_csv(
        output / "metrics_by_lead.csv", index=False
    )
    per_case, per_origin = _case_tables(config, library, results, runs, valid)
    per_case.to_parquet(output / "per_origin_lead_metrics.parquet", index=False)
    per_origin.to_parquet(output / "per_origin_metrics.parquet", index=False)
    if frequency_mapping is not None:
        np.savez_compressed(output / "pit_training_frequency_map.npz", midpoints=frequency_mapping)

    _plots(
        output,
        library,
        prepared,
        results,
        truth,
        predictions,
        valid,
        config,
        metrics=declared_metrics,
        figures=figures or EvaluationFiguresConfig(),
        base_figures_dir=Path(base_figures_dir) if base_figures_dir is not None else output / "figures",
    )
    _, off_diagonal = _training_correlations(library)
    summary = _scientific_summary(result_metrics, float(np.mean(np.abs(off_diagonal))))
    (output / "scientific_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "cache_path": str(cache_path),
        "method_runs": {name: str(path) for name, path in runs.items()},
        "methods": list(methods),
        "metrics": declared_metrics,
        "test_origin_count": int(truth.shape[0]),
        "valid_origin_lead_count": int(valid.sum()),
        "entity_ids": entity_ids,
        "group": {
            "name": config.data.entity_type,
            "entity_ids": entity_ids,
            "entity_count": len(all_entities),
        },
        "experimental_protocol": {
            "name": config.protocol.name,
            "full_group_only": config.protocol.full_group_only,
            "subset_training": False,
            "entity_selection_augmentation_enabled": False,
        },
        "joint_score_estimator": "empirical all-pairs estimator on the selected joint ensemble",
        "joint_score_num_samples": config.evaluation.joint_score_num_samples,
        "scenario_num_samples": config.sampling.num_samples,
        "common_random_numbers": config.sampling.common_random_numbers,
        "evaluation_seed": config.sampling.evaluation_seed,
        "config_sha256": config_sha256(config),
        "dataset_revision": config.data.revision,
        "chronos_source_revision": config.chronos.source_revision,
        "chronos_model_revision": config.chronos.model_revision,
        "dependence_pit_transform": config.pit.dependence_transform,
        "pit_mode": config.pit.mode,
        "model_sha256": {
            name: sha256_file(
                path
                / (
                    "best.pt"
                    if name in {"conditional_low_rank", "set_aware_low_rank", "conditional_kernel"}
                    else "model.npz"
                )
            )
            for name, path in runs.items()
        },
        "git_commit": git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if config.output.save_resolved_config:
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
    return output
