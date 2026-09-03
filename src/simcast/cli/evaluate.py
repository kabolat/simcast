"""Final test evaluation for fixed-marginal spatial dependence methods."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Protocol, cast

import numpy as np
import pandas as pd
import torch
import typer
import yaml  # type: ignore[import-untyped]

from simcast.cli.train_dependence import _cache_path
from simcast.config import SimcastConfig, load_config
from simcast.dependence import IndependentCopula, StaticGaussianCopula
from simcast.evaluation.aggregate import AggregateEvaluation, evaluate_aggregate_ensemble
from simcast.evaluation.plots import (
    plot_aggregate_fan,
    plot_correlation_heatmap,
    plot_dependence_dynamics,
    plot_eigenvalue_spectrum,
    plot_entity_load_traces,
    plot_entity_locations,
    plot_factor_parameters,
    plot_interval_coverage,
    plot_method_summary,
    plot_missingness,
    plot_pinball_by_quantile,
    plot_pit_histogram,
    plot_quantile_coverage,
    plot_score_by_lead,
    plot_variable_cardinality,
)
from simcast.fm.cache import PITLibrary, load_pit_library
from simcast.sampling.gaussian_copula import generate_scenarios
from simcast.training.checkpoint import LoadedConditionalModel, load_conditional_checkpoint

LOGGER = logging.getLogger(__name__)
CORE_METHODS = ("independent", "static_gaussian", "conditional_low_rank", "set_aware_low_rank")


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
    correlations: torch.Tensor


def _latest_run(root: Path, method: str) -> Path:
    candidates = sorted(path for path in root.glob(f"*_{method}") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no trained run matching '*_{method}' under {root}")
    return candidates[-1]


def _resolve_runs(
    config: SimcastConfig,
    methods: Sequence[str],
    supplied: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    provided = {} if supplied is None else {name: Path(path).expanduser().resolve() for name, path in supplied.items()}
    root = Path(config.output.root_dir).expanduser().resolve()
    for method in methods:
        if method != "independent" and method not in provided:
            provided[method] = _latest_run(root, method)
    return provided


def _evaluation_directory(config: SimcastConfig, override: str | Path | None) -> Path:
    if override is None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
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
    config: SimcastConfig,
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
        matrices = torch.stack([model.correlation_matrix(lead, entity_ids=ids) for lead in range(1, horizon + 1)])
        return PreparedMethod(name, matrices[None].expand(n_origin, -1, -1, -1).to(torch.float32).clone())
    checkpoint = load_conditional_checkpoint(run_paths[name] / "best.pt", device="cpu")
    if checkpoint.method != name:
        raise ValueError(f"{name} run contains a {checkpoint.method} checkpoint")
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


def _joint_scores(samples: torch.Tensor, truth: torch.Tensor, power: float) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.linalg.vector_norm(samples - truth[:, None, :], dim=-1).mean(dim=-1)
    paired = torch.linalg.vector_norm(samples - samples.roll(1, dims=1), dim=-1).mean(dim=-1)
    energy = first - 0.5 * paired
    left, right = torch.triu_indices(samples.shape[-1], samples.shape[-1], offset=1, device=samples.device)
    observed = torch.abs(truth[:, left] - truth[:, right]).pow(power)
    predicted = torch.abs(samples[:, :, left] - samples[:, :, right]).pow(power).mean(dim=1)
    variogram = (observed - predicted).square().sum(dim=-1)
    return energy, variogram


def _sample_and_evaluate(
    prepared: PreparedMethod,
    truth: torch.Tensor,
    predictions: torch.Tensor,
    levels: torch.Tensor,
    valid: torch.Tensor,
    config: SimcastConfig,
    *,
    num_samples: int | None = None,
    compute_joint: bool = True,
) -> MethodEvaluation:
    sample_count = config.sampling.num_samples if num_samples is None else num_samples
    n_origin, n_entity, horizon = truth.shape
    aggregate = torch.full((n_origin * horizon, sample_count), torch.nan)
    energy = torch.full((n_origin * horizon,), torch.nan)
    variogram = torch.full((n_origin * horizon,), torch.nan)
    correlations = prepared.correlations.reshape(-1, n_entity, n_entity)
    marginal = predictions.permute(0, 2, 1, 3).reshape(-1, n_entity, predictions.shape[-1])
    realized = truth.permute(0, 2, 1).reshape(-1, n_entity)
    case_indices = valid.reshape(-1).nonzero(as_tuple=False).flatten()
    device = torch.device(config.chronos.device if torch.cuda.is_available() else "cpu")
    for offset in range(0, case_indices.numel(), config.evaluation.scenario_batch_size):
        positions = case_indices[offset : offset + config.evaluation.scenario_batch_size]
        batch_size = positions.numel()
        generator = torch.Generator(device=device).manual_seed(config.seed + offset)
        base_normals = torch.randn((batch_size, sample_count, n_entity), generator=generator, device=device)
        scenarios = generate_scenarios(
            correlations.index_select(0, positions).to(device),
            marginal.index_select(0, positions).to(device),
            levels.to(device),
            num_samples=sample_count,
            base_normals=base_normals,
        )
        aggregate[positions] = scenarios.aggregate_samples.cpu()
        if compute_joint:
            joint_count = min(sample_count, config.evaluation.joint_score_num_samples)
            joint_samples = scenarios.entity_samples[:, :joint_count]
            batch_energy, batch_variogram = _joint_scores(
                joint_samples,
                realized.index_select(0, positions).to(device),
                config.evaluation.variogram_power,
            )
            energy[positions] = batch_energy.cpu()
            variogram[positions] = batch_variogram.cpu()
    aggregate = aggregate.reshape(n_origin, horizon, sample_count)
    energy = energy.reshape(n_origin, horizon)
    variogram = variogram.reshape(n_origin, horizon)
    aggregate_truth = truth.sum(dim=1)
    report = evaluate_aggregate_ensemble(
        aggregate,
        aggregate_truth,
        quantile_levels=torch.tensor(config.evaluation.quantile_levels),
        interval_coverages=tuple(config.evaluation.interval_levels),
        valid_mask=valid,
    )
    return MethodEvaluation(prepared.name, report, energy, variogram, prepared.correlations)


def _serializable_metrics(result: MethodEvaluation, valid: torch.Tensor) -> dict[str, float | int]:
    metrics: dict[str, float | int] = dict(result.aggregate.overall)
    metrics["valid_origin_lead_count"] = int(valid.sum())
    metrics["dropped_origin_lead_count"] = int(valid.numel() - valid.sum())
    if torch.isfinite(result.energy_score).any():
        metrics["energy_score"] = float(result.energy_score[torch.isfinite(result.energy_score)].mean())
        metrics["variogram_score"] = float(result.variogram_score[torch.isfinite(result.variogram_score)].mean())
    return metrics


def _save_method_result(run_dir: Path, result: MethodEvaluation, valid: torch.Tensor) -> dict[str, float | int]:
    metrics = _serializable_metrics(result, valid)
    np.savez_compressed(
        run_dir / f"{result.name}_aggregate_predictions.npz",
        quantile_predictions=result.aggregate.quantile_predictions.numpy(),
        correlations=result.correlations.numpy(),
        energy_score=result.energy_score.numpy(),
        variogram_score=result.variogram_score.numpy(),
        valid=valid.numpy(),
    )
    return metrics


def _lead_table(result: MethodEvaluation) -> pd.DataFrame:
    table = result.aggregate.by_lead.copy()
    for name, values in (("energy_score", result.energy_score), ("variogram_score", result.variogram_score)):
        table[name] = [
            float(column[torch.isfinite(column)].mean()) if torch.isfinite(column).any() else np.nan
            for lead in table.index
            for column in (values[:, int(lead) - 1],)
        ]
    table.insert(0, "method", result.name)
    return table.reset_index()


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
    config: SimcastConfig,
) -> None:
    figures = output / "figures"
    ids = _entity_ids(library)
    dataset = library.dataset
    latitude = dataset["latitude"].values if "latitude" in dataset else np.arange(len(ids))
    longitude = dataset["longitude"].values if "longitude" in dataset else np.zeros(len(ids))
    plot_entity_locations(latitude, longitude, ids, figures / "dataset_locations.png")
    plot_entity_load_traces(dataset["true_y"].values, ids, figures / "dataset_load_traces.png")
    plot_missingness(dataset["true_y"].values, ids, figures / "dataset_missingness.png")
    tune = dataset.where(dataset["split"] != "test", drop=True)
    plot_pit_histogram(tune["pit_u"].values, figures / "marginal_pit.png")
    plot_quantile_coverage(
        tune["true_y"].values,
        tune["quantile_prediction"].values,
        dataset["quantile"].values,
        figures / "marginal_quantile_coverage.png",
    )
    plot_pinball_by_quantile(
        tune["true_y"].values,
        tune["quantile_prediction"].values,
        dataset["quantile"].values,
        figures / "marginal_pinball.png",
    )
    empirical, _ = _training_correlations(library)
    if "static_gaussian" in prepared:
        static = prepared["static_gaussian"].correlations[0, 0].numpy()
        plot_correlation_heatmap(
            empirical, ids, figures / "pit_empirical_correlation.png", title="Training PIT correlation"
        )
        plot_correlation_heatmap(static, ids, figures / "static_correlation.png", title="Ledoit-Wolf PIT correlation")
        plot_eigenvalue_spectrum({"Empirical": empirical, "Ledoit-Wolf": static}, figures / "static_eigenvalues.png")
    valid_positions = valid.reshape(-1).nonzero(as_tuple=False).flatten()
    if valid_positions.numel():
        position = int(valid_positions[0])
        origin, lead_index = divmod(position, truth.shape[-1])
        aggregate_truth = truth[origin].sum(dim=0).numpy()
        for name, result in results.items():
            plot_aggregate_fan(
                result.aggregate.quantile_predictions[origin].numpy(),
                config.evaluation.quantile_levels,
                aggregate_truth,
                figures / f"aggregate_fan_{name}.png",
                title=f"Aggregate forecast: {name}",
            )
            plot_correlation_heatmap(
                result.correlations[origin, lead_index].numpy(),
                ids,
                figures / f"correlation_{name}.png",
                title=f"{name}, origin {origin}, lead {lead_index + 1}",
            )
    if "static_gaussian" in prepared:
        static_by_origin = prepared["static_gaussian"].correlations[:, 0]
        for name in ("conditional_low_rank", "set_aware_low_rank"):
            item = prepared.get(name)
            if item is None:
                continue
            plot_dependence_dynamics(
                item.correlations[:, 0].numpy(),
                static_by_origin.numpy(),
                figures / f"dependence_dynamics_{name}.png",
            )
            if item.conditional is not None and item.features is not None:
                model = item.conditional.model
                with torch.no_grad():
                    loadings, sigma = cast(_FactorModel, model).factor_parameters(item.features[0, :, 0])
                plot_factor_parameters(loadings.numpy(), sigma.numpy(), ids, figures / f"factors_{name}.png")
    plot_method_summary(
        {name: result.aggregate.overall for name, result in results.items()},
        figures / "summary_pinball.png",
    )
    plot_interval_coverage(
        {name: result.aggregate.overall for name, result in results.items()},
        config.evaluation.interval_levels,
        figures / "summary_coverage.png",
    )
    plot_score_by_lead(
        {name: result.aggregate.by_lead["mean_pinball"].tolist() for name, result in results.items()},
        figures / "summary_by_lead.png",
    )


def _scientific_summary(
    metrics: Mapping[str, Mapping[str, float | int]],
    mean_abs_pit_correlation: float,
    variable_k_path: Path,
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
        "question_6": f"Variable-cardinality aggregate diagnostics are stored in {variable_k_path.name}.",
    }


def evaluate_from_config(
    config: SimcastConfig,
    *,
    methods: Sequence[str] = CORE_METHODS,
    method_runs: Mapping[str, str | Path] | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Open sealed test truth once and perform final evaluation."""

    unknown = set(methods) - set((*CORE_METHODS, "conditional_kernel"))
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be non-empty and unique")
    cache_path = _cache_path(config, cache_dir)
    library = load_pit_library(cache_path, access="evaluation")
    runs = _resolve_runs(config, methods, method_runs)
    output = _evaluation_directory(config, output_dir)
    all_entities = list(range(library.dataset.sizes["entity"]))
    _, truth, predictions, _ = _test_arrays(library, all_entities)
    levels = torch.tensor(library.dataset["quantile"].values, dtype=torch.float32)
    valid = _valid_pairs(truth, predictions)
    prepared = {name: _prepare_method(name, runs, library, config, all_entities) for name in methods}
    results = {
        name: _sample_and_evaluate(
            item,
            truth,
            predictions,
            levels,
            valid,
            config,
            compute_joint=config.evaluation.energy_score or config.evaluation.variogram_score,
        )
        for name, item in prepared.items()
    }
    metrics = {name: _save_method_result(output, result, valid) for name, result in results.items()}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.concat([_lead_table(result) for result in results.values()], ignore_index=True).to_csv(
        output / "metrics_by_lead.csv", index=False
    )

    coverage_key = f"coverage_{config.evaluation.interval_levels[-1]:g}"
    variable_rows: list[dict[str, Any]] = []
    for cardinality in config.evaluation.variable_k_sizes:
        if cardinality > len(all_entities):
            continue
        if cardinality == len(all_entities):
            for name, result in results.items():
                variable_rows.append(
                    {
                        "method": name,
                        "entities": cardinality,
                        "mean_pinball": result.aggregate.overall["mean_pinball"],
                        coverage_key: result.aggregate.overall[coverage_key],
                    }
                )
            continue
        subset = all_entities[:cardinality]
        _, subset_truth, subset_predictions, _ = _test_arrays(library, subset)
        subset_valid = _valid_pairs(subset_truth, subset_predictions)
        for name in methods:
            subset_prepared = _prepare_method(name, runs, library, config, subset)
            subset_result = _sample_and_evaluate(
                subset_prepared,
                subset_truth,
                subset_predictions,
                levels,
                subset_valid,
                config,
                num_samples=min(1024, config.sampling.num_samples),
                compute_joint=False,
            )
            variable_rows.append(
                {
                    "method": name,
                    "entities": cardinality,
                    "mean_pinball": subset_result.aggregate.overall["mean_pinball"],
                    coverage_key: subset_result.aggregate.overall[coverage_key],
                }
            )
    variable_k_path = output / "variable_k.csv"
    variable_table = pd.DataFrame(variable_rows, columns=["method", "entities", "mean_pinball", coverage_key])
    variable_table.to_csv(variable_k_path, index=False)
    if not variable_table.empty:
        plot_variable_cardinality(
            {
                str(name): (
                    subset["entities"].astype(int).tolist(),
                    subset["mean_pinball"].astype(float).tolist(),
                )
                for name, subset in variable_table.groupby("method", sort=False)
            },
            output / "figures" / "variable_cardinality.png",
        )
    _plots(output, library, prepared, results, truth, predictions, valid, config)
    _, off_diagonal = _training_correlations(library)
    summary = _scientific_summary(metrics, float(np.mean(np.abs(off_diagonal))), variable_k_path)
    (output / "scientific_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "cache_path": str(cache_path),
        "method_runs": {name: str(path) for name, path in runs.items()},
        "methods": list(methods),
        "test_origin_count": int(truth.shape[0]),
        "valid_origin_lead_count": int(valid.sum()),
        "entity_ids": _entity_ids(library),
        "joint_score_estimator": "cyclic paired Monte Carlo estimator",
        "created_at": datetime.now(UTC).isoformat(),
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if config.output.save_resolved_config:
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
    return output


def _parse_run(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        method, separator, path = value.partition("=")
        if not separator or not method or not path:
            raise ValueError("--method-run values must use method=/path/to/run")
        result[method] = Path(path)
    return result


def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False, readable=True)],
    methods: Annotated[list[str] | None, typer.Option("--methods")] = None,
    method_run: Annotated[list[str] | None, typer.Option("--method-run")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir", file_okay=False)] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False)] = None,
    override: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    resolved = load_config(config, overrides=override or ())
    logging.basicConfig(level=getattr(logging, resolved.runtime.log_level))
    path = evaluate_from_config(
        resolved,
        methods=methods or CORE_METHODS,
        method_runs=_parse_run(method_run or ()),
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    typer.echo(path)


if __name__ == "__main__":
    typer.run(main)
