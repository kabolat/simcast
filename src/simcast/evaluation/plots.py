"""Compact Matplotlib figures for datasets, marginals, dependence, and aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _finish(figure: Figure, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_entity_locations(
    latitude: Sequence[float] | np.ndarray,
    longitude: Sequence[float] | np.ndarray,
    entity_ids: Sequence[str],
    path: str | Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(longitude, latitude, s=28)
    for x, y, label in zip(longitude, latitude, entity_ids, strict=True):
        axis.annotate(label.split("::")[-1], (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    axis.set(xlabel="Longitude", ylabel="Latitude", title="Entity locations")
    return _finish(figure, path)


def plot_entity_load_traces(true_y: np.ndarray, entity_ids: Sequence[str], path: str | Path) -> Path:
    values = np.asarray(true_y)
    figure, axis = plt.subplots(figsize=(10, 4))
    flattened = values.transpose(1, 0, 2).reshape(values.shape[1], -1)
    for entity, label in zip(flattened, entity_ids, strict=True):
        axis.plot(entity, linewidth=0.7, alpha=0.75, label=label.split("::")[-1])
    axis.set(xlabel="Evaluation step", ylabel="Load", title="Entity load traces")
    if len(entity_ids) <= 15:
        axis.legend(ncol=3, fontsize=6)
    return _finish(figure, path)


def plot_missingness(true_y: np.ndarray, entity_ids: Sequence[str], path: str | Path) -> Path:
    values = np.asarray(true_y).transpose(1, 0, 2).reshape(len(entity_ids), -1)
    figure, axis = plt.subplots(figsize=(10, max(3, len(entity_ids) * 0.25)))
    image = axis.imshow(np.isnan(values), aspect="auto", interpolation="nearest", cmap="Greys")
    axis.set(xlabel="Evaluation step", ylabel="Entity", title="Target missingness")
    axis.set_yticks(np.arange(len(entity_ids)), [item.split("::")[-1] for item in entity_ids], fontsize=6)
    figure.colorbar(image, ax=axis, label="Missing")
    return _finish(figure, path)


def plot_pit_histogram(pit_u: np.ndarray, path: str | Path) -> Path:
    values = np.asarray(pit_u)
    values = values[np.isfinite(values)]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=np.linspace(0, 1, 21).tolist(), density=True, alpha=0.8)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Discretized PIT pseudo-observation", ylabel="Density", title="FM marginal PIT")
    return _finish(figure, path)


def plot_quantile_coverage(
    true_y: np.ndarray,
    quantile_prediction: np.ndarray,
    levels: Sequence[float] | np.ndarray,
    path: str | Path,
) -> Path:
    truth = np.asarray(true_y)
    prediction = np.asarray(quantile_prediction)
    coverage_values: list[float] = []
    for index in range(prediction.shape[-1]):
        valid = np.isfinite(truth) & np.isfinite(prediction[..., index])
        coverage_values.append(float((truth[valid] <= prediction[..., index][valid]).mean()))
    coverage = np.asarray(coverage_values)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(levels, coverage, marker="o", label="Empirical")
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="Nominal")
    axis.set(xlabel="Nominal quantile", ylabel="Empirical coverage", title="Marginal quantile coverage")
    axis.legend()
    return _finish(figure, path)


def plot_pinball_by_quantile(
    true_y: np.ndarray,
    quantile_prediction: np.ndarray,
    levels: Sequence[float] | np.ndarray,
    path: str | Path,
) -> Path:
    truth = np.asarray(true_y)
    prediction = np.asarray(quantile_prediction)
    losses: list[float] = []
    for index, level in enumerate(levels):
        valid = np.isfinite(truth) & np.isfinite(prediction[..., index])
        error = truth[valid] - prediction[..., index][valid]
        losses.append(float(np.maximum(level * error, (level - 1) * error).mean()))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(levels, losses, marker="o")
    axis.set(xlabel="Native quantile", ylabel="Pinball loss", title="Marginal pinball loss")
    return _finish(figure, path)


def plot_correlation_heatmap(
    correlation: np.ndarray, entity_ids: Sequence[str], path: str | Path, *, title: str
) -> Path:
    matrix = np.asarray(correlation)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [item.split("::")[-1] for item in entity_ids]
    axis.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=6)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Correlation")
    return _finish(figure, path)


def plot_eigenvalue_spectrum(correlations: Mapping[str, np.ndarray], path: str | Path) -> Path:
    figure, axis = plt.subplots(figsize=(6, 4))
    for label, matrix in correlations.items():
        eigenvalues = np.linalg.eigvalsh(np.asarray(matrix))[::-1]
        axis.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker="o", label=label)
    axis.set(xlabel="Ordered component", ylabel="Eigenvalue", title="Spatial correlation spectrum")
    axis.legend()
    return _finish(figure, path)


def plot_dependence_dynamics(
    conditional: np.ndarray,
    static: np.ndarray,
    path: str | Path,
) -> Path:
    matrices = np.asarray(conditional)
    reference = np.asarray(static)
    consecutive = np.linalg.norm(np.diff(matrices, axis=0), axis=(-2, -1))
    to_static = np.linalg.norm(matrices - reference, axis=(-2, -1))
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(np.arange(1, len(matrices)), consecutive, label="Consecutive origins")
    axis.plot(np.arange(len(matrices)), to_static, label="Versus static")
    axis.set(xlabel="Test origin", ylabel="Frobenius distance", title="Conditional dependence dynamics")
    axis.legend()
    return _finish(figure, path)


def plot_factor_parameters(
    loadings: np.ndarray, sigma: np.ndarray, entity_ids: Sequence[str], path: str | Path
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    image = axes[0].imshow(np.asarray(loadings), aspect="auto", cmap="coolwarm")
    axes[0].set(xlabel="Factor", ylabel="Entity", title="Conditional factor loadings")
    figure.colorbar(image, ax=axes[0])
    axes[1].bar(np.arange(len(entity_ids)), np.asarray(sigma))
    axes[1].set_xticks(
        np.arange(len(entity_ids)), [item.split("::")[-1] for item in entity_ids], rotation=90, fontsize=6
    )
    axes[1].set(ylabel="Sigma", title="Entity uniqueness")
    return _finish(figure, path)


def plot_aggregate_fan(
    quantile_predictions: np.ndarray,
    levels: Sequence[float],
    truth: np.ndarray,
    path: str | Path,
    *,
    title: str,
) -> Path:
    predictions = np.asarray(quantile_predictions)
    probabilities = np.asarray(levels)
    figure, axis = plt.subplots(figsize=(10, 4))
    lead = np.arange(1, predictions.shape[0] + 1)
    median_index = int(np.argmin(np.abs(probabilities - 0.5)))
    axis.plot(lead, predictions[:, median_index], label="Predictive median")
    for lower_level in (0.05, 0.1, 0.25):
        upper_level = 1 - lower_level
        lower = int(np.argmin(np.abs(probabilities - lower_level)))
        upper = int(np.argmin(np.abs(probabilities - upper_level)))
        axis.fill_between(lead, predictions[:, lower], predictions[:, upper], alpha=0.15)
    axis.plot(lead, truth, color="black", linewidth=1, label="Realized aggregate")
    axis.set(xlabel="Lead", ylabel="Aggregate load", title=title)
    axis.legend()
    return _finish(figure, path)


def plot_method_summary(metrics: Mapping[str, Mapping[str, float]], path: str | Path) -> Path:
    methods = list(metrics)
    scores = [metrics[method]["mean_pinball"] for method in methods]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(methods, scores)
    axis.tick_params(axis="x", rotation=25)
    axis.set(ylabel="Mean aggregate pinball loss", title="Aggregate probabilistic accuracy")
    return _finish(figure, path)


def plot_interval_coverage(
    metrics: Mapping[str, Mapping[str, float]], coverages: Sequence[float], path: str | Path
) -> Path:
    figure, axis = plt.subplots(figsize=(6, 5))
    for method, values in metrics.items():
        empirical = [values[f"coverage_{coverage:g}"] for coverage in coverages]
        axis.plot(coverages, empirical, marker="o", label=method)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="Nominal")
    axis.set(xlabel="Nominal coverage", ylabel="Empirical coverage", title="Aggregate interval calibration")
    axis.legend()
    return _finish(figure, path)


def plot_variable_cardinality(table: Mapping[str, tuple[Sequence[int], Sequence[float]]], path: str | Path) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4))
    for method, (cardinalities, scores) in table.items():
        axis.plot(cardinalities, scores, marker="o", label=method)
    axis.set(xlabel="Number of entities", ylabel="Mean pinball loss", title="Variable-cardinality diagnostic")
    axis.legend()
    return _finish(figure, path)


def plot_score_by_lead(table: Mapping[str, Sequence[float]], path: str | Path) -> Path:
    figure, axis = plt.subplots(figsize=(9, 4))
    for method, scores in table.items():
        axis.plot(np.arange(1, len(scores) + 1), scores, label=method)
    axis.set(xlabel="Lead", ylabel="Mean pinball loss", title="Aggregate score by lead")
    axis.legend()
    return _finish(figure, path)


def plot_training_history(
    epochs: Sequence[int], training_nll: Sequence[float], validation_nll: Sequence[float], path: str | Path
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, training_nll, label="Training")
    axis.plot(epochs, validation_nll, label="Chronological validation")
    axis.set(xlabel="Epoch", ylabel="Gaussian copula pseudo-NLL", title="Dependence-adapter training")
    axis.legend()
    return _finish(figure, path)
