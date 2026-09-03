"""Static Gaussian copula estimated from training PIT scores."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import numpy as np
import torch
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]

from simcast.dependence.base import ArrayLike, BaseDependenceModel, entity_indices, validate_training_data


def _covariance_to_correlation(covariance: np.ndarray, jitter: float) -> np.ndarray:
    """Convert a covariance estimate to a symmetric positive-definite correlation."""

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not np.isfinite(covariance).all():
        raise ValueError("covariance must be a finite square matrix")
    covariance = 0.5 * (covariance + covariance.T)
    variances = np.diag(covariance)
    positive = variances > np.finfo(np.float64).eps
    scale = np.sqrt(np.maximum(variances, np.finfo(np.float64).eps))
    correlation = covariance / np.outer(scale, scale)
    correlation[~np.outer(positive, positive)] = 0.0
    np.fill_diagonal(correlation, 1.0)

    # Numerical projection followed by diagonal normalization preserves PSD.
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (correlation + correlation.T))
    correlation = (eigenvectors * eigenvalues.clip(min=0.0)) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(correlation), np.finfo(np.float64).eps))
    correlation /= np.outer(diagonal, diagonal)

    correlation += jitter * np.eye(correlation.shape[0])
    diagonal = np.sqrt(np.diag(correlation))
    correlation /= np.outer(diagonal, diagonal)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return np.asarray(correlation, dtype=np.float64)


class StaticGaussianCopula(BaseDependenceModel):
    """M1: Ledoit-Wolf spatial correlation, per lead or pooled over leads."""

    def __init__(
        self,
        *,
        shrinkage: Literal["ledoit_wolf"] = "ledoit_wolf",
        share_across_leads: bool = False,
        jitter: float = 1e-6,
    ) -> None:
        if shrinkage != "ledoit_wolf":
            raise ValueError("only ledoit_wolf shrinkage is supported")
        if not np.isfinite(jitter) or jitter <= 0:
            raise ValueError("jitter must be positive and finite")
        self.shrinkage = shrinkage
        self.share_across_leads = share_across_leads
        self.jitter = float(jitter)
        self.entity_ids: tuple[str, ...] = ()
        self.n_leads = 0
        self._correlations: torch.Tensor | None = None

    def _estimate(self, samples: np.ndarray) -> np.ndarray:
        if samples.shape[0] < 2:
            raise ValueError("at least two complete training vectors are required")
        covariance = np.asarray(LedoitWolf().fit(samples).covariance_, dtype=np.float64)
        return _covariance_to_correlation(covariance, self.jitter)

    def fit(
        self,
        train_z: ArrayLike,
        entity_ids: Sequence[str],
        valid_mask: ArrayLike | None = None,
        **training_data: object,
    ) -> Self:
        """Estimate correlations from an already selected training-only slice.

        Any ``(origin, lead)`` containing a non-finite entity is discarded as a
        whole vector; group membership is never changed to accommodate gaps.
        """

        del training_data
        values, ids, complete = validate_training_data(train_z, entity_ids, valid_mask)
        n_leads = values.shape[2]
        if self.share_across_leads:
            pooled = np.transpose(values, (0, 2, 1))[complete]
            estimate = self._estimate(pooled)
            correlations = np.repeat(estimate[None, :, :], n_leads, axis=0)
        else:
            correlations = np.stack(
                [self._estimate(values[complete[:, lead], :, lead]) for lead in range(n_leads)],
                axis=0,
            )
        self.entity_ids = ids
        self.n_leads = n_leads
        self._correlations = torch.from_numpy(correlations)
        return self

    def correlation_matrix(
        self,
        lead: int,
        *,
        entity_ids: Sequence[str] | None = None,
        **condition: object,
    ) -> torch.Tensor:
        del condition
        if self._correlations is None:
            raise RuntimeError("fit the static Gaussian copula before use")
        if lead < 1 or lead > self.n_leads:
            raise IndexError(f"lead index out of range: {lead}")
        indices = entity_indices(self.entity_ids, entity_ids)
        index = torch.as_tensor(indices, dtype=torch.long)
        return self._correlations[lead - 1].index_select(0, index).index_select(1, index).clone()

    def save(self, path: str | Path) -> Path:
        if self._correlations is None:
            raise RuntimeError("cannot save an unfitted static Gaussian copula")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray(1, dtype=np.int64),
                model=np.asarray("static_gaussian"),
                entity_ids=np.asarray(self.entity_ids, dtype=np.str_),
                correlations=self._correlations.numpy(),
                shrinkage=np.asarray(self.shrinkage),
                share_across_leads=np.asarray(self.share_across_leads),
                jitter=np.asarray(self.jitter, dtype=np.float64),
            )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with np.load(Path(path), allow_pickle=False) as saved:
            if int(saved["schema_version"]) != 1 or str(saved["model"]) != "static_gaussian":
                raise ValueError("not a supported static-Gaussian-copula model")
            model = cls(
                shrinkage=str(saved["shrinkage"]),  # type: ignore[arg-type]
                share_across_leads=bool(saved["share_across_leads"]),
                jitter=float(saved["jitter"]),
            )
            correlations = np.asarray(saved["correlations"], dtype=np.float64)
            ids = tuple(str(item) for item in saved["entity_ids"].tolist())
        if correlations.ndim != 3 or correlations.shape[1:] != (len(ids), len(ids)) or correlations.shape[0] == 0:
            raise ValueError("saved correlations have invalid dimensions")
        if not np.isfinite(correlations).all():
            raise ValueError("saved correlations contain non-finite values")
        model.entity_ids = ids
        model.n_leads = correlations.shape[0]
        model._correlations = torch.from_numpy(correlations.copy())
        return model


StaticGaussianDependenceModel = StaticGaussianCopula
