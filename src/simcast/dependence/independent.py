"""Independent copula baseline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import torch

from simcast.dependence.base import ArrayLike, BaseDependenceModel, entity_indices, validate_training_data


class IndependentCopula(BaseDependenceModel):
    """M0: identity correlation for any requested entity group."""

    def __init__(self, entity_ids: Sequence[str] | None = None, n_leads: int | None = None) -> None:
        ids = tuple(entity_ids) if entity_ids is not None else ()
        if any(not isinstance(entity_id, str) or not entity_id for entity_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("entity_ids must be unique non-empty strings")
        if n_leads is not None and n_leads <= 0:
            raise ValueError("n_leads must be positive")
        self.entity_ids = ids
        self.n_leads = n_leads

    def fit(
        self,
        train_z: ArrayLike,
        entity_ids: Sequence[str],
        valid_mask: ArrayLike | None = None,
        **training_data: object,
    ) -> Self:
        """Bind entity/lead metadata without estimating parameters."""

        del training_data
        values, ids, _ = validate_training_data(train_z, entity_ids, valid_mask)
        self.entity_ids = ids
        self.n_leads = values.shape[2]
        return self

    def correlation_matrix(
        self,
        lead: int,
        *,
        entity_ids: Sequence[str] | None = None,
        **condition: object,
    ) -> torch.Tensor:
        del condition
        if lead < 1 or self.n_leads is not None and lead > self.n_leads:
            raise IndexError(f"lead index out of range: {lead}")
        if self.entity_ids:
            size = len(entity_indices(self.entity_ids, entity_ids))
        elif entity_ids is not None:
            requested = tuple(entity_ids)
            if (
                not requested
                or any(not isinstance(entity_id, str) or not entity_id for entity_id in requested)
                or len(requested) != len(set(requested))
            ):
                raise ValueError("entity_ids must be unique non-empty strings")
            size = len(requested)
        else:
            raise RuntimeError("provide entity_ids or bind them with fit")
        return torch.eye(size, dtype=torch.float64)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray(1, dtype=np.int64),
                model=np.asarray("independent"),
                entity_ids=np.asarray(self.entity_ids, dtype=np.str_),
                n_leads=np.asarray(-1 if self.n_leads is None else self.n_leads, dtype=np.int64),
            )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with np.load(Path(path), allow_pickle=False) as saved:
            if int(saved["schema_version"]) != 1 or str(saved["model"]) != "independent":
                raise ValueError("not a supported independent-copula model")
            ids = tuple(str(item) for item in saved["entity_ids"].tolist())
            saved_n_leads = int(saved["n_leads"])
        return cls(ids, None if saved_n_leads < 0 else saved_n_leads)


IndependentDependenceModel = IndependentCopula
