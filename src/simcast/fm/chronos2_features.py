"""Frozen Chronos-2 inference with forecast-patch feature extraction.

Chronos-2 forecasts each physical entity separately.  Target and covariate rows
within an entity retain their normal Chronos group, while rows belonging to
different entities never share a group ID.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from chronos.chronos2 import Chronos2Pipeline  # type: ignore[import-untyped]
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode  # type: ignore[import-untyped]
from chronos.chronos2.preprocess import PreparedInput  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

TensorOrArray: TypeAlias = torch.Tensor | np.ndarray
ChronosInput: TypeAlias = (
    TensorOrArray | Sequence[TensorOrArray] | Sequence[Mapping[str, object]] | Sequence[PreparedInput]
)


@dataclass(frozen=True, slots=True)
class FMForecast:
    """Native marginal forecasts and frozen forecast-patch representations.

    ``quantile_predictions[k, tau - 1, j]`` is the fixed Chronos marginal
    quantile for physical entity ``k``, lead ``tau``, and native quantile
    level ``j``.  ``forecast_embeddings[k, p]`` is shared by every lead in
    output patch ``p``; it is deliberately not repeated once per lead.
    """

    entity_ids: list[str]
    quantile_levels: torch.Tensor
    quantile_predictions: torch.Tensor
    forecast_embeddings: torch.Tensor
    output_patch_size: int


class Chronos2FeatureExtractor:
    """Project-local adapter around a fully frozen :class:`Chronos2Pipeline`."""

    def __init__(self, pipeline: Chronos2Pipeline) -> None:
        self.pipeline = pipeline
        self.model = pipeline.model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def direct_prediction_capacity(self) -> int:
        """Largest horizon supported by one non-unrolled model forward pass."""

        config = self.model.chronos_config
        return int(config.max_output_patches * config.output_patch_size)

    @torch.no_grad()
    def predict(
        self,
        entity_ids: Sequence[str],
        inputs: ChronosInput,
        prediction_length: int,
        *,
        batch_size: int = 256,
        context_length: int | None = None,
    ) -> FMForecast:
        """Forecast one target variate per physical entity.

        Preprocessing, instance scaling, patching, and the target-row mapping
        all come from the official Chronos-2 dataset/model implementation.
        Only direct prediction is supported because autoregressive unrolling
        does not yield one coherent set of forecast-patch representations.
        """

        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if prediction_length > self.direct_prediction_capacity:
            raise NotImplementedError(
                "Chronos forecast-embedding extraction supports only a single direct forward pass; "
                f"requested horizon {prediction_length} exceeds direct capacity "
                f"{self.direct_prediction_capacity}."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        model_context_length = self.pipeline.model_context_length
        if context_length is None:
            context_length = model_context_length
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if context_length > model_context_length:
            raise ValueError(f"context_length {context_length} exceeds the model capacity {model_context_length}")

        dataset = Chronos2Dataset(
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=self.pipeline.model_output_patch_size,
            mode=DatasetMode.TEST,
        )
        ids = list(entity_ids)
        if len(ids) != len(dataset.inputs):
            raise ValueError(
                f"entity_ids and Chronos inputs must have equal length; got {len(ids)} and {len(dataset.inputs)}"
            )
        for entity_id, prepared in zip(ids, dataset.inputs, strict=True):
            if int(prepared["n_targets"]) != 1:
                raise ValueError(
                    f"entity {entity_id!r} must contain exactly one target variate; got {int(prepared['n_targets'])}"
                )

        loader = DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.model.device.type == "cuda",
            shuffle=False,
            drop_last=False,
        )
        num_output_patches = math.ceil(prediction_length / self.pipeline.model_output_patch_size)
        quantile_predictions: list[torch.Tensor] = []
        forecast_embeddings: list[torch.Tensor] = []

        for batch in loader:
            if batch["future_target"] is not None:
                raise RuntimeError("Chronos test preprocessing unexpectedly returned future targets")

            # Do not replace group_ids with zeros: each physical entity must
            # remain an independent Chronos task (cross_learning=False).
            context = batch["context"].to(device=self.model.device, dtype=torch.float32)
            group_ids = batch["group_ids"].to(device=self.model.device)
            future_covariates = batch["future_covariates"].to(device=self.model.device, dtype=torch.float32)
            output = self.model(
                context=context,
                group_ids=group_ids,
                future_covariates=future_covariates,
                num_output_patches=num_output_patches,
            )
            if output.quantile_preds is None:
                raise RuntimeError("Chronos-2 did not return quantile predictions")
            if output.forecast_embeds is None:
                raise RuntimeError(
                    "Chronos-2 did not return forecast embeddings; run scripts/setup_chronos.sh "
                    "to install the pinned patched source."
                )

            # This is the exact target mapping emitted and consumed by the
            # official Chronos2Dataset/Chronos2Pipeline prediction path.
            for start, end in batch["target_idx_ranges"]:
                if end - start != 1:
                    raise ValueError(f"expected one target row per physical entity, got range {(start, end)}")
                quantile_predictions.append(
                    output.quantile_preds[start:end, :, :prediction_length]
                    .permute(0, 2, 1)
                    .to(device="cpu", dtype=torch.float32)
                )
                forecast_embeddings.append(output.forecast_embeds[start:end].to(device="cpu", dtype=torch.float32))

        if len(quantile_predictions) != len(ids):
            raise RuntimeError(
                f"Chronos returned {len(quantile_predictions)} target forecasts for {len(ids)} entities"
            )

        return FMForecast(
            entity_ids=ids,
            quantile_levels=torch.as_tensor(self.model.chronos_config.quantiles, dtype=torch.float32, device="cpu"),
            quantile_predictions=torch.cat(quantile_predictions, dim=0),
            forecast_embeddings=torch.cat(forecast_embeddings, dim=0),
            output_patch_size=self.pipeline.model_output_patch_size,
        )
