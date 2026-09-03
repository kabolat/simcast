"""Deterministic lead-level features derived from frozen Chronos forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import torch


def lead_to_patch_indices(horizon: int, output_patch_size: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return ``p(tau) = floor((tau - 1) / output_patch_size)`` for leads ``1..H``."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if output_patch_size <= 0:
        raise ValueError("output_patch_size must be positive")
    return torch.arange(horizon, device=device, dtype=torch.long) // output_patch_size


class ScalarStandardizer:
    """Per-feature standardization statistics fitted on training origins only."""

    def __init__(self, eps: float = 1.0e-6) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.scale is not None

    @torch.no_grad()
    def fit(self, training_features: torch.Tensor) -> Self:
        """Fit on a scalar-feature tensor whose last axis indexes features."""

        values = torch.as_tensor(training_features, dtype=torch.float32)
        if values.ndim < 1 or values.shape[-1] == 0:
            raise ValueError("training_features must have a non-empty feature axis")
        flat = values.reshape(-1, values.shape[-1])
        if flat.shape[0] == 0:
            raise ValueError("training_features must contain at least one observation")
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("training scalar features must be finite")
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, correction=0)
        self.mean = mean.detach().clone()
        self.scale = torch.where(std > self.eps, std, torch.ones_like(std)).detach().clone()
        return self

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the training statistics without updating them."""

        if self.mean is None or self.scale is None:
            raise RuntimeError("ScalarStandardizer must be fit on training origins before transform")
        values = torch.as_tensor(features, dtype=torch.float32)
        if values.shape[-1] != self.mean.numel():
            raise ValueError(
                f"scalar feature dimension changed from {self.mean.numel()} during fit to {values.shape[-1]}"
            )
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        scale = self.scale.to(device=values.device, dtype=values.dtype)
        return (values - mean) / scale

    def fit_transform(self, training_features: torch.Tensor) -> torch.Tensor:
        return self.fit(training_features).transform(training_features)

    def state_dict(self) -> dict[str, torch.Tensor | float | None]:
        """Return portable training-only scalar statistics."""

        return {
            "eps": self.eps,
            "mean": None if self.mean is None else self.mean.detach().cpu(),
            "scale": None if self.scale is None else self.scale.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | float | None]) -> None:
        """Restore statistics previously produced by :meth:`state_dict`."""

        eps = state["eps"]
        if not isinstance(eps, (int, float)):
            raise ValueError("standardizer eps must be numeric")
        self.eps = float(eps)
        mean, scale = state["mean"], state["scale"]
        if (mean is None) != (scale is None):
            raise ValueError("standardizer mean and scale must both be present or absent")
        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32).clone()
        self.scale = None if scale is None else torch.as_tensor(scale, dtype=torch.float32).clone()


class FeatureBuilder:
    """Construct ``v[k, tau]`` without learning another temporal model.

    The deterministic concatenation order is raw patch embedding, normalized
    native-quantile shape, median, log 80% spread, normalized within-patch
    position, and static location.  No physical entity-ID embedding exists.
    Scalar standardization, when enabled, must be fitted explicitly using only
    training origins before validation or test features are transformed.
    """

    def __init__(
        self,
        output_patch_size: int,
        *,
        shape_eps: float = 1.0e-6,
        standardize_scalar_features: bool = True,
        use_forecast_embedding: bool = True,
        use_quantile_shape: bool = True,
        use_median: bool = True,
        use_log_spread: bool = True,
        use_within_patch_position: bool = True,
        use_location: bool = True,
    ) -> None:
        if output_patch_size <= 0:
            raise ValueError("output_patch_size must be positive")
        if shape_eps <= 0:
            raise ValueError("shape_eps must be positive")
        if not any(
            (
                use_forecast_embedding,
                use_quantile_shape,
                use_median,
                use_log_spread,
                use_within_patch_position,
                use_location,
            )
        ):
            raise ValueError("at least one feature component must be enabled")
        self.output_patch_size = output_patch_size
        self.shape_eps = shape_eps
        self.standardize_scalar_features = standardize_scalar_features
        self.use_forecast_embedding = use_forecast_embedding
        self.use_quantile_shape = use_quantile_shape
        self.use_median = use_median
        self.use_log_spread = use_log_spread
        self.use_within_patch_position = use_within_patch_position
        self.use_location = use_location
        self.scalar_standardizer = ScalarStandardizer(eps=shape_eps)

    def configuration(self) -> dict[str, int | float | bool]:
        """Return the deterministic construction settings for checkpoints."""

        return {
            "output_patch_size": self.output_patch_size,
            "shape_eps": self.shape_eps,
            "standardize_scalar_features": self.standardize_scalar_features,
            "use_forecast_embedding": self.use_forecast_embedding,
            "use_quantile_shape": self.use_quantile_shape,
            "use_median": self.use_median,
            "use_log_spread": self.use_log_spread,
            "use_within_patch_position": self.use_within_patch_position,
            "use_location": self.use_location,
        }

    def state_dict(self) -> dict[str, object]:
        """Serialize construction settings and fitted training statistics."""

        return {"configuration": self.configuration(), "scalar_standardizer": self.scalar_standardizer.state_dict()}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> FeatureBuilder:
        """Restore a fitted feature builder from a conditional-model checkpoint."""

        configuration = state.get("configuration")
        standardizer = state.get("scalar_standardizer")
        if not isinstance(configuration, dict) or not isinstance(standardizer, dict):
            raise ValueError("invalid feature-builder checkpoint")
        builder = cls(**configuration)
        builder.scalar_standardizer.load_state_dict(standardizer)
        return builder

    def _validate_and_convert(
        self,
        forecast_embeddings: torch.Tensor,
        quantile_predictions: torch.Tensor,
        quantile_levels: torch.Tensor | Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = torch.as_tensor(forecast_embeddings, dtype=torch.float32)
        predictions = torch.as_tensor(
            quantile_predictions,
            device=embeddings.device,
            dtype=torch.float32,
        )
        levels = torch.as_tensor(quantile_levels, device=embeddings.device, dtype=torch.float32)
        if embeddings.ndim != 4:
            raise ValueError("forecast_embeddings must have shape [origin, entity, patch, hidden]")
        if predictions.ndim != 4:
            raise ValueError("quantile_predictions must have shape [origin, entity, lead, quantile]")
        if embeddings.shape[:2] != predictions.shape[:2]:
            raise ValueError("embedding and prediction origin/entity dimensions must match")
        if levels.ndim != 1 or levels.numel() != predictions.shape[-1]:
            raise ValueError("quantile_levels must match the final prediction dimension")
        if not bool(torch.all((levels > 0) & (levels < 1))) or not bool(torch.all(levels[1:] > levels[:-1])):
            raise ValueError("quantile_levels must be strictly increasing inside (0, 1)")
        patch_indices = lead_to_patch_indices(predictions.shape[2], self.output_patch_size, device=embeddings.device)
        if int(patch_indices[-1]) >= embeddings.shape[2]:
            raise ValueError("forecast_embeddings do not cover every prediction lead")
        if not bool(torch.isfinite(embeddings).all()) or not bool(torch.isfinite(predictions).all()):
            raise ValueError("forecast embeddings and quantile predictions must be finite")
        return embeddings, predictions, levels

    @staticmethod
    def _level_index(levels: torch.Tensor, requested: float) -> int:
        matches = torch.isclose(levels, levels.new_tensor(requested), rtol=0.0, atol=1.0e-6).nonzero().flatten()
        if matches.numel() != 1:
            raise ValueError(f"native Chronos quantiles must contain q={requested:g}")
        return int(matches[0])

    def _location_features(
        self,
        locations: torch.Tensor | None,
        *,
        n_origin: int,
        n_entity: int,
        horizon: int,
        device: torch.device,
    ) -> torch.Tensor:
        if locations is None:
            raise ValueError("locations are required when use_location=True")
        values = torch.as_tensor(locations, device=device, dtype=torch.float32)
        if values.ndim == 2:
            if values.shape[0] != n_entity:
                raise ValueError("locations [entity, coordinate] must match the entity dimension")
            values = values[None, :, None, :].expand(n_origin, -1, horizon, -1)
        elif values.ndim == 3:
            if values.shape[:2] != (n_origin, n_entity):
                raise ValueError("locations [origin, entity, coordinate] have incompatible dimensions")
            values = values[:, :, None, :].expand(-1, -1, horizon, -1)
        else:
            raise ValueError("locations must have shape [entity, coordinate] or [origin, entity, coordinate]")
        if values.shape[-1] == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("locations must contain finite coordinates")
        return values

    def _components(
        self,
        forecast_embeddings: torch.Tensor,
        quantile_predictions: torch.Tensor,
        quantile_levels: torch.Tensor | Sequence[float],
        locations: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        embeddings, predictions, levels = self._validate_and_convert(
            forecast_embeddings, quantile_predictions, quantile_levels
        )
        n_origin, n_entity, horizon, _ = predictions.shape
        feature_parts: list[torch.Tensor] = []
        scalar_parts: list[torch.Tensor] = []

        patch_indices = lead_to_patch_indices(horizon, self.output_patch_size, device=embeddings.device)
        if self.use_forecast_embedding:
            feature_parts.append(embeddings.index_select(dim=2, index=patch_indices))

        q10 = predictions[..., self._level_index(levels, 0.1)]
        median = predictions[..., self._level_index(levels, 0.5)]
        q90 = predictions[..., self._level_index(levels, 0.9)]
        spread = q90 - q10
        if self.use_quantile_shape:
            scalar_parts.append((predictions - median.unsqueeze(-1)) / (spread.abs().unsqueeze(-1) + self.shape_eps))
        if self.use_median:
            scalar_parts.append(median.unsqueeze(-1))
        if self.use_log_spread:
            scalar_parts.append(torch.log(spread.abs() + self.shape_eps).unsqueeze(-1))
        if self.use_within_patch_position:
            denominator = max(self.output_patch_size - 1, 1)
            within_patch = (torch.arange(horizon, device=embeddings.device) % self.output_patch_size) / denominator
            scalar_parts.append(within_patch[None, None, :, None].expand(n_origin, n_entity, -1, -1))
        if self.use_location:
            scalar_parts.append(
                self._location_features(
                    locations,
                    n_origin=n_origin,
                    n_entity=n_entity,
                    horizon=horizon,
                    device=embeddings.device,
                )
            )

        scalar_features = torch.cat(scalar_parts, dim=-1) if scalar_parts else None
        return feature_parts, scalar_features

    def fit(
        self,
        forecast_embeddings: torch.Tensor,
        quantile_predictions: torch.Tensor,
        quantile_levels: torch.Tensor | Sequence[float],
        locations: torch.Tensor | None = None,
    ) -> Self:
        """Fit scalar statistics; arguments must contain training origins only."""

        _, scalar_features = self._components(forecast_embeddings, quantile_predictions, quantile_levels, locations)
        if self.standardize_scalar_features and scalar_features is not None:
            self.scalar_standardizer.fit(scalar_features)
        return self

    def transform(
        self,
        forecast_embeddings: torch.Tensor,
        quantile_predictions: torch.Tensor,
        quantile_levels: torch.Tensor | Sequence[float],
        locations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build features using frozen scalar statistics."""

        feature_parts, scalar_features = self._components(
            forecast_embeddings, quantile_predictions, quantile_levels, locations
        )
        if scalar_features is not None:
            if self.standardize_scalar_features:
                scalar_features = self.scalar_standardizer.transform(scalar_features)
            feature_parts.append(scalar_features)
        return torch.cat(feature_parts, dim=-1)

    def fit_transform(
        self,
        forecast_embeddings: torch.Tensor,
        quantile_predictions: torch.Tensor,
        quantile_levels: torch.Tensor | Sequence[float],
        locations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.fit(forecast_embeddings, quantile_predictions, quantile_levels, locations)
        return self.transform(forecast_embeddings, quantile_predictions, quantile_levels, locations)

    def build(
        self,
        forecast_embeddings: torch.Tensor,
        quantile_predictions: torch.Tensor,
        quantile_levels: torch.Tensor | Sequence[float],
        locations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for :meth:`transform` for inference-time call sites."""

        return self.transform(forecast_embeddings, quantile_predictions, quantile_levels, locations)
