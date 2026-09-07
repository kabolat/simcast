"""Typed configuration loading for Simcast experiments."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import time
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

EntityType: TypeAlias = Literal["transformer", "solar_park", "wind_park", "mv_feeder", "station_installation"]
Fraction = Annotated[float, Field(gt=0.0, lt=1.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Dropout = Annotated[float, Field(ge=0.0, lt=1.0)]

LIANDER2024_REVISION = "dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044"
CHRONOS_SOURCE_REVISION = "8589d1988e9676817548e9626738ff06b6ca6370"
CHRONOS_MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


class ConfigModel(BaseModel):
    """Base for strict configuration sections."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class DataConfig(ConfigModel):
    dataset_id: str = "OpenSTEF/liander2024-energy-forecasting-benchmark"
    revision: str = LIANDER2024_REVISION
    local_dir: Path = Path("data/liander2024")
    entity_type: EntityType = "transformer"
    target_column: str = "load"
    include_epex: bool = False
    include_profiles: bool = False


class ForecastConfig(ConfigModel):
    timezone: Literal["UTC"] = "UTC"
    frequency_minutes: PositiveInt = 15
    origin_time: time = time(23, 45)
    lookback_steps: PositiveInt = 672
    horizon_steps: PositiveInt = 96
    origin_stride_steps: PositiveInt = 96

    @model_validator(mode="after")
    def validate_cadence(self) -> Self:
        if 1440 % self.frequency_minutes:
            raise ValueError("forecast.frequency_minutes must divide one day exactly")
        origin_minutes = self.origin_time.hour * 60 + self.origin_time.minute
        if self.origin_time.second or self.origin_time.microsecond or origin_minutes % self.frequency_minutes:
            raise ValueError("forecast.origin_time must align to frequency_minutes")
        return self


class CalendarConfig(ConfigModel):
    enabled: bool = True
    include_hour: bool = True
    include_day_of_week: bool = True
    include_is_weekend: bool = True


class OptionalCovariateConfig(ConfigModel):
    enabled: bool = False


class CovariatesConfig(ConfigModel):
    weather: list[str] = Field(
        default_factory=lambda: [
            "temperature_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "wind_speed_10m",
            "shortwave_radiation",
        ]
    )
    future_weather_source: Literal["vintage", "oracle"] = "vintage"
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    epex: OptionalCovariateConfig = Field(default_factory=OptionalCovariateConfig)
    profiles: OptionalCovariateConfig = Field(default_factory=OptionalCovariateConfig)

    @field_validator("weather")
    @classmethod
    def unique_weather_features(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("covariates.weather cannot contain empty names")
        if len(value) != len(set(value)):
            raise ValueError("covariates.weather cannot contain duplicate names")
        return value


class SplitConfig(ConfigModel):
    tune_fraction: Fraction = 0.80
    validation_fraction_within_tune: Fraction = 0.20
    purge_overlapping_horizons: bool = True


class ChronosConfig(ConfigModel):
    source_url: str = "https://github.com/amazon-science/chronos-forecasting.git"
    source_revision: str = CHRONOS_SOURCE_REVISION
    model_id: str = "amazon/chronos-2"
    model_revision: str = CHRONOS_MODEL_REVISION
    device: str = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    batch_size: PositiveInt = 16
    cross_learning: Literal[False] = False

    @field_validator("device")
    @classmethod
    def nonempty_device(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chronos.device cannot be empty")
        return value


class PitConfig(ConfigModel):
    mode: Literal["discretized"] = "discretized"
    monotone_repair: Literal["none", "isotonic"] = "none"
    eps: Annotated[float, Field(gt=0.0, lt=0.5)] = 1.0e-7


class ProtocolConfig(ConfigModel):
    name: Literal["legacy", "full_group"] = "legacy"
    full_group_only: bool = False


class FeaturesConfig(ConfigModel):
    use_forecast_embedding: bool = True
    layer_normalize_embedding: bool = True
    use_quantile_shape: bool = True
    use_median: bool = True
    use_log_spread: bool = True
    use_within_patch_position: bool = True
    use_location: bool = True
    use_entity_id_embedding: bool = False
    shape_eps: PositiveFloat = 1.0e-6
    standardize_scalar_features: bool = True


class IndependentConfig(ConfigModel):
    pass


class StaticGaussianConfig(ConfigModel):
    shrinkage: Literal["ledoit_wolf"] = "ledoit_wolf"
    share_across_leads: bool = False
    jitter: PositiveFloat = 1.0e-6


class ConditionalLowRankConfig(ConfigModel):
    latent_rank: PositiveInt = 4
    hidden_dims: list[PositiveInt] = Field(default_factory=lambda: [256, 128], min_length=1)
    activation: Literal["gelu"] = "gelu"
    dropout: Dropout = 0.1
    sigma_floor: PositiveFloat = 1.0e-3
    jitter: PositiveFloat = 1.0e-6


class SetAwareLowRankConfig(ConfigModel):
    model_dim: PositiveInt = 128
    num_layers: PositiveInt = 2
    num_heads: PositiveInt = 4
    latent_rank: PositiveInt = 4
    dropout: Dropout = 0.1
    sigma_floor: PositiveFloat = 1.0e-3
    jitter: PositiveFloat = 1.0e-6

    @model_validator(mode="after")
    def validate_attention_shape(self) -> Self:
        if self.model_dim % self.num_heads:
            raise ValueError("dependence.set_aware_low_rank.model_dim must be divisible by num_heads")
        return self


class ConditionalKernelConfig(ConfigModel):
    hidden_dims: list[PositiveInt] = Field(default_factory=lambda: [256, 128], min_length=1)
    embedding_dim: PositiveInt = 16
    activation: Literal["gelu"] = "gelu"
    dropout: Dropout = 0.1
    initial_length_scale: PositiveFloat = 1.0
    nugget: PositiveFloat = 1.0e-3
    jitter: PositiveFloat = 1.0e-6
    smoke_only: bool = True
    smoke_max_origins: PositiveInt = 32


DependenceMethod: TypeAlias = Literal[
    "independent", "static_gaussian", "conditional_low_rank", "set_aware_low_rank", "conditional_kernel"
]


class DependenceConfig(ConfigModel):
    method: DependenceMethod = "independent"
    independent: IndependentConfig = Field(default_factory=IndependentConfig)
    static_gaussian: StaticGaussianConfig = Field(default_factory=StaticGaussianConfig)
    conditional_low_rank: ConditionalLowRankConfig = Field(default_factory=ConditionalLowRankConfig)
    set_aware_low_rank: SetAwareLowRankConfig = Field(default_factory=SetAwareLowRankConfig)
    conditional_kernel: ConditionalKernelConfig = Field(default_factory=ConditionalKernelConfig)


class SubsetTrainingConfig(ConfigModel):
    enabled: bool = False
    min_entities: Annotated[int, Field(ge=2)] = 4
    full_group_probability: Probability = 0.25

    @model_validator(mode="after")
    def require_full_group_samples(self) -> Self:
        if self.enabled and self.full_group_probability <= 0:
            raise ValueError("subset_training.full_group_probability must be positive when enabled")
        return self


class TrainingConfig(ConfigModel):
    optimizer: Literal["adamw"] = "adamw"
    batch_size: PositiveInt = 64
    epochs: PositiveInt = 100
    learning_rate: PositiveFloat = 1.0e-3
    weight_decay: Annotated[float, Field(ge=0.0)] = 1.0e-4
    gradient_clip_norm: PositiveFloat = 1.0
    patience: PositiveInt = 12

    @model_validator(mode="after")
    def validate_patience(self) -> Self:
        if self.patience > self.epochs:
            raise ValueError("training.patience cannot exceed training.epochs")
        return self


class SamplingConfig(ConfigModel):
    num_samples: PositiveInt = 4096
    empirical_quantile_method: Literal["nearest"] = "nearest"
    common_random_numbers: bool = True


class EvaluationConfig(ConfigModel):
    quantile_levels: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    interval_levels: list[float] = Field(default_factory=lambda: [0.50, 0.80, 0.90])
    energy_score: bool = True
    variogram_score: bool = True
    variogram_power: Annotated[float, Field(gt=0.0, le=2.0)] = 0.5
    report_by_lead: bool = True
    scenario_batch_size: PositiveInt = 16
    joint_score_num_samples: PositiveInt = 512
    variable_k_sizes: list[PositiveInt] = Field(default_factory=list)

    @field_validator("quantile_levels", "interval_levels")
    @classmethod
    def ordered_unique_probabilities(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("probability level lists cannot be empty")
        if any(level <= 0.0 or level >= 1.0 for level in value):
            raise ValueError("probability levels must lie strictly between zero and one")
        if value != sorted(set(value)):
            raise ValueError("probability levels must be sorted and unique")
        return value

    @field_validator("variable_k_sizes")
    @classmethod
    def ordered_unique_cardinalities(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)):
            raise ValueError("evaluation.variable_k_sizes must be sorted and unique")
        return value


class RuntimeConfig(ConfigModel):
    deterministic: bool = True
    num_workers: NonNegativeInt = 0
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class OutputConfig(ConfigModel):
    root_dir: Path = Path("runs")
    cache_dir: Path = Path("artifacts/cache")
    cache_name: str | None = None
    experiment_name: str | None = None
    save_resolved_config: bool = True


class SimcastConfig(ConfigModel):
    seed: NonNegativeInt = 42
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    covariates: CovariatesConfig = Field(default_factory=CovariatesConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    chronos: ChronosConfig = Field(default_factory=ChronosConfig)
    pit: PitConfig = Field(default_factory=PitConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    dependence: DependenceConfig = Field(default_factory=DependenceConfig)
    subset_training: SubsetTrainingConfig = Field(default_factory=SubsetTrainingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_full_group_protocol(self) -> Self:
        if self.protocol.name == "full_group" and not self.protocol.full_group_only:
            raise ValueError("full_group protocol requires full_group_only=true")
        if self.protocol.full_group_only and self.subset_training.enabled:
            raise ValueError("full-group protocol forbids subset training")
        if self.protocol.full_group_only and self.evaluation.variable_k_sizes:
            raise ValueError("full-group protocol forbids variable-cardinality evaluation")
        return self


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        environment_value = os.environ.get(name)
        if environment_value:
            return environment_value
        if default is not None:
            return default
        if environment_value is not None:
            return environment_value
        raise ValueError(f"environment variable {name!r} is not set")

    return _ENV_PATTERN.sub(replace, value)


def expand_environment(value: Any) -> Any:
    """Recursively expand ``${NAME}`` and ``${NAME:-default}`` in YAML values."""

    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; sequences and scalar values are replaced."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(previous, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def parse_overrides(overrides: Sequence[str]) -> dict[str, Any]:
    """Parse repeatable ``section.key=value`` CLI overrides into a nested mapping."""

    parsed: dict[str, Any] = {}
    for override in overrides:
        dotted_key, separator, raw_value = override.partition("=")
        parts = dotted_key.split(".")
        if not separator or any(not part for part in parts):
            raise ValueError(f"invalid override {override!r}; expected dotted.key=value")

        value = expand_environment(yaml.safe_load(raw_value))
        cursor = parsed
        for part in parts[:-1]:
            existing = cursor.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ValueError(f"override path {dotted_key!r} conflicts with another override")
            cursor = existing
        cursor[parts[-1]] = value
    return parsed


def _read_yaml(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"configuration inheritance cycle: {cycle}")
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file not found: {resolved}")

    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration root must be a mapping: {resolved}")
    document = expand_environment(loaded)

    parents = document.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(not isinstance(parent, str) for parent in parents):
        raise ValueError(f"extends must be a path or list of paths: {resolved}")

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        merged = deep_merge(merged, _read_yaml(parent_path, (*stack, resolved)))
    return deep_merge(merged, document)


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> SimcastConfig:
    """Load, compose, override, and validate a Simcast YAML configuration."""

    values = _read_yaml(Path(path), ())
    if overrides:
        values = deep_merge(values, parse_overrides(overrides))
    return SimcastConfig.model_validate(values)
