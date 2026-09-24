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
    TypeAdapter,
    field_validator,
    model_validator,
)

from simcast.reproducibility import canonical_json_hash

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
    mode: Literal["discretized", "linear_interpolation"] = "discretized"
    monotone_repair: Literal["none", "isotonic"] = "none"
    dependence_transform: Literal["nominal_cells", "training_frequency"] = "nominal_cells"
    eps: Annotated[float, Field(gt=0.0, lt=0.5)] = 1.0e-7

    @model_validator(mode="after")
    def validate_dependence_transform(self) -> Self:
        if self.mode == "linear_interpolation" and self.dependence_transform == "training_frequency":
            raise ValueError("training_frequency is defined only for discretized PIT cells")
        return self


class ProtocolConfig(ConfigModel):
    name: Literal["full_group"] = "full_group"
    full_group_only: Literal[True] = True
    ordered_entity_ids: list[str] = Field(default_factory=list)
    entity_count: PositiveInt | None = None

    @field_validator("ordered_entity_ids")
    @classmethod
    def unique_entity_ids(cls, value: list[str]) -> list[str]:
        if any(not entity_id.strip() for entity_id in value) or len(value) != len(set(value)):
            raise ValueError("protocol.ordered_entity_ids must be non-empty unique strings when supplied")
        return value


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


DependenceMethod: TypeAlias = Literal[
    "independent", "static_gaussian", "conditional_low_rank", "set_aware_low_rank", "conditional_kernel"
]


class DependenceConfig(ConfigModel):
    method: DependenceMethod = "independent"
    model: (
        StaticGaussianConfig
        | ConditionalLowRankConfig
        | SetAwareLowRankConfig
        | ConditionalKernelConfig
        | None
    ) = None

    @model_validator(mode="after")
    def model_matches_method(self) -> Self:
        expected = {
            "static_gaussian": StaticGaussianConfig,
            "conditional_low_rank": ConditionalLowRankConfig,
            "set_aware_low_rank": SetAwareLowRankConfig,
            "conditional_kernel": ConditionalKernelConfig,
        }.get(self.method)
        if expected is None:
            if self.model is not None:
                raise ValueError("independent dependence must not define model parameters")
        elif not isinstance(self.model, expected):
            raise ValueError(f"{self.method} dependence requires {expected.__name__}")
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
    evaluation_seed: NonNegativeInt = 2027
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


class ResolvedExperimentConfig(ConfigModel):
    seed: NonNegativeInt = 42
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    covariates: CovariatesConfig = Field(default_factory=CovariatesConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    chronos: ChronosConfig = Field(default_factory=ChronosConfig)
    pit: PitConfig = Field(default_factory=PitConfig)
    features: FeaturesConfig | None = None
    dependence: DependenceConfig = Field(default_factory=DependenceConfig)
    training: TrainingConfig | None = None
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_full_group_protocol(self) -> Self:
        if (
            self.protocol.entity_count is not None
            and self.protocol.entity_count != len(self.protocol.ordered_entity_ids)
        ):
            raise ValueError("protocol.entity_count must equal the ordered entity-ID count")
        optimized = self.dependence.method in {
            "conditional_low_rank",
            "set_aware_low_rank",
            "conditional_kernel",
        }
        if optimized and (self.features is None or self.training is None):
            raise ValueError("conditional dependence requires features and optimization parameters")
        if not optimized and (self.features is not None or self.training is not None):
            raise ValueError("independent and static dependence must not define features or optimization parameters")
        return self


# Human-authored files are validated through the role-specific scientific
# models below. ResolvedExperimentConfig is only their selected numerical view.

Slug = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?$")]


class BaseOutputConfig(ConfigModel):
    cache_dir: Path = Path("artifacts/cache")
    save_resolved_config: bool = True


BaseEvaluationConfig: TypeAlias = EvaluationConfig


class BaseExperimentConfig(ConfigModel):
    kind: Literal["base"]
    id: Slug
    protocol: ProtocolConfig
    data: DataConfig
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    covariates: CovariatesConfig = Field(default_factory=CovariatesConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    chronos: ChronosConfig = Field(default_factory=ChronosConfig)
    pit: PitConfig = Field(default_factory=PitConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    evaluation: BaseEvaluationConfig = Field(default_factory=BaseEvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: BaseOutputConfig = Field(default_factory=BaseOutputConfig)

    @model_validator(mode="after")
    def require_complete_static_group(self) -> Self:
        if not self.protocol.ordered_entity_ids or self.protocol.entity_count is None:
            raise ValueError("base configurations require ordered entity IDs and entity_count")
        if self.protocol.entity_count != len(self.protocol.ordered_entity_ids):
            raise ValueError("entity_count must equal the ordered entity-ID count")
        prefix = f"{self.data.entity_type}::"
        if any(not entity_id.startswith(prefix) for entity_id in self.protocol.ordered_entity_ids):
            raise ValueError("ordered entity IDs must belong to data.entity_type")
        return self


class OptimizationConfig(TrainingConfig):
    seed: NonNegativeInt = 42


class IndependentMethodConfig(ConfigModel):
    kind: Literal["method"]
    id: Slug
    family: Literal["independent"]


class StaticGaussianMethodConfig(ConfigModel):
    kind: Literal["method"]
    id: Slug
    family: Literal["static_gaussian"]
    model: StaticGaussianConfig = Field(default_factory=StaticGaussianConfig)


class ConditionalLowRankMethodConfig(ConfigModel):
    kind: Literal["method"]
    id: Slug
    family: Literal["conditional_low_rank"]
    features: FeaturesConfig
    model: ConditionalLowRankConfig
    optimization: OptimizationConfig


class SetAwareLowRankMethodConfig(ConfigModel):
    kind: Literal["method"]
    id: Slug
    family: Literal["set_aware_low_rank"]
    features: FeaturesConfig
    model: SetAwareLowRankConfig
    optimization: OptimizationConfig


KernelModelConfig: TypeAlias = ConditionalKernelConfig


class ConditionalKernelMethodConfig(ConfigModel):
    kind: Literal["method"]
    id: Slug
    family: Literal["conditional_kernel"]
    features: FeaturesConfig
    model: KernelModelConfig
    optimization: OptimizationConfig


MethodConfig: TypeAlias = Annotated[
    IndependentMethodConfig
    | StaticGaussianMethodConfig
    | ConditionalLowRankMethodConfig
    | SetAwareLowRankMethodConfig
    | ConditionalKernelMethodConfig,
    Field(discriminator="family"),
]


class CompositeBaseEntry(ConfigModel):
    id: Slug
    config: Path
    overrides: dict[str, Any] = Field(default_factory=dict)


class CompositeExperimentEntry(ConfigModel):
    id: Slug
    method: Path
    seeds: list[NonNegativeInt] = Field(default_factory=list)
    base_ids: list[Slug] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("seeds", "base_ids")
    @classmethod
    def unique_entry_values(cls, value: list[Any]) -> list[Any]:
        if len(value) != len(set(value)):
            raise ValueError("composite entry values must be unique")
        return value


class EvaluationDocumentConfig(ConfigModel):
    kind: Literal["evaluation"]
    id: Slug
    reference: Slug
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


class CompositeEvaluationEntry(ConfigModel):
    id: Slug
    config: Path
    base_ids: list[Slug] = Field(default_factory=list)
    experiment_ids: list[Slug] = Field(default_factory=list)

    @field_validator("base_ids", "experiment_ids")
    @classmethod
    def unique_selection_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("composite evaluation selections must be unique")
        return value


class CompositeReportEntry(ConfigModel):
    id: Slug
    config: Path
    evaluation_ids: list[Slug] = Field(min_length=1)

    @field_validator("evaluation_ids")
    @classmethod
    def unique_evaluation_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("composite report evaluation IDs must be unique")
        return value


class CompositeAnalysisConfig(ConfigModel):
    reference: Slug
    bootstrap_replicates: PositiveInt = 10_000
    primary_block_length: PositiveInt = 7
    sensitivity_block_lengths: list[PositiveInt] = Field(default_factory=lambda: [3, 14])

    @field_validator("sensitivity_block_lengths")
    @classmethod
    def unique_block_lengths(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("sensitivity block lengths must be unique")
        return value


class CompositeExperimentConfig(ConfigModel):
    kind: Literal["composite"]
    name: Slug
    venue: Slug
    bases: list[CompositeBaseEntry] = Field(min_length=1)
    experiments: list[CompositeExperimentEntry] = Field(min_length=1)
    evaluations: list[CompositeEvaluationEntry] = Field(min_length=1)
    reports: list[CompositeReportEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        base_ids = [entry.id for entry in self.bases]
        experiment_ids = [entry.id for entry in self.experiments]
        evaluation_ids = [entry.id for entry in self.evaluations]
        report_ids = [entry.id for entry in self.reports]
        if len(base_ids) != len(set(base_ids)) or len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("composite base and experiment IDs must be unique")
        if len(evaluation_ids) != len(set(evaluation_ids)) or len(report_ids) != len(set(report_ids)):
            raise ValueError("composite evaluation and report IDs must be unique")
        known = set(base_ids)
        for experiment in self.experiments:
            unknown = set(experiment.base_ids) - known
            if unknown:
                raise ValueError(f"experiment {experiment.id!r} references unknown bases: {sorted(unknown)}")
        known_experiments = set(experiment_ids)
        for evaluation in self.evaluations:
            if set(evaluation.base_ids) - known:
                raise ValueError(f"evaluation {evaluation.id!r} references unknown bases")
            if set(evaluation.experiment_ids) - known_experiments:
                raise ValueError(f"evaluation {evaluation.id!r} references unknown experiments")
        known_evaluations = set(evaluation_ids)
        for report in self.reports:
            if set(report.evaluation_ids) - known_evaluations:
                raise ValueError(f"report {report.id!r} references unknown evaluations")
        return self


class ReportConfig(ConfigModel):
    """Regenerate a composite report from an existing ``runs/`` directory.

    Unlike a composite, this is never hashed against fitted/evaluated cells:
    it only reads already-computed ``per_origin_metrics.parquet`` records.
    """

    kind: Literal["report"]
    run_root: Path
    metrics: list[str] = Field(default_factory=lambda: ["mean_pinball"], min_length=1)
    analysis: CompositeAnalysisConfig
    output_dir: Path | None = None

    @field_validator("metrics")
    @classmethod
    def unique_metrics(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value) or len(value) != len(set(value)):
            raise ValueError("report.metrics must be non-empty, unique metric names")
        return value


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


def _load_role_values(path: str | Path, overrides: Sequence[str]) -> dict[str, Any]:
    values = _read_yaml(Path(path), ())
    return deep_merge(values, parse_overrides(overrides)) if overrides else values


def load_base_config(path: str | Path, overrides: Sequence[str] = ()) -> BaseExperimentConfig:
    """Load one complete frozen-marginal experiment base."""

    return BaseExperimentConfig.model_validate(_load_role_values(path, overrides))


def load_method_config(path: str | Path, overrides: Sequence[str] = ()) -> MethodConfig:
    """Load exactly one strictly typed dependence-method configuration."""

    return TypeAdapter(MethodConfig).validate_python(_load_role_values(path, overrides))


def load_composite_config(path: str | Path, overrides: Sequence[str] = ()) -> CompositeExperimentConfig:
    """Load a venue-scoped explicit composite experiment declaration."""

    source = Path(path).expanduser().resolve()
    config = CompositeExperimentConfig.model_validate(_load_role_values(source, overrides))
    if source.parent.parent.name != "venues" or source.parent.parent.parent.name != "configs":
        raise ValueError("composite configurations must be stored directly under configs/venues/<venue>")
    directory_venue = source.parent.name
    if config.venue != directory_venue:
        raise ValueError(
            f"composite venue {config.venue!r} does not match its directory {directory_venue!r}"
        )
    return config


def load_evaluation_config(path: str | Path, overrides: Sequence[str] = ()) -> EvaluationDocumentConfig:
    """Load one evaluation design for a composite evaluation entry."""

    return EvaluationDocumentConfig.model_validate(_load_role_values(path, overrides))


def load_report_config(path: str | Path, overrides: Sequence[str] = ()) -> ReportConfig:
    """Load a standalone report-regeneration configuration."""

    return ReportConfig.model_validate(_load_role_values(path, overrides))


def marginal_fingerprint(values: Mapping[str, Any]) -> str:
    """Hash the normalized scientific inputs that determine a marginal cache."""

    protocol = values["protocol"]
    data = values["data"]
    chronos = values["chronos"]
    payload = {
        "protocol": {
            "full_group_only": protocol["full_group_only"],
            "ordered_entity_ids": protocol["ordered_entity_ids"],
            "entity_count": protocol["entity_count"],
        },
        "data": {key: value for key, value in data.items() if key != "local_dir"},
        "forecast": values["forecast"],
        "covariates": values["covariates"],
        "split": values["split"],
        "chronos": {
            key: value
            for key, value in chronos.items()
            if key not in {"device", "batch_size"}
        },
        "pit": {
            key: value
            for key, value in values["pit"].items()
            if key != "dependence_transform"
        },
    }
    return canonical_json_hash(payload)


def base_fingerprint(config: BaseExperimentConfig) -> str:
    """Hash only the scientific inputs that determine the frozen marginal cache."""

    return marginal_fingerprint(config.model_dump(mode="json"))


def resolve_run_config(
    base: BaseExperimentConfig,
    method: MethodConfig,
    *,
    seed: int | None = None,
) -> ResolvedExperimentConfig:
    """Adapt role-specific scientific configuration to the existing numerical pipeline."""

    values: dict[str, Any] = {
        key: value
        for key, value in base.model_dump(mode="python").items()
        if key not in {"kind", "id", "output"}
    }
    values["output"] = {
        "root_dir": "runs",
        "cache_dir": base.output.cache_dir,
        "cache_name": f"{base.id}-{base_fingerprint(base)[:12]}",
        "experiment_name": f"{base.id}_{method.id}",
        "save_resolved_config": base.output.save_resolved_config,
    }
    values["dependence"] = {"method": method.family, "model": None}
    if isinstance(method, StaticGaussianMethodConfig):
        values["dependence"]["model"] = method.model.model_dump(mode="python")
    elif isinstance(method, ConditionalLowRankMethodConfig):
        values["dependence"]["model"] = method.model.model_dump(mode="python")
    elif isinstance(method, SetAwareLowRankMethodConfig):
        values["dependence"]["model"] = method.model.model_dump(mode="python")
    elif isinstance(method, ConditionalKernelMethodConfig):
        values["dependence"]["model"] = method.model.model_dump(mode="python")
    if isinstance(
        method,
        (ConditionalLowRankMethodConfig, SetAwareLowRankMethodConfig, ConditionalKernelMethodConfig),
    ):
        optimization = method.optimization.model_dump(mode="python")
        configured_seed = int(optimization.pop("seed"))
        values["seed"] = configured_seed if seed is None else seed
        values["features"] = method.features.model_dump(mode="python")
        values["training"] = optimization
    else:
        values["seed"] = 42 if seed is None else seed
    return ResolvedExperimentConfig.model_validate(values)
