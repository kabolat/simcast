# Configuration reference

Configuration is loaded by strict Pydantic models: unknown fields, invalid
ranges, and inconsistent dimensions fail immediately. YAML files can contain
`extends` as one path or a list. Parents are merged in order, mappings merge
recursively, and scalars/lists in children replace parent values. Cycles are
rejected. `${NAME}` requires an environment variable;
`${NAME:-default}` supplies a fallback. Repeatable CLI `--set dotted.key=value`
overrides are YAML-parsed after inheritance and environment expansion.

Defaults below are code defaults; `configs/base.yaml` overrides Chronos batch
size from 16 to 256. The supplied experiments resolve through `base.yaml`.

## Root and data

| Key | Default | Meaning |
|---|---|---|
| `seed` | `42` | nonnegative global seed |
| `protocol.name` | `legacy` in code; `full_group` in `base.yaml` | declared experiment protocol |
| `protocol.full_group_only` | `false` in code; `true` in `base.yaml` | enforce complete static groups |
| `data.dataset_id` | OpenSTEF Liander repo | Hugging Face dataset ID |
| `data.revision` | `dce7fe...b7044` | required exact dataset snapshot |
| `data.local_dir` | `data/liander2024` | local snapshot root; base config honors `SIMCAST_DATA_DIR` |
| `data.entity_type` | `transformer` | one of the five supported homogeneous types |
| `data.target_column` | `load` | target column in measurement Parquet |
| `data.include_epex` | `false` | download EPEX file; not used in current covariate builder |
| `data.include_profiles` | `false` | download profiles file; not used in current covariate builder |

## Forecast protocol

| Key | Default | Meaning/constraint |
|---|---:|---|
| `forecast.timezone` | `UTC` | only supported internal timezone |
| `forecast.frequency_minutes` | 15 | positive and must divide 1,440 |
| `forecast.origin_time` | `23:45` | UTC, minute precision, aligned to frequency |
| `forecast.lookback_steps` | 672 | inclusive history length |
| `forecast.horizon_steps` | 96 | number of future leads |
| `forecast.origin_stride_steps` | 96 | step distance between origins |

## Covariates and split

| Key | Default | Meaning |
|---|---|---|
| `covariates.weather` | five named weather fields | unique nonempty columns in fixed order |
| `covariates.future_weather_source` | `vintage` | `vintage` uses latest released forecast; `oracle` uses realized future measurements |
| `covariates.calendar.enabled` | `true` | append cyclic calendar channels |
| `...include_hour` | `true` | sine/cosine fractional UTC hour |
| `...include_day_of_week` | `true` | sine/cosine weekday |
| `...include_is_weekend` | `true` | binary UTC Saturday/Sunday indicator |
| `covariates.epex.enabled` | `false` | reserved; currently not consumed |
| `covariates.profiles.enabled` | `false` | reserved; currently not consumed |
| `split.tune_fraction` | 0.80 | open interval `(0,1)`; train+validation prefix |
| `split.validation_fraction_within_tune` | 0.20 | open interval `(0,1)` in config model |
| `split.purge_overlapping_horizons` | `true` | remove leaking boundary origins |

## Chronos and PIT

| Key | Base experiment value | Meaning |
|---|---|---|
| `chronos.source_url` | official Amazon repository | clone source |
| `chronos.source_revision` | `8589...6370` | exact patched source commit |
| `chronos.model_id` | `amazon/chronos-2` | Hugging Face model ID |
| `chronos.model_revision` | `29ec...498c` | exact model artifact revision |
| `chronos.device` | `${SIMCAST_DEVICE:-cuda}` | load/inference device |
| `chronos.dtype` | `bfloat16` | one of float16/bfloat16/float32 |
| `chronos.batch_size` | 256 | entity-row batch size used by Chronos dataset |
| `chronos.cross_learning` | `false` | literal false; cannot be enabled |
| `pit.mode` | `discretized` | only implemented PIT method |
| `pit.monotone_repair` | `none` | `none` or `isotonic`; solar config uses isotonic |
| `pit.eps` | $10^{-7}$ | clamp before inverse normal, in `(0,0.5)` |

## Conditional features

| Key | Default | Meaning |
|---|---|---|
| `features.use_forecast_embedding` | `true` | include raw patch representation |
| `features.layer_normalize_embedding` | `true` | controls M2 input LayerNorm |
| `features.use_quantile_shape` | `true` | include all normalized native quantiles |
| `features.use_median` | `true` | include native median |
| `features.use_log_spread` | `true` | include log absolute q90-q10 spread |
| `features.use_within_patch_position` | `true` | include lead position in output patch |
| `features.use_location` | `true` | include latitude/longitude; locations then required |
| `features.use_entity_id_embedding` | `false` | unsupported ablation; true raises an error |
| `features.shape_eps` | $10^{-6}$ | spread denominator/log and standardizer threshold |
| `features.standardize_scalar_features` | `true` | train-only scalar standardization |

At least one feature component must remain enabled. Quantile-derived components
require native levels 0.1, 0.5, and 0.9.

## Dependence models

`dependence.method` is one of `independent`, `static_gaussian`,
`conditional_low_rank`, `set_aware_low_rank`, or `conditional_kernel`.

| Section/key | Default | Meaning |
|---|---:|---|
| `static_gaussian.shrinkage` | `ledoit_wolf` | only supported estimator |
| `static_gaussian.share_across_leads` | `false` | pool all lead vectors if true |
| `static_gaussian.jitter` | $10^{-6}$ | positive stabilization |
| `conditional_low_rank.latent_rank` | 4 | factor rank |
| `conditional_low_rank.hidden_dims` | `[256,128]` | nonempty positive MLP widths |
| `conditional_low_rank.activation` | `gelu` | only supported activation |
| `conditional_low_rank.dropout` | 0.1 | in `[0,1)` |
| `conditional_low_rank.sigma_floor` | $10^{-3}$ | positive uniqueness floor |
| `conditional_low_rank.jitter` | $10^{-6}$ | positive stabilization |
| `set_aware_low_rank.model_dim` | 128 | transformer width |
| `set_aware_low_rank.num_layers` | 2 | encoder depth |
| `set_aware_low_rank.num_heads` | 4 | attention heads; must divide model width |
| `set_aware_low_rank.latent_rank` | 4 | factor rank |
| `set_aware_low_rank.dropout` | 0.1 | transformer/head dropout |
| `set_aware_low_rank.sigma_floor` | $10^{-3}$ | uniqueness floor |
| `set_aware_low_rank.jitter` | $10^{-6}$ | stabilization |
| `conditional_kernel.hidden_dims` | `[256,128]` | embedding MLP widths |
| `conditional_kernel.embedding_dim` | 16 | learned RBF coordinate width |
| `conditional_kernel.activation` | `gelu` | only supported activation |
| `conditional_kernel.dropout` | 0.1 | MLP dropout |
| `conditional_kernel.initial_length_scale` | 1.0 | positive initial RBF scale |
| `conditional_kernel.nugget` | $10^{-3}$ | positive diagonal nugget |
| `conditional_kernel.jitter` | $10^{-6}$ | stabilization |
| `conditional_kernel.smoke_only` | `true` | if true, limit origins; full M4 config sets false |
| `conditional_kernel.smoke_max_origins` | 32 | max complete train and validation origins in smoke mode only |

## Full-group enforcement and optimization

| Key | Default | Meaning |
|---|---:|---|
| `subset_training.enabled` | `false` | legacy capability; forbidden by the full-group protocol |
| `subset_training.min_entities` | 4 | legacy-only lower bound |
| `subset_training.full_group_probability` | 0.25 | legacy-only full-group probability |
| `training.optimizer` | `adamw` | only supported optimizer |
| `training.batch_size` | 64 | complete origin-lead cases per batch |
| `training.epochs` | 100 | maximum epochs |
| `training.learning_rate` | $10^{-3}$ | AdamW learning rate |
| `training.weight_decay` | $10^{-4}$ | nonnegative AdamW decay |
| `training.gradient_clip_norm` | 1.0 | positive global norm cap |
| `training.patience` | 12 | no-improvement epochs; cannot exceed epochs |

## Sampling and evaluation

| Key | Default | Meaning/current behavior |
|---|---:|---|
| `sampling.num_samples` | 4096 | aggregate scenarios per valid case |
| `sampling.empirical_quantile_method` | `nearest` | only accepted method |
| `sampling.common_random_numbers` | `true` | documented intent; evaluator currently always uses common normals |
| `evaluation.quantile_levels` | `[.05,.10,.25,.50,.75,.90,.95]` | sorted unique report levels |
| `evaluation.interval_levels` | `[.50,.80,.90]` | sorted unique central coverages |
| `evaluation.energy_score` | `true` | jointly controls whether both joint scores are computed |
| `evaluation.variogram_score` | `true` | jointly controls whether both joint scores are computed |
| `evaluation.variogram_power` | 0.5 | in `(0,2]` |
| `evaluation.report_by_lead` | `true` | retained config field; lead table is currently always written |
| `evaluation.scenario_batch_size` | 16 | cases sampled together |
| `evaluation.joint_score_num_samples` | 512 | cap for entity-level scores |
| `evaluation.variable_k_sizes` | `[]` | legacy-only diagnostic; nonempty is forbidden by the full-group protocol |

Current code computes both Energy and Variogram Scores when either score flag
is true; it computes neither only when both are false. The individual flags
should therefore not be interpreted as independent switches.

For `protocol.name: full_group`, validation requires
`full_group_only: true`, `subset_training.enabled: false`, and an empty
`evaluation.variable_k_sizes`. Attempts to override either entity-selection
setting fail before training or evaluation starts.

## Runtime and output

| Key | Default | Meaning |
|---|---|---|
| `runtime.deterministic` | `true` | request deterministic PyTorch algorithms with warnings |
| `runtime.num_workers` | 0 | validated and recorded; current trainer does not pass it to DataLoader |
| `runtime.log_level` | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `output.root_dir` | `runs` | default model/evaluation root |
| `output.cache_dir` | `artifacts/cache` | default PIT-library root |
| `output.cache_name` | null | defaults to `liander2024_<entity_type>` |
| `output.experiment_name` | null | defaults to entity type in high-level runner |
| `output.save_resolved_config` | `true` | persist replayable config |

The currently passive settings called out above are documented so a researcher
does not mistake recorded intent for implemented behavior.
