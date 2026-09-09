# Scientific configuration reference

## 1. Why the configuration has three roles

A scientific comparison needs to distinguish what is held fixed from what is
varied. Simcast therefore validates three YAML document kinds.

| Kind | Statistical role | Contents |
|---|---|---|
| `base` | defines the common marginal experiment | data, $\mathcal E_g$, information set, origins, Chronos, PIT, sampling, estimands |
| `method` | defines one dependence hypothesis | exactly one of M0--M4 and only its meaningful parameters |
| `composite` | defines a family of comparisons | explicit bases, method variants, repetitions, reference, uncertainty analysis, venue |

All models forbid unknown fields. `extends` accepts one relative path or an
ordered list of parents; nested mappings merge recursively and lists replace.
Inheritance cycles fail. Environment expressions use `${NAME}` or
`${NAME:-default}`. Loader overrides use dotted YAML values.

Implementation: the three public loaders are `load_base_config`,
`load_method_config`, and `load_composite_config` in `simcast.config`.

## 2. Base configuration

### 2.1 Complete group and data

```yaml
kind: base
id: transformer
protocol:
  name: full_group
  full_group_only: true
  ordered_entity_ids: [transformer::A, transformer::B]
  entity_count: 2
data:
  entity_type: transformer
```

`ordered_entity_ids` defines
$\mathcal E_g=[k_1,\ldots,k_{K_g}]$ including its order, and `entity_count`
must equal $K_g$. Every ID must have the selected homogeneous entity-type
prefix. No entity-selection or variable-cardinality field exists.

| Key | Supplied value | Scientific meaning |
|---|---|---|
| `data.dataset_id` | pinned Liander repository | population source |
| `data.revision` | exact commit | immutable data version |
| `data.local_dir` | `${SIMCAST_DATA_DIR:-data/liander2024}` | operational location; not part of the scientific fingerprint |
| `data.entity_type` | one homogeneous type | physical group $g$ |
| `data.target_column` | `load` | observation $y_{k,t}$ |
| `data.include_epex`, `include_profiles` | `false` | optional files, currently excluded |

Example: transformer has $K_g=15$ in its supplied base; solar and wind each
have $K_g=5$. The group is invalid at $(i,\tau)$ if any member is invalid.

### 2.2 Forecast instances and information set

| Key | Supplied value | Meaning |
|---|---:|---|
| `forecast.timezone` | `UTC` | time coordinate |
| `forecast.frequency_minutes` | 15 | $\Delta$ |
| `forecast.origin_time` | `23:45` | daily phase of $t^{(i)}$ |
| `forecast.lookback_steps` | 672 | observed history length $L$ |
| `forecast.horizon_steps` | 96 | number of leads $H$ |
| `forecast.origin_stride_steps` | 96 | distance between candidate origins |
| `covariates.weather` | five ordered fields | weather variables supplied to Chronos |
| `covariates.future_weather_source` | `vintage` | latest forecast with release time no later than $t^{(i)}$ |
| same | `oracle` | realized future weather; an explicitly non-operational information set |
| `covariates.calendar.include_hour` | `true` | cyclic hour coordinates |
| `covariates.calendar.include_day_of_week` | `true` | cyclic weekday coordinates |
| `covariates.calendar.include_is_weekend` | `true` | binary Saturday/Sunday indicator |

For `vintage`, future weather at physical time $t^{(i)}+\tau\Delta$ is selected
only from a forecast vintage available by $t^{(i)}$. Target measurements are
assumed available when measured, but future targets are never included in
$\mathcal I^{(i)}$. See [Chapter 2](02_data_and_information_set.md).

### 2.3 Chronological partitions

`split.tune_fraction: 0.80` forms the chronological train-plus-validation
prefix. `validation_fraction_within_tune: 0.20` assigns its final portion to
validation. `purge_overlapping_horizons: true` removes boundary origins if
their realized forecast horizons overlap across partitions.

Changing a split changes the sample used to estimate PIT dependence and
therefore changes the base fingerprint.

### 2.4 Frozen Chronos forecast

| Key | Meaning |
|---|---|
| `chronos.source_url`, `source_revision` | exact source implementation |
| `chronos.model_id`, `model_revision` | exact pretrained parameters |
| `chronos.device`, `dtype`, `batch_size` | numerical execution |
| `chronos.cross_learning: false` | each physical entity is forecast separately |

Source and model revisions determine the statistical forecast and enter the
fingerprint. Device and batch size are operational and do not. Cross-entity
attention in the foundation model is forbidden by the literal `false` field.

### 2.5 Finite pseudo-PIT

| Key | Meaning |
|---|---|
| `pit.mode: discretized` | deterministic $Q+1$-cell pseudo-PIT; no CDF interpolation |
| `pit.monotone_repair` | `none` or deterministic `isotonic` repair |
| `pit.dependence_transform` | nominal cells or a training-only cell-frequency sensitivity |
| `pit.eps` | probability clamp before $\Phi^{-1}$ |

If native levels are $q_1<\cdots<q_Q$, the predictive value thresholds divide
the real line into $Q+1$ cells. An observation receives the midpoint of its
cell's probability interval. Solar declares `isotonic`; its repaired grid is
used consistently for historical cells and scenario projection. See
[Chapter 3](03_chronos_and_pit.md) for equations and numerical examples.

### 2.6 Sampling and estimands

| Key | Supplied value | Meaning |
|---|---:|---|
| `sampling.num_samples` | 4096 | aggregate scenarios $M$ per valid case |
| `sampling.evaluation_seed` | 2027 | scenario randomness |
| `sampling.common_random_numbers` | `true` | case-keyed Gaussian draws shared across methods |
| `sampling.empirical_quantile_method` | `nearest` | finite ensemble order statistic |
| `evaluation.quantile_levels` | seven probabilities | reported aggregate quantiles |
| `evaluation.interval_levels` | 0.50, 0.80, 0.90 | central interval levels |
| `evaluation.joint_score_num_samples` | 512 | selected ensemble size for joint scores |
| `evaluation.energy_score` | `true` | empirical all-pairs Energy Score |
| `evaluation.variogram_score` | `true` | Variogram Score |
| `evaluation.variogram_power` | 0.5 | pairwise-difference exponent |

These settings do not alter cached Chronos quantiles. They define Monte Carlo
resolution and reported estimands and therefore belong to the common base,
not to a dependence method.

## 3. Method configuration

### 3.1 M0: independence

```yaml
kind: method
id: m0
family: independent
```

No feature, model, or optimization section is admissible because
$R_{g,\tau}^{(i)}=I_{K_g}$ is completely specified.

### 3.2 M1: static Gaussian copula

```yaml
kind: method
id: m1
family: static_gaussian
model:
  shrinkage: ledoit_wolf
  share_across_leads: false
  jitter: 1.0e-6
```

`share_across_leads: false` estimates one training correlation per lead;
`true` pools all valid leads. Ledoit--Wolf shrinkage and positive jitter
stabilize the estimate. M1 has no gradient optimization or conditioning
features.

### 3.3 Features for M2--M4

Only conditional methods admit `features`:

| Key | Meaning |
|---|---|
| `use_forecast_embedding` | Chronos output-patch representation |
| `layer_normalize_embedding` | LayerNorm before downstream prediction |
| `use_quantile_shape` | normalized native quantile shape |
| `use_median` | native median level |
| `use_log_spread` | log absolute q90--q10 spread |
| `use_within_patch_position` | lead position inside an output patch |
| `use_location` | latitude and longitude |
| `use_entity_id_embedding` | currently unsupported and fixed false |
| `shape_eps` | numerical threshold in shape/spread construction |
| `standardize_scalar_features` | fit scalar normalization on training origins only |

These features form $x_{k,\tau}^{(i)}$. A feature ablation is expressed by a
repeated composite entry with an explicit override; it is not hidden in the
base.

### 3.4 M2 model

`model.latent_rank` is $r$;
`hidden_dims`, `activation`, and `dropout` specify the shared entity-wise map;
`sigma_floor` lower-bounds uniqueness; and `jitter` stabilizes the final
correlation. See [Chapter 4](04_dependence_models.md) for
$\Sigma=\Lambda\Lambda^\top+\operatorname{diag}(\sigma^2)$ and its
correlation normalization.

### 3.5 M3 model

`model_dim`, `num_layers`, and `num_heads` specify position-free entity
self-attention. `model_dim` must be divisible by `num_heads`. `latent_rank`,
`dropout`, `sigma_floor`, and `jitter` define the subsequent low-rank map. No
positional entity index is supplied, preserving permutation equivariance.

### 3.6 M4 model

`hidden_dims` and `embedding_dim` define the map to learned coordinates $h_k$.
`initial_length_scale` initializes $\ell$ in
$\exp(-\|h_a-h_b\|^2/(2\ell^2))$; `nugget` and `jitter` stabilize the
correlation. M4 has no reduced-data flag, origin cap, or special budget.

### 3.7 Optimization for M2--M4

| Key | Supplied value | Meaning |
|---|---:|---|
| `seed` | 42 in standalone files | initialization and batch order |
| `optimizer` | `adamw` | optimizer family |
| `batch_size` | 64 | complete $(i,\tau)$ vectors per batch |
| `epochs` | 100 | maximum passes |
| `patience` | 12 | validation pseudo-NLL stopping patience |
| `learning_rate` | $10^{-3}$ | step size |
| `weight_decay` | $10^{-4}$ | AdamW decay |
| `gradient_clip_norm` | 1.0 | gradient norm bound |

All three trainable methods use the same semantics. A composite may override
budgets explicitly and validation then applies to that method type.

## 4. Composite configuration

`bases` assigns local IDs to base files and optional valid base overrides.
`experiments` assigns distinct IDs to method files, optional base selections,
optional conditional seeds, and role-valid method overrides. Expansion is the
literal nested sequence of each entry's selected bases and seeds.

Deterministic M0/M1 entries cannot declare repeated seeds. Conditional entries
without `seeds` use their method file's optimization seed. `analysis.reference`
must name a deterministic M0 or M1 entry when executed. Bootstrap replicate
count and block lengths define the paired origin-level uncertainty calculation.

Example: five bases, two deterministic entries, and three conditional entries
with ten seeds expand to $5+5+50+50+50=160$ cells.

## 5. Venue and path validity

A composite at `configs/venues/<venue>/study.yaml` must declare the same safe
slug in `venue`. The output roots are fixed:

```text
runs/<venue>/<composite>/<run-id>/
reports/<venue>/<composite>/<run-id>/
```

This rule makes a venue cloneable and prevents path escape. `run-id` is either
a UTC timestamp or an explicit lowercase safe slug.

## 6. Cache fingerprint

`base_fingerprint` hashes precisely the base fields that determine the frozen
marginal/PIT record. It excludes the local data path, Chronos device and batch
size, sampling, evaluation, runtime, method, composite, venue, and reporting.
`locate_compatible_cache` verifies metadata, not a user-supplied cache label.

Example: changing `evaluation.joint_score_num_samples` reuses the same marginal
cache. Changing `pit.monotone_repair`, ordered entity IDs, weather source,
forecast horizon, or Chronos model revision requires a different cache.
