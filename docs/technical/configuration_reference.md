# Configuration reference

## 1. Why the configuration has three roles

A scientific comparison needs to distinguish what is held fixed from what is
varied. Simcast therefore validates three YAML document kinds.

| Kind | Statistical role | Contents |
|---|---|---|
| `base` | defines the common marginal experiment | data, $\mathcal E_g$, information set, origins, Chronos, PIT, sampling, estimands |
| `method` | defines one dependence hypothesis | exactly one of M0--M4 and only its meaningful parameters |
| `composite` | defines a family of comparisons | explicit bases, method variants, repetitions, reference, uncertainty analysis, venue |

Begin with one supplied base, method, or composite file; the tables then help
you understand or deliberately change a particular field. Each table states the
YAML type, admissible values, default (when one exists), and scientific role. A
value marked **required** has no default. Practical YAML composition—inheritance,
local environment values, and one-time command-line changes—is explained later
in the [user guide](usage_guide.md#12-advanced-configuration-and-temporary-overrides).

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

| Key | YAML type and admissible values | Default / supplied value | Scientific meaning |
|---|---|---|---|
| `kind` | literal string `base` | **required** | declares a common marginal experiment |
| `id` | safe slug: lowercase letters, digits, `_`, `-` | **required** | human-readable base identity; not a cache-compatibility claim |
| `protocol.name` | literal string `full_group` | `full_group` | declares the static complete-group protocol |
| `protocol.full_group_only` | literal Boolean `true` | `true` | forbids entity subsets and variable cardinality |
| `protocol.ordered_entity_ids` | non-empty list of unique strings | **required** | ordered definition of $\mathcal E_g$ |
| `protocol.entity_count` | positive integer | **required** | $K_g$; must equal the list length |
| `data.dataset_id` | string | Liander dataset identifier | population source |
| `data.revision` | string | pinned Git revision | immutable data version |
| `data.local_dir` | path string | `${SIMCAST_DATA_DIR:-data/liander2024}` | local files; deliberately excluded from the scientific fingerprint |
| `data.entity_type` | one of `transformer`, `solar_park`, `wind_park`, `mv_feeder`, `station_installation` | `transformer` | homogeneous physical group $g$; every ID must begin with this type prefix |
| `data.target_column` | string | `load` | measured target $y_{k,t}$ |
| `data.include_epex` | Boolean | `false` | include optional EPEX files; currently not used as a covariate |
| `data.include_profiles` | Boolean | `false` | include optional profile files; currently not used as a covariate |

Example: transformer has $K_g=15$ in its supplied base; solar and wind each
have $K_g=5$. The group is invalid at $(i,\tau)$ if any member is invalid.

### 2.2 Forecast instances and information set

| Key | YAML type and admissible values | Default / supplied value | Meaning |
|---|---|---|---|
| `forecast.timezone` | literal string `UTC` | `UTC` | time coordinate |
| `forecast.frequency_minutes` | positive integer dividing 1440 | `15` | sampling interval $\Delta$ in minutes |
| `forecast.origin_time` | `HH:MM` time aligned with `frequency_minutes` | `23:45` | daily phase of $t^{(i)}$ |
| `forecast.lookback_steps` | positive integer | `672` | observed history length $L$ |
| `forecast.horizon_steps` | positive integer | `96` | number of leads $H$ |
| `forecast.origin_stride_steps` | positive integer | `96` | separation of candidate origins, in steps |
| `covariates.weather` | list of unique non-empty strings | five supplied weather fields | ordered weather variables supplied to Chronos |
| `covariates.future_weather_source` | `vintage` or `oracle` | `vintage` | `vintage`: latest forecast available at $t^{(i)}$; `oracle`: realized future weather, hence non-operational |
| `covariates.calendar.enabled` | Boolean | `true` | include calendar covariates as a block |
| `covariates.calendar.include_hour` | Boolean | `true` | hour-of-day coordinates |
| `covariates.calendar.include_day_of_week` | Boolean | `true` | weekday coordinates |
| `covariates.calendar.include_is_weekend` | Boolean | `true` | binary Saturday/Sunday indicator |
| `covariates.epex.enabled` | Boolean | `false` | request optional EPEX covariate input |
| `covariates.profiles.enabled` | Boolean | `false` | request optional profile covariate input |

For `vintage`, future weather at physical time $t^{(i)}+\tau\Delta$ is selected
only from a forecast vintage available by $t^{(i)}$. Target measurements are
assumed available when measured, but future targets are never included in
$\mathcal I^{(i)}$. See [Chapter 2](../scientific/02_data_and_information_set.md).

### 2.3 Chronological partitions

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `split.tune_fraction` | real number strictly between 0 and 1 | `0.80` | chronological train-plus-validation fraction |
| `split.validation_fraction_within_tune` | real number strictly between 0 and 1 | `0.20` | final fraction of the tuning prefix assigned to validation |
| `split.purge_overlapping_horizons` | Boolean | `true` | remove boundary origins whose realized horizons overlap partitions |

Changing a split changes the sample used to estimate PIT dependence and
therefore changes the base fingerprint.

### 2.4 Frozen Chronos forecast

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `chronos.source_url` | non-empty string URL | pinned Chronos source URL | source implementation identity |
| `chronos.source_revision` | non-empty string | pinned commit | source revision used by the local setup |
| `chronos.model_id` | non-empty string | `amazon/chronos-2` | pretrained model identity |
| `chronos.model_revision` | non-empty string | pinned commit | pretrained parameter revision |
| `chronos.device` | non-empty string accepted by the backend | `cuda` | numerical device, not a scientific change |
| `chronos.dtype` | `float16`, `bfloat16`, or `float32` | `bfloat16` | inference arithmetic type |
| `chronos.batch_size` | positive integer | `16` | inference batch size |
| `chronos.cross_learning` | literal Boolean `false` | `false` | entities remain separate Chronos tasks |

Source and model revisions determine the statistical forecast and enter the
fingerprint. Device and batch size are operational and do not. Cross-entity
attention in the foundation model is forbidden by the literal `false` field.

### 2.5 Finite-quantile marginal law and PIT

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `pit.mode` | `discretized` or `linear_interpolation` | `discretized` | fixes both historical PIT construction and scenario projection; `discretized` uses $Q+1$ cell midpoints and nearest native values, while `linear_interpolation` is piecewise linear between native quantiles with constant boundary segments |
| `pit.monotone_repair` | `none` or `isotonic` | `none` | `none` invalidates a crossed row; `isotonic` applies least-squares monotone repair before PIT construction and projection |
| `pit.dependence_transform` | `nominal_cells` or `training_frequency` | `nominal_cells` | `nominal_cells` Gaussianizes declared cell midpoints; `training_frequency` replaces each midpoint by its training-only empirical cell-mass midpoint before Gaussianization |
| `pit.eps` | real number strictly between 0 and 0.5 | `1.0e-7` | clamp used before $\Phi^{-1}$ |

With `discretized`, the predictive thresholds divide the real line into $Q+1$
cells and an observation receives its cell's probability midpoint. With
`linear_interpolation`, probabilities and values are linearly related between
adjacent native quantiles; the map remains constant below $q_1$ and above
$q_Q$, so it does not extrapolate unsupported tails. At endpoint or tied-value
atoms, the forward PIT uses the atom's probability midpoint. Isotonic repair,
when declared, is applied before either map and the same repaired grid is used
for historical PITs and scenario projection.

`training_frequency` is admissible only with `pit.mode: discretized`, because
it estimates masses for the named $Q+1$ cells. The configuration validator
rejects it with `linear_interpolation`. Changing `pit.mode` changes the
scientific marginal fingerprint and therefore selects a different cache. See
[Chapter 3](../scientific/03_chronos_and_pit.md) for equations and numerical examples.

### 2.6 Evaluation settings

Sampling and score settings are no longer valid base fields. They belong to
the optional `kind: evaluation` documents referenced by composite evaluation
entries (or run standalone against an existing run; see
[usage guide §14](usage_guide.md#14-re-evaluating-a-completed-run)). The
legacy table below is retained only as a field reference for those evaluation
documents; it must not be copied into a base.

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `sampling.num_samples` | positive integer | `4096` | aggregate scenarios $M$ per valid case |
| `sampling.evaluation_seed` | non-negative integer | `2027` | scenario randomness |
| `sampling.common_random_numbers` | Boolean | `true` | case-keyed Gaussian draws shared across methods |
| `sampling.empirical_quantile_method` | literal string `nearest` | `nearest` | order-statistic convention for aggregate quantiles after Monte Carlo sampling; it does not control entity-level marginal projection |
| `evaluation.quantile_levels` | sorted, unique non-empty list in $(0,1)$ | `[.05,.10,.25,.50,.75,.90,.95]` | reported aggregate quantiles |
| `evaluation.interval_levels` | sorted, unique non-empty list in $(0,1)$ | `[.50,.80,.90]` | central interval levels |
| `evaluation.energy_score` | Boolean | `true` | evaluate empirical all-pairs Energy Score |
| `evaluation.variogram_score` | Boolean | `true` | evaluate Variogram Score |
| `evaluation.variogram_power` | real number in $(0,2]$ | `0.5` | pairwise-difference exponent |
| `evaluation.report_by_lead` | Boolean | `true` | retain lead-specific summaries |
| `evaluation.scenario_batch_size` | positive integer | `16` | scenario-generation batch size |
| `evaluation.joint_score_num_samples` | positive integer | `512` | selected joint-ensemble size for Energy/Variogram Scores |
| `runtime.deterministic` | Boolean | `true` | request deterministic numerical execution |
| `runtime.num_workers` | non-negative integer | `0` | data-loading worker count |
| `runtime.log_level` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` | `INFO` | console verbosity only |
| `output.cache_dir` | path string | `artifacts/cache` | cache root; not part of the marginal fingerprint |
| `output.save_resolved_config` | Boolean | `true` | save the resolved scientific configuration with the cache/run |

These settings do not alter cached Chronos quantiles. They define Monte Carlo
resolution and reported estimands and therefore belong to the common base,
not to a dependence method. The score definitions are in
[Chapter 5](../scientific/05_training_sampling_scoring.md), and the way their
records support a report is in
[Chapter 7](../scientific/07_experiments_and_results.md).

## 3. Method configuration

### 3.1 M0: independence

```yaml
kind: method
id: m0
family: independent
```

No feature, model, or optimization section is admissible because
$R_{g,\tau}^{(i)}=I_{K_g}$ is completely specified. `kind` is the literal
string `method`; `id` is a required safe slug; and `family` is the literal
string `independent`. These are the only valid M0 fields.

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

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `kind` | literal string `method` | **required** | declares one dependence hypothesis |
| `id` | safe slug | **required** | method-file identity |
| `family` | literal string `static_gaussian` | **required** | selects M1 |
| `model.shrinkage` | literal string `ledoit_wolf` | `ledoit_wolf` | covariance shrinkage estimator |
| `model.share_across_leads` | Boolean | `false` | `false`: one training correlation per lead; `true`: pool all valid leads |
| `model.jitter` | strictly positive real | `1.0e-6` | diagonal numerical stabilization |

M1 has no gradient optimization or conditioning features.

### 3.3 Features for M2--M4

Only conditional methods admit `features`:

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `use_forecast_embedding` | Boolean | `true` | Chronos output-patch representation |
| `layer_normalize_embedding` | Boolean | `true` | LayerNorm before downstream prediction |
| `use_quantile_shape` | Boolean | `true` | normalized native quantile shape |
| `use_median` | Boolean | `true` | native median level |
| `use_log_spread` | Boolean | `true` | log absolute q90--q10 spread |
| `use_within_patch_position` | Boolean | `true` | lead position inside an output patch |
| `use_location` | Boolean | `true` | latitude and longitude |
| `use_entity_id_embedding` | literal Boolean `false` | `false` | explicitly unsupported entity-ID embedding |
| `shape_eps` | strictly positive real | `1.0e-6` | numerical threshold in shape/spread construction |
| `standardize_scalar_features` | Boolean | `true` | estimate scalar-feature normalization from training origins only |

These features form $x_{k,\tau}^{(i)}$. A feature ablation is expressed by a
repeated composite entry with an explicit override; it is not hidden in the
base.

### 3.4 M2 model

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `family` | literal string `conditional_low_rank` | **required** | selects M2 |
| `model.latent_rank` | positive integer | `4` | factor rank $r$ |
| `model.hidden_dims` | non-empty list of positive integers | `[256,128]` | widths of the shared entity-wise network |
| `model.activation` | literal string `gelu` | `gelu` | hidden nonlinearity |
| `model.dropout` | real number in $[0,1)$ | `0.1` | training-time dropout probability |
| `model.sigma_floor` | strictly positive real | `1.0e-3` | lower bound for uniqueness standard deviations |
| `model.jitter` | strictly positive real | `1.0e-6` | final correlation stabilization |

See [Chapter 4](../scientific/04_dependence_models.md) for
$\Sigma=\Lambda\Lambda^\top+\operatorname{diag}(\sigma^2)$ and correlation
normalization.

### 3.5 M3 model

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `family` | literal string `set_aware_low_rank` | **required** | selects M3 |
| `model.model_dim` | positive integer divisible by `num_heads` | `128` | attention representation dimension |
| `model.num_layers` | positive integer | `2` | entity self-attention layers |
| `model.num_heads` | positive integer | `4` | attention heads |
| `model.latent_rank` | positive integer | `4` | downstream factor rank |
| `model.dropout` | real number in $[0,1)$ | `0.1` | training-time dropout probability |
| `model.sigma_floor` | strictly positive real | `1.0e-3` | uniqueness standard-deviation floor |
| `model.jitter` | strictly positive real | `1.0e-6` | final correlation stabilization |

No positional entity index is supplied, preserving permutation equivariance.

### 3.6 M4 model

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `family` | literal string `conditional_kernel` | **required** | selects M4 |
| `model.hidden_dims` | non-empty list of positive integers | `[256,128]` | network widths mapping features to kernel coordinates |
| `model.embedding_dim` | positive integer | `16` | dimension of learned coordinates $h_k$ |
| `model.activation` | literal string `gelu` | `gelu` | hidden nonlinearity |
| `model.dropout` | real number in $[0,1)$ | `0.1` | training-time dropout probability |
| `model.initial_length_scale` | strictly positive real | `1.0` | initialization of $\ell$ in $\exp(-\|h_a-h_b\|^2/(2\ell^2))$ |
| `model.nugget` | strictly positive real | `1.0e-3` | kernel diagonal stabilization |
| `model.jitter` | strictly positive real | `1.0e-6` | final correlation stabilization |

M4 has no reduced-data flag, origin cap, or special budget.

### 3.7 Optimization for M2--M4

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `optimization.seed` | non-negative integer | `42` | initialization and batch order |
| `optimization.optimizer` | literal string `adamw` | `adamw` | optimization algorithm |
| `optimization.batch_size` | positive integer | `64` | complete $(i,\tau)$ vectors per batch |
| `optimization.epochs` | positive integer | `100` | maximum fitting passes |
| `optimization.patience` | positive integer no greater than `epochs` | `12` | validation pseudo-NLL stopping patience |
| `optimization.learning_rate` | strictly positive real | `1.0e-3` | AdamW step size |
| `optimization.weight_decay` | non-negative real | `1.0e-4` | AdamW decay coefficient |
| `optimization.gradient_clip_norm` | strictly positive real | `1.0` | gradient-norm bound |

All three trainable methods use the same semantics. A composite may override
budgets explicitly and validation then applies to that method type.

## 4. Composite configuration

A composite has two compulsory sections (`bases`, `methods`) and two optional
ones (`evaluations`, `reports`). Evaluations depend on bases and methods;
reports depend on evaluations. Declaring `reports` without `evaluations` fails
validation rather than silently doing nothing.

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `kind` | literal string `composite` | **required** | declares an explicit collection of comparisons |
| `name` | safe slug | **required** | composite identity within the venue |
| `venue` | safe slug, matching `configs/venues/<venue>/` | **required** | reproducible research workspace identity |
| `bases` | non-empty list of base entries | **required** | named base configurations available to the composite |
| `bases[].id` | safe slug, unique within `bases` | **required** | local base reference |
| `bases[].config` | relative or absolute YAML path | **required** | base configuration file |
| `bases[].overrides` | mapping of valid base fields to YAML values | `{}` | explicit base variant; invalid fields fail validation |
| `methods` | non-empty list of method entries | **required** | explicitly declared method fits |
| `methods[].id` | safe slug, unique within `methods` | **required** | local method-entry reference |
| `methods[].method` | relative or absolute YAML path | **required** | one M0--M4 method configuration |
| `methods[].seeds` | list of unique non-negative integers | `[]` | repetitions for conditional methods; forbidden for deterministic M0/M1 entries |
| `methods[].base_ids` | list of unique existing base-entry IDs | `[]` | restrict this method entry to selected bases; an empty list means every listed base |
| `methods[].overrides` | mapping of valid fields for that method family | `{}` | explicit method variant; cross-method fields fail validation |
| `evaluations` | list of evaluation entries | `[]` | optional post-fit evaluation designs, run immediately after fitting |
| `evaluations[].id` | safe slug, unique within `evaluations` | **required** | evaluation identity |
| `evaluations[].config` | evaluation YAML path | **required** | sampling, score, and evaluation settings |
| `evaluations[].base_ids` | list of known base IDs | `[]` | bases to evaluate; empty means every base |
| `evaluations[].method_ids` | list of known method IDs | `[]` | fitted methods to evaluate independently; empty means every fitted method |
| `reports` | list of report entries | `[]` | optional explicit post-evaluation reports |
| `reports[].id` | safe slug, unique within `reports` | **required** | report identity |
| `reports[].config` | report YAML path | **required** | report metrics and uncertainty settings |
| `reports[].evaluation_ids` | non-empty list of known evaluation IDs | **required** | evaluations selected for the report |

`bases` assigns local IDs to base files and optional valid base overrides.
`methods` assigns distinct IDs to method files, optional base selections,
optional conditional seeds, and role-valid method overrides. `evaluations`,
when declared, assigns post-fit evaluation designs to selected base/method
combinations and runs immediately after fitting completes in the same
`simcast.cli.run_composite` invocation. `reports` only declares which report
designs refer to which evaluations; reports are never generated automatically
and must be run explicitly with `simcast.cli.report_composite`.

Deterministic M0/M1 entries cannot declare repeated seeds. Conditional entries
without `seeds` use their method file's optimization seed. Each evaluation
document names its deterministic M0/M1-family reference, which must resolve to
exactly one deterministic fit per base. Bootstrap replicate count and block
lengths belong to a report, not to fitting or evaluation. A run's fits are
independent of any `sampling`/`evaluation` settings, so a new evaluation
design can be applied to an existing run without retraining, using
`simcast.cli.evaluate_composite` (see
[usage guide §14](usage_guide.md#14-re-evaluating-a-completed-run)).
The scientific purpose of this declaration and the resulting claim discipline
are explained in [Chapters 6](../scientific/06_scientific_workflow.md) and
[7](../scientific/07_experiments_and_results.md); the executable command is in
the [usage guide](usage_guide.md#6-composite-experiment).

Example: five bases, two deterministic entries, and three conditional entries
with ten seeds expand to $5+5+50+50+50=160$ fits.

## 5. Venue and path validity

A composite at `configs/venues/<venue>/study.yaml` must declare the same safe
slug in `venue`. Every artifact of a run nests under one output root:

```text
runs/<venue>/<composite>/<run-id>/<base-id>/<method-id>/<seed-label>/
runs/<venue>/<composite>/<run-id>/evaluations/<evaluation-id>/<base-id>/<method-id>/<seed-label>/
runs/<venue>/<composite>/<run-id>/reports/<report-id>/
```

This rule makes a venue cloneable and prevents path escape. `run-id` is either
a UTC timestamp or an explicit lowercase safe slug. `seed-label` is
`deterministic` for M0/M1 fits or `seed_<N>` for a repeated conditional fit.

## 6. Cache fingerprint

See [usage guide §8](usage_guide.md#8-cache-compatibility) for the SHA-256
definition and compatibility rule; this section lists only its field-level
consequences. `base_fingerprint` hashes precisely the base fields that determine the frozen
marginal/PIT record. It excludes the local data path, Chronos device and batch
size, sampling, evaluation, runtime, method, composite, venue, and reporting.
`locate_compatible_cache` verifies metadata, not a user-supplied cache label.

Example: changing `evaluation.joint_score_num_samples` reuses the same marginal
cache. Changing `pit.monotone_repair`, ordered entity IDs, weather source,
forecast horizon, or Chronos model revision requires a different cache.

## 7. Evaluation and report configuration

A `kind: evaluation` document (§2.6) can also be run standalone against an
already-completed run, without a composite `evaluations` entry, using
`simcast.cli.evaluate_composite`. In that mode its own `run_root`, `base_ids`,
and `method_ids` fields select the target run and fits directly:

```yaml
kind: evaluation
id: variogram_power_1
evaluation:
  variogram_power: 1.0
run_root: runs/lab/quick_shot/2026-09-16_093812
```

A `kind: report` document regenerates a report from an already-completed run's
recorded evaluations, without re-fitting or re-evaluating anything. See
[usage guide §15](usage_guide.md#15-regenerating-or-customizing-a-report) for
when and how to use it; this section lists its fields.

```yaml
kind: report
run_root: runs/powertech2027/main/2026-09-16_093812
reference: m0
metrics: [mean_pinball, crps]
evaluation_ids: [standard]
analysis:
  bootstrap_replicates: 10000
  primary_block_length: 7
```

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `kind` | literal string `report` | **required** | declares a standalone report-regeneration document |
| `run_root` | path to an existing `runs/<venue>/<composite>/<run-id>/` directory | `null` (must be given here or via `--run-root`) | source of already-computed evaluation records |
| `metrics` | non-empty list of unique column names from `per_origin_metrics.parquet` | `[mean_pinball]` | which metrics get a summary, paired-effect table, and figure |
| `evaluation_ids` | list of known evaluation IDs recorded in the run's manifest | `[]` | which recorded evaluations to report on; empty means every evaluation in the manifest |
| `analysis` | a composite `analysis` block (§4) | **required** | bootstrap replicates and block lengths for this report only |
| `output_dir` | path | `<run_root>/reports/<report-id>` | where the regenerated report is written |

`metrics` accepts any column already present in `per_origin_metrics.parquet`
(for example `mean_pinball`, `crps`, `weighted_interval_score`,
`coverage_0.9`, `energy_score`, `variogram_score`); an unknown name fails
validation at report time rather than silently being ignored.
