# Configuration reference

## 1. Document kinds

A scientific comparison needs to distinguish what is held fixed from what is
varied, and both from how results are scored and compared. Simcast therefore
validates five YAML document kinds.

| Kind | Statistical role | Contents | Section |
|---|---|---|---|
| `base` | defines the common marginal experiment | data, $\mathcal E_g$, information set, origins, split, Chronos, PIT | §2 |
| `method` | defines one dependence hypothesis | exactly one of M0--M4 and only its meaningful parameters | §3 |
| `composite` | defines a declared study | venue, bases, method variants, seeds, optional evaluation and report entries | §4 |
| `evaluation` | defines one scoring design | sampling, metrics, cross-entity statistic, quantile/interval levels, figures | §7.1 |
| `report` | defines one cross-method comparison | reference method, presented metrics, bootstrap design | §7.2 |

Begin with a supplied file; the tables then help you understand or
deliberately change a particular field. Each table states the YAML type,
admissible values, default (when one exists), and scientific role. A value
marked **required** has no default. Unknown fields are errors. Inheritance,
environment values, and temporary command-line changes are explained in the
[usage guide](usage_guide.md#12-advanced-configuration-and-temporary-overrides).

Implementation: `load_base_config`, `load_method_config`,
`load_composite_config`, `load_evaluation_config`, and `load_report_config` in
`simcast.config`.

## 2. Base configuration

### 2.1 Complete group and data

```yaml
kind: base
id: transformer
protocol:
  ordered_entity_ids: [transformer::A, transformer::B]
  entity_count: 2
data:
  entity_type: transformer
```

`ordered_entity_ids` defines
$\mathcal E_g=[k_1,\ldots,k_{K_g}]$ including its order, and `entity_count`
must equal $K_g$. Every ID must have the selected homogeneous entity-type
prefix. Every fit and evaluation uses the complete group.

| Key | YAML type and admissible values | Default / supplied value | Scientific meaning |
|---|---|---|---|
| `kind` | literal string `base` | **required** | declares a common marginal experiment |
| `id` | safe slug: lowercase letters, digits, `_`, `-` | **required** | human-readable base identity; not a cache-compatibility claim |
| `protocol.ordered_entity_ids` | non-empty list of unique strings | **required** | ordered definition of $\mathcal E_g$ |
| `protocol.entity_count` | positive integer | **required** | $K_g$; must equal the list length |
| `data.dataset_id` | string | Liander dataset identifier | population source |
| `data.revision` | string | pinned Git revision | immutable data version |
| `data.local_dir` | path string | `${SIMCAST_DATA_DIR:-data/liander2024}` | local files; deliberately excluded from the scientific fingerprint |
| `data.entity_type` | one of `transformer`, `solar_park`, `wind_park`, `mv_feeder`, `station_installation` | `transformer` | homogeneous physical group $g$; every ID must begin with this type prefix |
| `data.target_column` | string | `load` | measured target $y_{k,t}$ |

Example: transformer has $K_g=15$ in its supplied base; solar and wind each
have $K_g=5$. The group is invalid at $(i,\tau)$ if any member is invalid.

### 2.2 Forecast instances and information set

| Key | YAML type and admissible values | Default / supplied value | Meaning |
|---|---|---|---|
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

All timestamps are UTC. For `vintage`, future weather at physical time $t^{(i)}+\tau\Delta$ is selected
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

Source and model revisions determine the statistical forecast and enter the
fingerprint. Device and batch size are operational and do not. Each entity is
forecast as a separate Chronos task, without cross-entity attention.

### 2.5 Finite-quantile marginal law and PIT

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `pit.mode` | `discretized` or `linear_interpolation` | `discretized` | fixes both historical PIT construction and scenario projection; `discretized` uses $Q+1$ cell midpoints and nearest native values, while `linear_interpolation` is piecewise linear between native quantiles with constant boundary segments |
| `pit.monotone_repair` | `none` or `isotonic` | `isotonic` | `isotonic` applies least-squares monotone repair before PIT construction and projection; `none` invalidates a crossed row |
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

### 2.6 Runtime and output

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `runtime.deterministic` | Boolean | `true` | request deterministic numerical execution |
| `runtime.log_level` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` | `INFO` | console verbosity only |
| `output.cache_dir` | path string | `artifacts/cache` | cache root; not part of the marginal fingerprint |
| `output.save_resolved_config` | Boolean | `true` | save the resolved configuration with the cache, fit, and evaluation |

These fields are operational and do not enter the marginal fingerprint.
Sampling and scoring are not base fields; they belong to evaluation
documents (§7.1).

## 3. Method configuration

### 3.1 M0: independence

```yaml
kind: method
id: m0
family: independent
```

No feature, model, or optimization section is admissible because
$R_{g,\tau}^{(i)}=I_{K_g}$ is completely specified. `kind` is the literal
string `method`; `id` is a safe slug, defaulting to the method YAML filename
stem when omitted; and `family` is the literal string `independent`. These are
the only valid M0 fields.

### 3.2 M1: static Gaussian copula

```yaml
kind: method
id: m1
family: static_gaussian
model:
  share_across_leads: false
  jitter: 1.0e-6
```

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `kind` | literal string `method` | **required** | declares one dependence hypothesis |
| `id` | safe slug | method YAML filename stem | method-file identity |
| `family` | literal string `static_gaussian` | **required** | selects M1 |
| `model.share_across_leads` | Boolean | `false` | `false`: one training correlation per lead; `true`: pool all valid leads |
| `model.jitter` | strictly positive real | `1.0e-6` | diagonal numerical stabilization |

M1 estimates each correlation with Ledoit--Wolf covariance shrinkage and has no
gradient optimization or conditioning features.

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
| `model.hidden_dims` | non-empty list of positive integers | `[256,128]` | widths of the shared entity-wise GELU network |
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
| `model.hidden_dims` | non-empty list of positive integers | `[256,128]` | widths of the GELU network mapping features to kernel coordinates |
| `model.embedding_dim` | positive integer | `16` | dimension of learned coordinates $h_k$ |
| `model.dropout` | real number in $[0,1)$ | `0.1` | training-time dropout probability |
| `model.initial_length_scale` | strictly positive real | `1.0` | initialization of $\ell$ in $\exp(-\|h_a-h_b\|^2/(2\ell^2))$ |
| `model.nugget` | strictly positive real | `1.0e-3` | kernel diagonal stabilization |
| `model.jitter` | strictly positive real | `1.0e-6` | final correlation stabilization |

M4 has no reduced-data flag, origin cap, or special budget.

### 3.7 Optimization for M2--M4

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `optimization.seed` | non-negative integer | `42` | initialization and batch order |
| `optimization.batch_size` | positive integer | `64` | complete $(i,\tau)$ vectors per batch |
| `optimization.epochs` | positive integer | `100` | maximum fitting passes |
| `optimization.patience` | positive integer no greater than `epochs` | `12` | validation pseudo-NLL stopping patience |
| `optimization.learning_rate` | strictly positive real | `1.0e-3` | AdamW step size |
| `optimization.weight_decay` | non-negative real | `1.0e-4` | AdamW decay coefficient |
| `optimization.gradient_clip_norm` | strictly positive real | `1.0` | gradient-norm bound |

All three trainable methods use AdamW with the same semantics. A composite may
override budgets explicitly and validation then applies to that method type.

## 4. Composite configuration

A composite has two compulsory sections (`bases`, `methods`) and two optional
ones (`evaluations`, `reports`). Evaluations depend on bases and methods;
reports depend on evaluations. Declaring `reports` without `evaluations` fails
validation rather than silently doing nothing.

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `kind` | literal string `composite` | **required** | declares an explicit collection of comparisons |
| `name` | safe slug | composite YAML filename stem | composite identity within the venue |
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
| `reports` | list of report entries | `[]` | optional report designs generated after their evaluations complete |
| `reports[].id` | safe slug, unique within `reports` | **required** | report identity |
| `reports[].config` | report YAML path | **required** | report metrics and uncertainty settings |
| `reports[].evaluation_ids` | non-empty list of known evaluation IDs | **required** | evaluations selected for the report |

`bases` assigns local IDs to base files and optional valid base overrides.
`methods` assigns distinct IDs to method files, optional base selections,
optional conditional seeds, and role-valid method overrides. `evaluations`
assigns evaluation documents (§7.1) to selected base/method fits;
`uv run composite` runs them immediately after fitting. `reports` assigns
report documents (§7.2) to evaluation IDs; `composite` generates them after
their selected evaluations complete. `uv run report` can regenerate these
reports or apply a standalone report document to an existing run.

Deterministic M0/M1 entries cannot declare repeated seeds. Conditional entries
without `seeds` use their method file's optimization seed. The reference
method and bootstrap settings belong to a report, not to fitting or
evaluation. Fits are independent of every evaluation setting, so a new
evaluation can be applied to an existing run without retraining (see
[usage guide §7](usage_guide.md#7-evaluating-fits)).
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
runs/<venue>/<composite>/<run-id>/
  models/<base-id>/<method-id>/<seed-label>/
  evaluations/<evaluation-id>/<base-id>/figures/
  evaluations/<evaluation-id>/<base-id>/<method-id>/<seed-label>/
  reports/<report-id>/<evaluation-id>/
```

This rule makes a venue cloneable and prevents path escape. `run-id` is either
a UTC timestamp or an explicit lowercase safe slug. `seed-label` is
`deterministic` for M0/M1 fits or `seed_<N>` for a repeated conditional fit.

## 6. Cache fingerprint

See the [usage guide](usage_guide.md#cache-compatibility) for the SHA-256
definition and compatibility rule; this section lists only its field-level
consequences. `base_fingerprint` hashes precisely the base fields that determine the frozen
marginal/PIT record. It excludes the local data path, Chronos device and batch
size, runtime, output, method, composite, venue, evaluation, and report
settings. `locate_compatible_cache` verifies metadata, not a user-supplied
cache label.

Example: changing `evaluation.joint_score_num_samples` reuses the same marginal
cache. Changing `pit.monotone_repair`, ordered entity IDs, weather source,
forecast horizon, or Chronos model revision requires a different cache.

## 7. Evaluation and report configuration

Evaluation and report documents are reusable: a composite references them by
path, and `uv run evaluate --config` / `uv run report --config` apply them to
any completed run supplied with `--run-root`. The run path is never stored in
the YAML. Commands and output locations are described in the usage guide
([evaluating](usage_guide.md#7-evaluating-fits),
[reporting](usage_guide.md#8-reporting)).

### 7.1 Evaluation document

```yaml
kind: evaluation
id: variogram_power_1
metrics: [mean_pinball, crps, energy_score, variogram_score]
evaluation:
  variogram_power: 1.0
figures:
  aggregate_origin: "2024-01-15T23:45:00Z"
  correlation_lead: 1
```

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `kind` | literal string `evaluation` | **required** | declares one scoring design |
| `id` | safe slug | evaluation YAML filename stem | evaluation directory name in standalone mode; a composite entry's own `id` takes precedence |
| `metrics` | non-empty unique subset of `mean_pinball`, `crps`, `weighted_interval_score`, `energy_score`, `variogram_score`, `test_pseudo_nll` | all six | exact computation contract: only listed scores are computed and persisted |
| `sampling.num_samples` | positive integer | `4096` | scenarios $M$ per valid case |
| `sampling.evaluation_seed` | non-negative integer | `2027` | scenario randomness |
| `sampling.common_random_numbers` | Boolean | `true` | case-keyed Gaussian draws shared across methods |
| `evaluation.cross_entity_statistic` | `sum`, `absolute_sum`, `max`, or `absolute_max` | `sum` | scalar $T(\mathbf y)$ scored by aggregate metrics: $\sum_k y_k$, $\sum_k \lvert y_k\rvert$, $\max_k y_k$, or $\max_k \lvert y_k\rvert$, applied to every scenario and the observation |
| `evaluation.quantile_levels` | sorted, unique non-empty list in $(0,1)$ | `[.05,.10,.25,.50,.75,.90,.95]` | aggregate quantiles scored by pinball loss |
| `evaluation.interval_levels` | sorted, unique non-empty list in $(0,1)$ | `[.50,.80,.90]` | central interval coverages $c$, using the $(1-c)/2$ and $(1+c)/2$ scenario quantiles |
| `evaluation.variogram_power` | real number in $(0,2]$ | `0.5` | Variogram Score pairwise-difference exponent |
| `evaluation.joint_score_num_samples` | positive integer | `512` | selected joint-ensemble size for Energy/Variogram Scores |
| `evaluation.scenario_batch_size` | positive integer | `16` | scenario-generation batch size; operational only |
| `figures.aggregate_origin` | ISO timestamp of a test origin | first test origin | origin shown in `aggregate_fan.png` |
| `figures.correlation_origin` | ISO timestamp of a test origin | first test origin | origin shown in `correlation.png` |
| `figures.correlation_lead` | positive integer no greater than the horizon | `1` | lead shown in `correlation.png` |
| `base_ids` | list of unique base IDs | `[]` | standalone mode only: bases to evaluate; empty means all |
| `method_ids` | list of unique method IDs | `[]` | standalone mode only: methods to evaluate; empty means all |

Evaluations are always written to `<run-root>/evaluations/<evaluation-id>/`.
Aggregate quantiles are nearest empirical order statistics of the scenarios.
`mean_pinball` also persists its per-level `pinball_q*` values;
`weighted_interval_score` also persists `coverage_*`, `interval_width_*`, and
`interval_score_*` values. No score is present unless its corresponding
canonical metric is listed. The score definitions, references, and caveats are in
[Chapter 5](../scientific/05_training_sampling_scoring.md#aggregate-scores).
`weighted_interval_score` uses the levels implied by `interval_levels`; when
those equal `quantile_levels` (as by default) it is exactly twice
`mean_pinball`.

### 7.2 Report document

```yaml
kind: report
reference: m0
metrics: [mean_pinball, crps]
evaluation_ids: [standard]
analysis:
  bootstrap_replicates: 10000
  primary_block_length: 7
  sensitivity_block_lengths: [3, 14]
```

| Key | YAML type and admissible values | Default | Meaning |
|---|---|---|---|
| `kind` | literal string `report` | **required** | declares one cross-method comparison |
| `reference` | safe slug naming a composite method ID | **required** | method against which paired effects are computed |
| `metrics` | list of unique names declared by the selected evaluations | `[]` | presentation subset; empty means every declared metric |
| `evaluation_ids` | list of evaluation IDs recorded in the run manifest | `[]` | standalone mode only: evaluations to report; empty means all. In composite mode the entry's `evaluation_ids` are used |
| `analysis` | mapping | **required** | bootstrap design for this report; its fields have the defaults below |
| `analysis.bootstrap_replicates` | positive integer | `10000` | moving-block bootstrap replicates |
| `analysis.primary_block_length` | positive integer | `7` | primary block length, in origins |
| `analysis.sensitivity_block_lengths` | list of unique positive integers | `[3, 14]` | additional block lengths shown alongside the primary one |
| `output_dir` | path | `<run-root>/reports/<report-id>/` | report root; each evaluation is written to `<output_dir>/<evaluation-id>/` |

Evaluation `metrics` is authoritative and recorded in
`evaluation_manifest.json`. Report `metrics` is only a presentation filter: it
cannot request a metric the selected evaluations did not declare. Paired
effects are computed for every reported metric, after averaging seeds within
method and origin.
