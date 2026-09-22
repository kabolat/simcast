# Artifact schemas and output files

This is the storage-level counterpart to the scientific study. The meaning of
the cached marginals and PITs is defined in
[Frozen marginal forecasts and finite PITs](../scientific/03_chronos_and_pit.md);
the meaning of fitted methods and reported scores is defined in
[Dependence fitting, scenario generation, and scoring](../scientific/05_training_sampling_scoring.md).
The declaration-to-evidence chain is described in
[Scientific workflow](../scientific/06_scientific_workflow.md), and the
reporting interpretation is described in
[Experiments, reporting, and result interpretation](../scientific/07_experiments_and_results.md).

## PIT library directory

The canonical location is
`artifacts/cache/<base-id>-<marginal-fingerprint-prefix>/`:

```text
dataset.zarr/                 public arrays; test labels replaced by NaN
test_labels.zarr/             true_y, pit_u, pit_z for test origins only
metadata.json                 provenance, protocol, missingness, crossings
marginal_diagnostics.json     train+validation marginal diagnostics only
resolved_config.json          internal numerical view used to build the record
```

### `dataset.zarr`

The xarray schema is `simcast.pit_library`, version 1.

| Variable/coordinate | Dimensions | Typical dtype | Meaning |
|---|---|---|---|
| `origin` | `[N]` | int64 | zero-based stable index |
| `origin_timestamp` | `[N]` | datetime64 ns | UTC origin represented without xarray timezone metadata |
| `split` | `[N]` | string | train/validation/test |
| `entity` | `[K]` | int64 | positional entity coordinate |
| `entity_id` | `[K]` | string | canonical `group::name` ID |
| `lead` | `[H]` | int64 | one-based lead 1..H |
| `quantile` | `[Q]` | float32 | native Chronos probabilities |
| `patch` | `[P]` | int64 | zero-based output patch |
| `hidden` | `[D]` | int64 | embedding channel |
| `true_y` | `[N,K,H]` | float | realization; test slice masked publicly |
| `quantile_prediction` | `[N,K,H,Q]` | float32 | fixed native marginal; never test-masked |
| `crossing_{origin,entity,lead}_index` | `[C]` | int64 | coordinates of the `C` raw quantile rows that cross, retained only with isotonic repair |
| `crossing_raw_quantile_prediction` | `[C,Q]` | float32 | raw Chronos values for those crossed rows; the matching repaired row is addressed in `quantile_prediction` by its stored coordinates |
| `pit_u` | `[N,K,H]` | float32 | configured finite-quantile PIT; test slice masked publicly |
| `pit_z` | `[N,K,H]` | float32 | Gaussian score; test slice masked publicly |
| `forecast_embedding` | `[N,K,P,D]` | float16 | frozen output-patch representation |
| `pit_valid` | `[N,H]` | bool | complete-vector gate |
| `latitude`, `longitude` | `[K]` | float32 | static location |

`test_labels.zarr` stores only label variables for test `origin` positions.
Training access leaves public NaNs intact. Evaluation access loads the separate
group and inserts those arrays at their original integer positions.

### Cache metadata

`metadata.json` contains dataset ID/revision/path, group membership/names and
coordinates, Chronos source/model revisions and native output structure,
forecast window protocol, covariate config, raw crossing diagnostics, invalid
vector counts, historical/future missingness, eligible/skipped/retained origin
counts, purged boundary timestamps, full resolved config, schema version, and
`test_labels_sealed: true`.

`marginal_diagnostics.json` is explicitly scoped `train_and_validation`. It
contains overall and per-entity median MAE, native-level pinball losses,
central interval coverage, PIT histogram counts/frequencies, PIT mean and
variance, and tail-cell fractions. Missing values are omitted metric by metric.

## Dependence-fit directory

Every fitted method directory contains:

```text
resolved_config.yaml
run_metadata.json
model.npz                       M0/M1 only
best.pt                         M2/M3/M4 only
final.pt                        M2/M3/M4 only
training_metrics.csv/json       M2/M3/M4 only
training_summary.json           M2/M3/M4 only
training_curve.png              M2/M3/M4 only
```

`run_metadata.json` records creation time, Git commit, Python and key package
versions, absolute cache path, seed, group name, complete ordered entity IDs,
$K_g$, `full_group_only`, and the absence of entity selection. It also records
resolved-config and model SHA-256
hashes, pinned data/FM revisions, optimization and evaluation seeds, checkpoint
criterion, and PIT dependence transform. M0's NPZ stores IDs and
lead count. M1's stores all correlation matrices plus estimator settings.

Conditional PyTorch checkpoints contain `schema_version`, method, model
constructor arguments, feature-builder configuration and fitted training-only
means/scales, entity IDs, epoch, and `model_state_dict`. `best.pt` is the model
used in evaluation; `final.pt` is retained to diagnose optimization.

## Evaluation directory

```text
metrics.json
metrics_by_lead.csv
per_origin_lead_metrics.parquet
per_origin_metrics.parquet
<method>_aggregate_predictions.npz
scientific_summary.json
evaluation_manifest.json
resolved_config.yaml
figures/*.png
```

### `metrics.json`

Top-level keys are method names. Each method includes mean pinball, pinball by
evaluation quantile, CRPS, coverage/width/interval score by central interval,
WIS, valid and dropped case counts, and—when enabled—mean Energy and Variogram
Scores. Values pool all valid test origins and leads.
The method payload also includes mean test Gaussian-copula pseudo-NLL.

### Tidy per-case tables

`per_origin_lead_metrics.parquet` has one row per complete declared group,
method, neural seed, origin, and lead. It records $K_g$, validity, observed and
forecast aggregate quantiles, aggregate proper scores, joint scores, and test
pseudo-NLL. Invalid rows remain present with `valid: false`; the group never
shrinks. `per_origin_metrics.parquet` averages metrics across valid leads within
each origin and is the input to temporal block resampling.

### `metrics_by_lead.csv`

Each row is `(lead, method)`. It holds mean pinball, CRPS, WIS, interval
coverage/width/scores, and mean joint scores for that one-based lead. Invalid
cases at a lead are omitted. Although `evaluation.report_by_lead` is recorded,
the current evaluator always writes this table.

### Per-method NPZ

| Array | Shape | Meaning |
|---|---|---|
| `quantile_predictions` | `[N_test,H,Q_eval]` | empirical aggregate quantiles |
| `correlations` | `[N_test,H,K,K]` | evaluated copula correlations |
| `energy_score` | `[N_test,H]` | empirical all-pairs score on the selected joint ensemble, or NaN |
| `variogram_score` | `[N_test,H]` | score or NaN |
| `pseudo_nll` | `[N_test,H]` | finite-quantile Gaussian-copula pseudo-NLL or NaN |
| `valid` | `[N_test,H]` | common evaluation mask |

Entity-level and aggregate Monte Carlo samples are not persisted, which keeps
artifacts manageable but means exact secondary analyses requiring raw scenarios
must regenerate them.

### Manifests and summary

`evaluation_manifest.json` records cache path, method-run paths, methods,
test-origin/case counts, the complete ordered group and $K_g$, full-group
protocol flags, scenario and selected joint ensemble sizes, common-random-number
status, evaluation seed, revisions, config/model hashes, PIT mode and dependence transform,
confirmatory labels, Git commit, and time.
`scientific_summary.json` gives six
machine-readable descriptive answers; it is not a substitute for inspecting
proper scores and uncertainty.

## Figures

Evaluation normally creates dataset location/load/missingness plots; marginal
PIT, coverage, and pinball diagnostics; empirical/static correlation heatmaps
and eigenvalues; per-method correlation and aggregate-fan examples; M2/M3
dependence-dynamics and factor plots; and full-group score/coverage summaries.
Example fans use the first valid flattened
origin-lead position, while the fan traces all leads for that origin.

Factor plots must be interpreted cautiously because low-rank factors are
rotation non-identifiable. Correlation matrices and aggregate score changes are
scientifically more stable objects.

## Singular experiment root

A singular root contains `resolved_base.yaml`, `resolved_method.yaml`, an
optional `resolved_reference_method.yaml`, `singular_manifest.json`, fitted
method directories, and one evaluation directory. The role separation makes it
possible to verify which quantities defined the fixed marginal law and which
defined the dependence hypothesis.

## Composite experiment and report roots

Composite output follows:

```text
runs/<venue>/<composite>/<run-id>/
reports/<venue>/<composite>/<run-id>/
```

The run root contains the resolved composite, exact expansion manifest,
environment and Git record, base/method hashes, composite log, shared M0/M1
fits, seed-specific M2--M4 fits, evaluations, and completion state. The report
root contains concatenated per-origin records, method summaries, paired
effects, and figures. M4 is processed by the same generic traversal as M2 and
M3. The [usage guide](usage_guide.md#10-outputs-and-interpretation) gives the
practical inspection order; this reference defines the meaning and location of
the files.

The current generic report root contains:

```text
per_origin_metrics.parquet          concatenated origin-level records
method_summary.csv                  mean value per (base, experiment, method, seed), one row per metric
paired_effects.csv                  paired moving-block bootstrap effects, one row per metric/comparison
method_comparison_<metric>.png      base-panel absolute score bar charts for one metric
paired_effect_<metric>.png          base-panel relative-improvement plot with bootstrap intervals
report_summary.md                   plain-text index describing every file above
```

`per_origin_metrics.parquet` retains the complete evaluation rows, including
reference-method rows repeated inside a primary method's evaluation. The
`method_summary.csv` and `method_comparison_<metric>` figures include only the
primary method declared by each experiment entry; reference rows are retained
for `paired_effects.csv` but are not shown as duplicate method bars. Each
visual uses one panel per base, because absolute score scales are not
comparable across entity types. `paired_effect_<metric>.png` plots positive
relative improvement, $-100d/\bar S_{reference}$, so upward values favor the
tested method.

`run_composite` always writes this report using only `mean_pinball`.
`simcast.cli.report_composite` regenerates the same four data files plus
`report_summary.md` from an existing `composite_manifest.json`, for any
configured `metrics` list and `analysis` settings, without re-fitting or
re-evaluating; see
[usage guide §15](usage_guide.md#15-regenerating-or-customizing-a-report) and
[configuration reference §7](configuration_reference.md#7-report-configuration).

The report fields retain `base_id`, `experiment_id`, `method`, and configured
seed so that a displayed comparison can be traced to its resolved declaration.

Resume compares the stored composite SHA-256 digest with the newly resolved
declaration. A mismatch is rejected; a validated complete cell is preserved.
Artifact paths may be absolute in manifests, so archival releases should
package the complete referenced tree or provide stable remapping metadata.

## Historical artifacts

`read_evaluation_artifacts` recursively discovers current and historical
`evaluation_manifest.json` files and returns their original JSON payloads. It
performs no schema migration and never rewrites an artifact. Historical
artifacts remain inspectable but are not executable through removed monolithic
configuration interfaces.

## Scientific example and implementation guidance

A tensor named `quantile_prediction` with dimensions
`[origin, entity, lead, quantile]` is evidence only when its coordinate values
and metadata identify $t^{(i)}$, the complete $\mathcal E_g$, $\tau$, and
$q_j$. Likewise, a score without its valid-case mask and method correlation is
not independently interpretable. The record schema preserves these links.

`save_pit_library` and `load_pit_library` control the marginal record;
dependence fitting and evaluation entry points write method and score records.
A human-chosen directory name has no scientific meaning. Revision
hashes, ordered entity IDs, resolved configuration, and content checksums are
the evidence used to establish compatibility.
