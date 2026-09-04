# Artifact schemas and output files

## PIT library directory

Default location is `artifacts/cache/<cache_name>/`:

```text
dataset.zarr/                 public arrays; test labels replaced by NaN
test_labels.zarr/             true_y, pit_u, pit_z for test origins only
metadata.json                 provenance, protocol, missingness, crossings
marginal_diagnostics.json     train+validation marginal diagnostics only
resolved_config.json          fully expanded configuration
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
| `pit_u` | `[N,K,H]` | float32 | discrete PIT; test slice masked publicly |
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

## Model run directory

Every trained method directory contains:

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
versions, absolute cache path, entity IDs, and seed. M0's NPZ stores IDs and
lead count. M1's stores all correlation matrices plus estimator settings.

Conditional PyTorch checkpoints contain `schema_version`, method, model
constructor arguments, feature-builder configuration and fitted training-only
means/scales, entity IDs, epoch, and `model_state_dict`. `best.pt` is the model
used in evaluation; `final.pt` is retained to diagnose optimization.

## Evaluation directory

```text
metrics.json
metrics_by_lead.csv
variable_k.csv
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
| `energy_score` | `[N_test,H]` | paired MC score or NaN |
| `variogram_score` | `[N_test,H]` | score or NaN |
| `valid` | `[N_test,H]` | common evaluation mask |

Entity-level and aggregate Monte Carlo samples are not persisted, which keeps
artifacts manageable but means exact secondary analyses requiring raw scenarios
must regenerate them.

### `variable_k.csv`

Rows contain method, entity-prefix cardinality, mean pinball, and widest
configured coverage. The manifest maps every cardinality to its exact entity
IDs, preventing ambiguous subset interpretation.

### Manifests and summary

`evaluation_manifest.json` records cache path, method-run paths, methods,
test-origin/case counts, full entity IDs, variable-K subset IDs, joint-score
estimator name, Git commit, and time. `scientific_summary.json` gives six
machine-readable descriptive answers; it is not a substitute for inspecting
proper scores and uncertainty.

## Figures

Evaluation normally creates dataset location/load/missingness plots; marginal
PIT, coverage, and pinball diagnostics; empirical/static correlation heatmaps
and eigenvalues; per-method correlation and aggregate-fan examples; M2/M3
dependence-dynamics and factor plots; score/coverage summaries; and
variable-cardinality plots. Example fans use the first valid flattened
origin-lead position, while the fan traces all leads for that origin.

Factor plots must be interpreted cautiously because low-rank factors are
rotation non-identifiable. Correlation matrices and aggregate score changes are
scientifically more stable objects.

## Experiment root and replay

The high-level runner adds `experiment_manifest.json` at the root with cache,
method, evaluation, bounded-M4 status, Git commit, and creation time. Its
`resolved_config.yaml` is the base experiment config; each method subdirectory
also records the method-specific resolved config (including M4 caps).

Artifact paths are absolute in manifests. Moving a repository or deleting a
cache/checkpoint breaks replay until paths are updated or explicit inputs are
provided. Checksums are not currently stored, so archival research releases
should checksum or package the full cache/run tree externally.
