# Software pipeline and module map

## Pipeline stages

The implementation is organized as a sequence of scientific artifacts. Each
stage has a narrow input/output contract so expensive frozen forecasts can be
reused without reopening the test set or rerunning Chronos.

| Stage | Entry point | Reads | Writes |
|---|---|---|---|
| 1. Acquire | `simcast.cli.download_data` | resolved config, Hugging Face snapshot | pinned Parquet/YAML file selection |
| 2. Cache marginals | `simcast.cli.build_cache` | raw data, patched frozen Chronos | PIT library and tune-only diagnostics |
| 3. Fit dependence | `simcast.cli.train_dependence` | training-visible PIT library | M0/M1 model or M2--M4 checkpoints |
| 4. Final evaluate | `simcast.cli.evaluate` | sealed test labels, fitted runs | metrics, predictions, manifests, figures |
| 5. Orchestrate | `simcast.cli.run_experiment` | one config | stages 2--4 under one experiment dir |
| 6. Replay | `simcast.cli.reproduce` | saved run/evaluation dir | new training or evaluation artifact |

## Stage 1: data acquisition

`download_data` resolves YAML inheritance, environment variables, and CLI
overrides before downloading. It requires an exact dataset revision and uses
allow-pattern filtering for the configured entity type. No data download occurs
at package import time.

## Stage 2: immutable marginal cache

`build_cache` performs the expensive foundation-model work:

1. read target metadata and construct one homogeneous group;
2. read and normalize target, measured-weather, and forecast-weather frames;
3. intersect target coverage across entities;
4. generate candidate origin windows;
5. enforce point-in-time target and weather availability;
6. skip origins with incomplete covariates;
7. run frozen Chronos separately for each physical entity;
8. validate stable entity order, native quantile levels, patch size, and tensor
   dimensions across origins;
9. chronologically split and purge origins;
10. diagnose crossings, optionally repair, construct PIT scores, and apply the
    complete-vector gate;
11. write an atomic Zarr cache plus scientific metadata and tune-only marginal
    diagnostics.

The cache is built in a temporary sibling directory and moved into place only
after all writes succeed. Existing destinations are rejected unless
`--overwrite` is explicit.

## Stage 3: dependence fitting

Training opens the cache with `access="training"`. Test truths and PIT values
are physically absent from the in-memory public dataset.

- M0 records metadata and saves `model.npz`.
- M1 fits training-only Ledoit--Wolf matrices and saves `model.npz`.
- M2--M4 construct train/validation features using training-only
  standardization, flatten complete cases, optimize pseudo-NLL, and save
  `best.pt`/`final.pt` plus histories.

Each output directory is created with `exist_ok=False`; reruns cannot silently
overwrite a prior model artifact.

## Stage 4: single final evaluation

Evaluation explicitly opens the cache with `access="evaluation"`, which merges
`test_labels.zarr` into test positions in memory. It then:

1. loads M1 matrices and best conditional checkpoints;
2. reconstructs test features from stored builder statistics;
3. predicts all origin/lead correlations in chunks of 256;
4. creates one common valid-case mask;
5. generates method scenarios with common base normals;
6. computes aggregate and joint scores;
7. asserts that every correlation has exact full-group shape
   $K_g\times K_g$;
8. writes full-group metrics, scenario-derived quantiles, correlations, figures, a
   scientific summary, and an evaluation manifest.

The phrase “single final evaluation” is a protocol, not a cryptographic access
limit: a user can rerun the command. Scientific practice must avoid iteratively
tuning decisions against the resulting test scores.

## High-level runner behavior

`run_experiment` creates a new experiment directory, reuses an existing cache
unless `--rebuild-cache` is given, trains M0--M3 sequentially in named
subdirectories, then evaluates all methods once. With
`--include-kernel-smoke`, it adds optional M4 and programmatically caps M4
epochs and patience. All full-group methods retain the complete entity group.

Cache compatibility is the user's responsibility when reusing a manually
specified cache path. The cache metadata and run manifests make mismatches
auditable, but the runner does not hash and compare the entire resolved config
before reuse.

## Source tree map

```text
src/simcast/
  config.py                         strict config composition and validation
  types.py                          entity metadata/group dataclasses
  data/
    liander2024.py                  Parquet and timestamp normalization
    grouping.py                     homogeneous static groups and canonical IDs
    availability.py                 point-in-time target/weather selection
    covariates.py                   weather and cyclic calendar features
    windows.py                      aligned windows and purged chronological split
  fm/
    chronos2_features.py            frozen direct inference and patch extraction
    pit.py                          crossings, isotonic repair, PIT/Gaussianization
    diagnostics.py                  tune-only frozen-marginal summaries
    feature_builder.py              deterministic conditional-copula features
    cache.py                        labeled Zarr schema and test-label gate
  dependence/
    base.py                         common model interface
    independent.py                  M0
    static_gaussian.py              M1
    conditional_low_rank.py         M2 and low-rank math
    set_aware_low_rank.py           M3
    conditional_kernel.py           M4
  training/
    dataset.py                      complete cases and legacy-capable collator
    losses.py                       stable Gaussian-copula pseudo-NLL
    trainer.py                      AdamW, validation, early stopping, histories
    checkpoint.py                   portable conditional checkpoint loader
  sampling/
    quantile_projection.py          nearest-native-probability projection
    gaussian_copula.py              Cholesky uniforms and spatial scenarios
  evaluation/
    metrics.py                      reusable proper-score implementations
    aggregate.py                    pooled and lead-wise aggregate evaluation
    plots.py                        diagnostic and paper-oriented figures
  cli/                              six stage/orchestration entry points
```

## Data-shape trace

For the transformer study, the principal shapes are:

| Object | Shape | Meaning |
|---|---|---|
| target history per origin | `[15,672]` | one week per entity |
| future truth | `[347,15,96]` | retained labels |
| native quantiles | `[347,15,96,21]` | frozen Chronos marginals |
| forecast embeddings | `[347,15,6,768]` | one vector per output patch |
| PIT `u`, `z` | `[347,15,96]` | complete-gated pseudo-observations |
| conditional features | `[N_split,15,96,794]` | built in float32 |
| one training case | `[15,794]`, `[15]` | features and PIT score vector |
| test correlations | `[70,96,15,15]` | method-specific matrices |
| aggregate samples | `[70,96,4096]` | complete-group sums; not persisted directly |
| aggregate quantiles | `[70,96,7]` | persisted in result NPZ |

For the two five-entity groups, replace entity dimension 15 by 5. Every case
retains that full dimension. The feature dimension remains 794 because model
parameters and feature construction do not depend on group cardinality.

## Scientific state boundaries

Three mechanisms prevent accidental state mixing:

- upstream source/data/model revisions are exact hashes;
- every run saves a fully resolved configuration and software/Git metadata;
- test labels reside in a physically distinct Zarr group and require an
  explicit evaluation access mode.

The system does not provide content-addressed caches, experiment databases,
distributed locking, or remote orchestration. Those would add engineering
complexity without strengthening the present proof-of-concept result.

## Error behavior worth knowing

The pipeline fails rather than guessing when entity order changes, quantile
levels or embedding shapes change between origins, a horizon exceeds direct
Chronos capacity, configured weather columns are missing, future covariates
cannot be known at the origin, a lead lacks two complete M1 vectors, or a model
matrix is not Cholesky-factorable. Origin-level incomplete covariates are the
main exception: those origins are deliberately skipped and recorded in cache
metadata.
