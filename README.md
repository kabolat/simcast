# Simcast

> **Detailed research documentation:** start with
> [`docs/README.md`](docs/README.md) for the mathematical formulation, exact
> information-set rules, M0--M4 derivations, pipeline operation, experiment
> protocol/results, artifact schemas, limitations, and reproduction guide.

Simcast is a research proof of concept for learning same-lead, cross-entity
forecast-error dependence on top of frozen Chronos-2 forecasts. FM-derived
marginal quantile grids are fixed across dependence methods: raw native grids
are used for most groups, while solar uses one deterministic isotonic repair
before the grid is fixed. Only the copula used to form spatial aggregates
changes.

For forecast instance $i$, lead $\tau$, and complete static group
$\mathcal E_g$, each method draws entity scenarios and evaluates the observed
aggregate

```text
a[g, tau, i] = sum(y[k, tau, i] for k in E_g).
```

The dependence target is the vector of discretized, Gaussianized PIT
pseudo-observations at one lead. Simcast never models temporal covariance
between different leads and never treats raw-load correlation as forecast-error
dependence.

## Methods

- M0 `independent`: identity Gaussian copula; no fitting.
- M1 `static_gaussian`: Ledoit-Wolf PIT correlation per lead, with optional
  pooling across leads.
- M2 `conditional_low_rank`: shared entity-wise MLP predicting factor loadings
  and uniqueness from frozen Chronos features.
- M3 `set_aware_low_rank`: position-free entity self-attention followed by the
  same low-rank correlation construction.
- M4 `conditional_kernel`: optional RBF-kernel correlation with separate full
  training and bounded smoke configurations.

M2 and M3 have parameter counts independent of entity cardinality, and M3 is
permutation equivariant. The full-group protocol nevertheless uses every
member of each complete static group in every training, validation, and test
case. Configuration validation rejects subset training and reduced-cardinality
evaluation.

The corrected five-group baseline table and the audit that excludes
the earlier subset-trained neural runs are in
[`docs/08_experiments_and_results.md`](docs/08_experiments_and_results.md).
Machine-readable results are tracked in
[`results/full_group_metrics.csv`](results/full_group_metrics.csv).

## Reproducible setup with uv

Python dependencies are locked by `uv.lock`. The official Chronos source is
pinned and installed editable after applying the minimal output-contract patch.

```bash
uv sync --group dev
bash scripts/setup_chronos.sh
```

Pinned revisions:

- Liander2024: `dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044`
- Chronos source: `8589d1988e9676817548e9626738ff06b6ca6370`
- Chronos-2 model: `29ec3766d36d6f73f0696f85560a422f50e8498c`

The patch exposes the already computed output-patch `forecast_embeds`. It does
not alter encoder states, normalization, attention, loss, or quantile values.

## Liander2024 workflow

The default experiment uses one static group containing all entities with
`group_name: transformer`, 15-minute resolution, seven days of context
(`L=672`), a 24-hour horizon (`H=96`), and daily origins at 23:45 UTC. Canonical
entity IDs are `group_name::name`; cardinality is read from
`liander2024_targets.yaml`, never hardcoded.

Set the local data directory once, then run the individual stages:

```bash
export SIMCAST_DATA_DIR=/path/to/liander2024

uv run python -m simcast.cli.download_data \
  --config configs/liander2024_transformer.yaml

uv run python -m simcast.cli.build_cache \
  --config configs/liander2024_transformer.yaml

uv run python -m simcast.cli.train_dependence \
  --config configs/method_independent.yaml
uv run python -m simcast.cli.train_dependence \
  --config configs/method_static_gaussian.yaml
uv run python -m simcast.cli.train_dependence \
  --config configs/method_conditional_low_rank.yaml
uv run python -m simcast.cli.train_dependence \
  --config configs/method_set_aware_low_rank.yaml

uv run python -m simcast.cli.evaluate \
  --config configs/liander2024_transformer.yaml \
  --methods independent \
  --methods static_gaussian \
  --methods conditional_low_rank \
  --methods set_aware_low_rank
```

Or run cache construction, M0--M3 training, and the one-time final evaluation
with one command:

```bash
uv run python -m simcast.cli.run_experiment \
  --config configs/liander2024_transformer.yaml
```

Add `--include-kernel-smoke` to include bounded M4. An existing compatible
cache is reused. `--rebuild-cache` explicitly replaces it.

To train optional M4 with the full transformer train/validation partitions and
the same optimizer budget as M2/M3:

```bash
uv run python -m simcast.cli.train_dependence \
  --config configs/method_conditional_kernel.yaml
```

Named configs are also provided for every homogeneous Liander entity type:

- `configs/liander2024_transformer.yaml`
- `configs/liander2024_solar_park.yaml`
- `configs/liander2024_wind_park.yaml`
- `configs/liander2024_mv_feeder.yaml`
- `configs/liander2024_station_installation.yaml`

The solar config explicitly enables isotonic repair because raw Chronos solar
quantiles cross frequently around zero-output hours; without repair, several
leads have too few complete PIT vectors to estimate M1. Raw crossing rates are
still recorded in cache diagnostics.

Configuration values can be overridden with repeatable dotted assignments:

```bash
uv run python -m simcast.cli.run_experiment \
  --config configs/liander2024_transformer.yaml \
  --set training.epochs=20 \
  --set sampling.num_samples=2048
```

## Leakage controls

- Past targets and measured weather must be available by the forecast origin.
- Each future timestamp uses the latest versioned weather forecast whose
  `available_at` is no later than the origin.
- Origins are chronological; train, validation, and test boundaries are purged
  when forecast horizons overlap.
- Scalar feature statistics are fitted on training origins only.
- Test `true_y`, `pit_u`, and `pit_z` are physically stored in a separate Zarr
  group. Training APIs cannot open them; only final evaluation requests
  `access="evaluation"`.
- A missing realization or invalid/crossing marginal drops the complete
  `(origin, lead)` spatial vector, never an entity from the static group.

Quantiles are not interpolated. Historical pseudo-PIT values use deterministic
`Q+1` cells; scenario uniforms use a separate nearest-quantile probability-cell
projection. Consequently every method has exactly the same fixed discrete
marginal law within a group; only joint co-occurrence changes.

## Outputs

The PIT library is an xarray/Zarr artifact with named `origin`, `entity`,
`lead`, `quantile`, `patch`, and `hidden` dimensions. Forecast-patch embeddings
are stored once per patch in float16 and converted to float32 for adapter
training.

Each model run contains its resolved configuration, complete ordered entity IDs,
software and Git metadata, model/checkpoints, training histories, and a training
curve. Final evaluation writes:

- overall JSON and lead-wise CSV metrics;
- aggregate pinball, coverage, interval width/score, WIS, and CRPS;
- spatial Energy and Variogram Scores (Energy Score uses the empirical
  all-pairs estimator on the selected 512-member joint ensemble, with chunking
  only for memory efficiency);
- marginal, correlation, factor, dynamics, aggregate-fan, and summary figures;
- a six-question scientific summary.

The test labels are opened only during this final evaluation. To rerun a saved
training or evaluation artifact:

```bash
uv run python -m simcast.cli.reproduce runs/<run-directory> \
  --output-dir runs/<new-directory>
```

## Development checks

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
```

CUDA seeds and deterministic-algorithm requests are set. Exact bitwise replay
can still be affected by GPU kernels, PyTorch/CUDA versions, and device model;
these versions are stored with each training run.

## Attribution and license

The [Liander2024 Energy Forecasting Benchmark](https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark)
is published by OpenSTEF under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Chronos is Copyright Amazon.com, Inc. or its affiliates and licensed under
Apache-2.0. Simcast's own source code is licensed under the MIT License.
