# Simcast

Simcast studies same-lead, cross-entity forecast-error dependence while holding
Chronos-2 marginal forecasts fixed. For forecast instance $i$, lead
$\tau\in[H]$, and canonical static physical entity universe
$\mathcal E_g=\{k_1,\ldots,k_{K_g}\}$, Chronos supplies fixed marginal
quantile grids for the random variables $Y_{k,\tau}^{(i)}$. A dependence method
supplies only the copula $C_{g,\tau}^{(i)}$:

$$
P(Y_{k,\tau}^{(i)}\le y_k,\ k\in\mathcal E_g\mid\mathcal I^{(i)})
=C_{g,\tau}^{(i)}(\{F_{k,\tau}^{(i)}(y_k)\}_{k\in\mathcal E_g}).
$$

The default `protocol.cardinality_mode: fixed` uses the full universe at every
case. Opt-in `variable` mode uses the forecastable, label-valid active set
$\mathcal A_{g,\tau}^{(i)}\subseteq\mathcal E_g$ when it has at least
`protocol.min_entities_per_case` members. Entity IDs and cache axes always
retain the complete canonical universe. Evaluation targets a cross-entity statistic of the active set: the
simple sum $\sum_k Y_k$, absolute sum $\sum_k |Y_k|$, maximum $\max_k Y_k$,
or absolute maximum $\max_k |Y_k|$. The evaluation YAML selects the statistic;
`sum` is the default.

In fixed mode, one invalid entity invalidates the complete case. In variable
mode, valid entity PITs are retained independently and all methods use the same
active subset for fitting, scenarios, and scoring. Test-label validity remains
sealed until evaluation. Variable mode covers both entities that are
intermittently missing within a shared period and entities with staggered
observation lifetimes; candidate origins then span the envelope of all
lifetimes, and an entity is forecastable only with at least one finite lookback
target (see
[Chapter 2](docs/scientific/02_data_and_information_set.md#minimum-target-history)).

Detailed scientific documentation begins at
[`docs/README.md`](docs/README.md). Executable explanations are in
[`notebooks/README.md`](notebooks/README.md).

## Dependence hypotheses

- M0: independent Gaussian copula, $R_{g,\tau}^{(i)}=I_{K_g}$.
- M1: static Gaussian copula estimated from training pseudo-PIT scores, either
  lead-specific or pooled.
- M2: conditional low-rank Gaussian copula whose shared entity-wise network
  predicts factor parameters from FM-derived features.
- M3: set-aware conditional low-rank Gaussian copula that contextualizes the
  active entity set with position-free entity self-attention.
- M4: conditional RBF-kernel Gaussian copula in a learned embedding space.

M4 has the same experimental status as M2 and M3. All three use the same
chronological training and validation partitions; variable mode compacts each
case to its active entities. Optimization budgets are explicit. A reduced
laboratory budget is a property of a composite experiment, not of a method.

FM-derived marginal grids are fixed across all five methods. Three entity groups
use raw native Chronos-2 quantiles. The transformer and solar bases use
deterministic isotonic repair; each repaired grid is then fixed for every
method. The base configuration chooses one finite-quantile marginal
law: the default discretized midpoint construction or piecewise-linear
interpolation between native quantiles. The same law is used for historical PIT
construction and scenario projection.

## Configuration as scientific design

Human-authored YAML files declare five distinct objects:

```text
configs/
  bases/<dataset>/    # data adapter, group, information set, Chronos, PIT
  methods/             # exactly one of M0--M4
  venues/<venue>/      # composites: which bases, methods, and seeds to fit
  evaluations/         # sampling, metrics, cross-entity statistic, figures
  reports/             # reference method and bootstrap design
```

A base defines the frozen-marginal experiment. A method file contains only
parameters meaningful to one dependence hypothesis. A composite explicitly
lists bases, method variants, and seeds; no hidden Cartesian grid is generated.
Evaluations score fits and reports compare evaluated methods; neither changes
a fit, so both can be repeated on a completed run.

Data ingestion is selected by the explicit `data.adapter` field. The
[dataset references](docs/datasets/README.md) document each adapter's data,
availability rules, base files, and adapter-specific options. The generic
[adapter interface](docs/technical/data_adapter_interface.md) describes how
the data layer is extended.

The marginal cache is identified by a SHA-256 fingerprint of the scientific
inputs that determine it, so method, venue, evaluation, and report settings
never invalidate it; see
[cache compatibility](docs/technical/usage_guide.md#cache-compatibility).

## Quick start

The [usage guide](docs/technical/usage_guide.md) covers every step in detail.

```bash
uv sync --group dev
bash scripts/setup_chronos.sh
uv run download-data --base configs/bases/liander2024/transformer.yaml
```

For the GEFCom2014 Wind example, download the two pinned Parquet files, build
its Chronos cache, then run the lab composite:

```bash
uv run download-data --base configs/bases/gefcom2014/wind.yaml
uv run cache --base configs/bases/gefcom2014/wind.yaml
uv run composite --config configs/venues/lab/gefcom_wind_smoke.yaml
```

Pinned revisions are recorded in the base configuration and lockfile. The
Chronos patch exposes the already computed output-patch representation; it does
not change Chronos attention, normalization, loss, or quantile values.
`SIMCAST_DATA_DIR` and `SIMCAST_DEVICE` change the data location and device
without changing cache identity.

Fit and evaluate one method, optionally alongside a reference:

```bash
uv run singular \
  --base configs/bases/liander2024/transformer.yaml \
  --method configs/methods/m4_conditional_kernel.yaml \
  --reference-method configs/methods/m0_independent.yaml
```

Run a declared study, evaluate it, and report it:

```bash
uv run composite --config configs/venues/lab/quick_all_methods.yaml
uv run evaluate  --composite-config configs/venues/lab/quick_all_methods.yaml
uv run report    --config configs/reports/lab_main.yaml \
  --run-root runs/lab/quick_all_methods/<run-id>
```

`composite` fits every declared base/method/seed, runs declared evaluations,
then generates declared reports. `evaluate` adds or completes evaluations;
`report` regenerates reports or applies a standalone report design, computing
method summaries and moving-block paired effects against its reference.
Missing caches are built automatically; `uv run cache` builds one in advance.
Every command documents its options with `--help`.

Outputs nest under one run root:

```text
runs/<venue>/<composite>/<run-id>/
  models/<base-id>/<method-id>/<seed-label>/
  evaluations/<evaluation-id>/<base-id>/<method-id>/<seed-label>/
  reports/<report-id>/<evaluation-id>/
```

Long composites accept `--run-id <id>` and `--resume`; resume is accepted only
when the stored resolved-composite hash is identical, and partial outputs are
never silently overwritten. The `lab` venue has deliberately small budgets for
understanding the workflow; its results are preliminary by design.

## Evaluation

Aggregate quantiles, pinball loss, CRPS, interval coverage, width, and score,
and WIS are evaluated on the selected cross-entity statistic of the scenarios
$\widetilde A_{g,\tau}^{(i,m)}$. Energy Score uses the empirical all-pairs
estimator on the selected 512-member joint ensemble; Variogram Score assesses
pairwise contrasts on the active subset. Each method is evaluated independently under the same
declared sampling design with common random numbers. Cross-method effects are
a reporting concern: the report selects the reference method and uses an
origin-level moving-block bootstrap to retain temporal dependence.

Every run records the complete ordered entity universe, revisions, resolved
configurations, hashes, seeds, environment and Git metadata, fitted methods,
per-case active counts, evaluation manifests, and completion state.

## Validation

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
```

These checks use synthetic data and temporary artifacts. They do not run
Chronos inference or scientific experiments.

## Attribution and license

The [Liander2024 Energy Forecasting Benchmark](https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark)
is published by OpenSTEF under CC BY 4.0. Chronos is Copyright Amazon.com, Inc.
or its affiliates and licensed under Apache-2.0. Simcast source is MIT licensed.
