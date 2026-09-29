# Simcast

Simcast studies same-lead, cross-entity forecast-error dependence while holding
Chronos-2 marginal forecasts fixed. For forecast instance $i$, lead
$\tau\in[H]$, and complete static physical group
$\mathcal E_g=\{k_1,\ldots,k_{K_g}\}$, Chronos supplies fixed marginal
quantile grids for the random variables $Y_{k,\tau}^{(i)}$. A dependence method
supplies only the copula $C_{g,\tau}^{(i)}$:

$$
P\!\left(Y_{k,\tau}^{(i)}\le y_k,\ k\in\mathcal E_g\mid\mathcal I^{(i)}\right)
=C_{g,\tau}^{(i)}\!\left(\{F_{k,\tau}^{(i)}(y_k)\}_{k\in\mathcal E_g}\right).
$$

The evaluation targets a cross-entity statistic of the complete group: the
simple sum $\sum_k Y_k$, absolute sum $\sum_k |Y_k|$, maximum $\max_k Y_k$,
or absolute maximum $\max_k |Y_k|$. The evaluation YAML selects the statistic;
`sum` is the default.

Every training, validation, and test case uses all $K_g$ entities. If one
entity is invalid at $(i,\tau)$, the complete vector is invalid; the group is
never reduced.

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
  complete group with position-free entity self-attention.
- M4: conditional RBF-kernel Gaussian copula in a learned embedding space.

M4 has the same experimental status as M2 and M3. All three use the complete
training partition, chronological validation, and explicitly configured
optimization budgets. A reduced laboratory budget is a property of a
composite experiment, not of a method.

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
  bases/liander2024/   # data, group, information set, Chronos, PIT
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
pairwise contrasts. Each method is evaluated independently under the same
declared sampling design with common random numbers. Cross-method effects are
a reporting concern: the report selects the reference method and uses an
origin-level moving-block bootstrap to retain temporal dependence.

Every run records the complete ordered entity set, revisions, resolved
configurations, hashes, seeds, environment and Git metadata, fitted methods,
evaluation manifests, and completion state.

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
