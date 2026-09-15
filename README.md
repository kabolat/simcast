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

The estimand of primary interest is the distribution of the spatial aggregate

$$A_{g,\tau}^{(i)}=\sum_{k\in\mathcal E_g}Y_{k,\tau}^{(i)}.$$

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

Human-authored YAML files represent three distinct objects:

```text
configs/
  bases/liander2024/       # data, group, information set, Chronos, PIT, estimands
  methods/                 # exactly one of M0--M4
  venues/<venue>/          # explicit composite experiments and analyses
```

A base defines the frozen-marginal experiment. A method file contains only
parameters meaningful to one dependence hypothesis. A composite explicitly
lists bases, method variants, repetitions, and paired analyses; no hidden
Cartesian grid is generated.

The marginal cache is identified by a SHA-256 fingerprint of the scientific
inputs that determine it: ordered group, data revision, forecast protocol,
covariates, split, Chronos revision, and PIT construction. Method, venue,
evaluation, and report settings do not enter this fingerprint. Cache directory
names are never trusted without compatible metadata.

## Setup with uv

For a task-oriented guide to installation, data, caches, commands, notebooks,
temporary overrides, and output locations, start with
[docs/technical/usage_guide.md](docs/technical/usage_guide.md).

```bash
uv sync --group dev
bash scripts/setup_chronos.sh
```

Pinned revisions are recorded in the base configuration and lockfile. The
Chronos patch exposes the already computed output-patch representation; it does
not change Chronos attention, normalization, loss, or quantile values.

The default data path is `data/liander2024`. It can be changed without changing
cache identity:

```bash
export SIMCAST_DATA_DIR=/absolute/path/to/liander2024
export SIMCAST_DEVICE=cuda
uv run python -m simcast.cli.download_data \
  --base configs/bases/liander2024/transformer.yaml
```

## Frozen-marginal cache

Construct the Chronos quantile, representation, and finite-PIT record for one
base without fitting a dependence method:

```bash
uv run python -m simcast.cli.build_cache \
  --base configs/bases/liander2024/transformer.yaml
```

The canonical cache location is determined by the scientific marginal
fingerprint of the base. To choose a location explicitly, use `--output-dir`.
An existing cache is never replaced unless `--overwrite` is supplied.

## Singular experiment

A singular experiment fits and evaluates exactly one selected method:

```bash
uv run python -m simcast.cli.run_singular \
  --base configs/bases/liander2024/transformer.yaml \
  --method configs/methods/m4_conditional_kernel.yaml
```

An explicit reference can be evaluated on the same valid cases, fixed
marginals, and case-keyed Gaussian draws:

```bash
uv run python -m simcast.cli.run_singular \
  --base configs/bases/liander2024/transformer.yaml \
  --method configs/methods/m4_conditional_kernel.yaml \
  --reference-method configs/methods/m0_independent.yaml
```

## Composite experiment

A composite performs only the experiments explicitly declared in its YAML:

```bash
uv run python -m simcast.cli.run_composite \
  --config configs/venues/<venue>/main.yaml
```

Long composites can use a stable identifier and resume validated completed
cells:

```bash
uv run python -m simcast.cli.run_composite \
  --config configs/venues/<venue>/main.yaml \
  --run-id replication_01 --resume
```

Resume is accepted only when the stored resolved-composite hash is identical.
Completed compatible cells are preserved; changed designs and partial outputs
are never silently overwritten.

A venue is a reproducible research workspace, not a Python environment.
Outputs follow:

```text
runs/<venue>/<composite>/<run-id>/
reports/<venue>/<composite>/<run-id>/
```

The repository includes `lab` composites with deliberately small explicit
budgets for understanding the workflow. Such results are preliminary by
design, regardless of which trainable method is selected.

## Evaluation

Aggregate scenario quantiles, coverage, interval width and score, WIS, and CRPS
are evaluated from $\widetilde A_{g,\tau}^{(i,m)}$. Energy Score uses the
empirical all-pairs estimator on the selected 512-member joint ensemble;
chunking changes only memory consumption. Variogram Score assesses pairwise
spatial contrasts. Composite effects are paired at the origin level and use a
moving-block bootstrap to retain temporal dependence.

Every run records the complete ordered entity set, revisions, resolved
role-specific configurations, hashes, seeds, environment and Git metadata,
fitted methods, evaluation manifests, and completion state. Historical
evaluation manifests remain readable through `simcast.reporting`, but the old
execution schemas are not accepted or rewritten.

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
