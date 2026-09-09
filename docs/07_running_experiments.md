# Reproducing singular and composite experiments

## 1. Scientific objects

The command line distinguishes three objects that answer different questions.

1. A **base** fixes the observed sample, complete entity group
   $\mathcal E_g$, information set $\mathcal I^{(i)}$, Chronos-2 marginal
   quantiles, PIT rule, scenario law, and evaluation estimands.
2. A **method** specifies one copula hypothesis M0--M4. It cannot alter the
   marginal quantile values.
3. A **composite** explicitly states which base--method combinations and neural
   seeds form a larger comparison, together with its reference and uncertainty
   analysis.

This separation prevents irrelevant parameters from entering a hypothesis. M0
has no features or optimization. M1 has only its static covariance estimator.
M2, M3, and M4 each have features, a model parameterization, and an
optimization design.

Implementation: `BaseExperimentConfig`, the discriminated `MethodConfig`
union, and `CompositeExperimentConfig` are defined in
`src/simcast/config.py`. Unknown or cross-method fields are errors.

## 2. Environment and pinned sources

Create the locked environment from the repository root:

```bash
uv sync --group dev
bash scripts/setup_chronos.sh
```

The second command installs the pinned Chronos source with the minimal
representation-output patch used by the experiment. It does not modify the
forecast quantiles.

The dataset defaults to `data/liander2024` inside this repository. An absolute
location may be selected explicitly:

```bash
export SIMCAST_DATA_DIR=/absolute/path/to/liander2024
export SIMCAST_DEVICE=cuda
```

Download only the files required by one base:

```bash
uv run python -m simcast.cli.download_data \
  --base configs/bases/liander2024/transformer.yaml
```

Example: selecting the solar base downloads the solar group and applies no
model fitting. Its base declares isotonic quantile repair; changing to the
transformer base changes both $\mathcal E_g$ and $K_g$.

## 3. Singular experiment

To estimate and evaluate exactly M4 for the transformer group:

```bash
uv run python -m simcast.cli.run_singular \
  --base configs/bases/liander2024/transformer.yaml \
  --method configs/methods/m4_conditional_kernel.yaml
```

The transformation is:

$$
(\text{base},\text{method})
\longrightarrow
\{\widehat R_{g,\tau}^{(i)}\}
\longrightarrow
\{\widetilde{\mathbf Y}_{g,\tau}^{(i,m)}\}
\longrightarrow
\text{scores}.
$$

The command locates or constructs the compatible frozen-marginal cache, fits
only M4, then evaluates only M4. M4 consumes every valid complete training
vector and has no method-specific origin limit or implicit epoch reduction.

To estimate a paired contrast against M0:

```bash
uv run python -m simcast.cli.run_singular \
  --base configs/bases/liander2024/transformer.yaml \
  --method configs/methods/m4_conditional_kernel.yaml \
  --reference-method configs/methods/m0_independent.yaml
```

Both methods use the same complete test cases, fixed marginal grids, scenario
count, evaluation seed, and case-keyed base-normal draws. This is the common
random numbers design. It reduces Monte Carlo noise in the score difference;
it does not remove parameter-estimation uncertainty.

Implementation: `run_singular` in `src/simcast/cli/run_singular.py` performs
these transformations and stores `resolved_base.yaml`,
`resolved_method.yaml`, the optional resolved reference, and
`singular_manifest.json`.

## 4. Composite experiment

A composite is an explicit collection rather than an implicit parameter grid:

```yaml
kind: composite
name: main
venue: example

bases:
  - id: transformer
    config: ../../bases/liander2024/transformer.yaml

experiments:
  - id: m0
    method: ../../methods/m0_independent.yaml
  - id: m4
    method: ../../methods/m4_conditional_kernel.yaml
    seeds: [11, 23, 37]

analysis:
  reference: m0
  bootstrap_replicates: 10000
  primary_block_length: 7
  sensitivity_block_lengths: [3, 14]
```

Run it with:

```bash
uv run python -m simcast.cli.run_composite \
  --config configs/venues/<venue>/main.yaml
```

If five bases are listed, the M0 entry creates five deterministic cells; an M4
entry with ten seeds creates fifty fitted cells. Nothing else is generated.
Repeated entries with distinct IDs express feature ablations or sensitivity
variants. Overrides are validated against the referenced method type, so
`model.latent_rank` is valid for M2/M3 and invalid for M4.

M0 and M1 fits are reused across neural seeds when their resolved scientific
definitions are identical. M2, M3, and M4 have equal full-partition status.
All five appear in generic summaries whenever declared.

## 5. Laboratory budgets

Preliminary computation is expressed at the composite level:

```yaml
- id: m4_quick
  method: ../../methods/m4_conditional_kernel.yaml
  overrides:
    optimization:
      epochs: 2
      patience: 1
```

The same override is meaningful for M2 or M3. It says that the *experiment*
uses a reduced optimization budget; it does not define a special method class.
The supplied `configs/venues/lab/quick_all_methods.yaml` applies equal reduced
budgets to all three gradient-fitted methods.

## 6. Cache compatibility

Let $b$ denote a resolved base and let $h(b)$ be the SHA-256 digest of:

$$
h(b)=h(\mathcal E_g,\text{data revision},\text{forecast protocol},
\mathcal I,\text{split},\text{Chronos revision},\text{PIT construction}).
$$

The cache path normally contains the first 12 hexadecimal characters of
$h(b)$. The path is only a convenience. Before reuse, metadata are normalized
and hashed again. If a differently named directory is the unique compatible
cache it may be reused; a plausible name with incompatible metadata is
rejected; multiple compatible candidates are treated as ambiguous.

Method, venue, score presentation, and report settings do not determine the
frozen marginal record and therefore do not enter $h(b)$. Local data location,
GPU choice, and Chronos batch size are operational and likewise excluded.

Implementation: `base_fingerprint` and `locate_compatible_cache` implement
this rule.

## 7. Venues, identifiers, and resume

A venue is a named reproducible research workspace. It is not a Python virtual
environment. A composite file must be physically located below
`configs/venues/<venue>/` and its `venue` field must match that directory.
Venue, composite, and run identifiers are safe lowercase slugs.

```text
runs/<venue>/<composite>/<run-id>/
reports/<venue>/<composite>/<run-id>/
```

Supply a fixed identifier for a long run:

```bash
uv run python -m simcast.cli.run_composite \
  --config configs/venues/<venue>/main.yaml \
  --run-id replication_01
```

After interruption, repeat the command with `--resume`. Resume first recomputes
the fully resolved composite hash. A mismatch is rejected before fitting. A
validated complete cell is skipped; missing or incomplete cells continue. No
existing incompatible directory is overwritten.

## 8. Outputs and interpretation

A singular directory records the two resolved roles, fitted method(s), one
evaluation, and its manifest. A composite directory records the resolved
composite, exact expansion, role hashes, environment and Git metadata,
shared deterministic fits, seed-specific conditional fits, evaluations,
completion state, and log. Its report directory contains pooled per-origin
records, method summaries, paired effects, and figures.

Lower scores are better. Interpret paired effects with their temporal
moving-block intervals, not only point estimates. Inspect marginal diagnostics
before attributing aggregate error to dependence. A model comparison is valid
only when the base fingerprint and complete entity ordering coincide.

Historical artifacts can be inspected with
`simcast.reporting.read_evaluation_artifacts`. The reader returns original JSON
payloads without validation migration or rewriting. Historical execution
commands and monolithic configurations are intentionally unsupported.

## 9. Scientific validation without experiments

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
```

These commands test schemas, matrix properties, complete-vector invalidation,
permutation equivariance, finite PIT/projection laws, cache identity, expansion,
resume, and reporting on synthetic temporary data. They do not run Chronos,
fit the empirical experiments, or alter existing `runs/` and `reports/`.
