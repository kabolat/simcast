# User guide: running singular and composite experiments

This is the practical entry point for installation, data, caches, commands,
notebooks, and outputs. The [configuration reference](configuration_reference.md)
defines every valid YAML field, type, option, and constraint; the scientific
chapters explain the underlying estimands and methods. For the scientific
reading sequence, begin with the
[research problem](../scientific/01_research_problem.md).

## Scientific map

Use this guide when you need to act on a declared experiment; use the linked
scientific chapter when you need to understand why that action is valid. The
same base, method, and composite files connect both perspectives.

| If you are about to... | Read this scientific account first | Then return here for... |
|---|---|---|
| choose a group, target, weather-information policy, or forecast origins | [Data and information sets](../scientific/02_data_and_information_set.md) | data setup and base selection in Sections 1 and 3 |
| build or inspect frozen Chronos quantiles, crossings, repair, or pseudo-PITs | [Frozen Chronos forecasts and finite PITs](../scientific/03_chronos_and_pit.md) | cache construction in Section 4 and notebooks in Section 11 |
| select M0, M1, M2, M3, or M4 | [Dependence models](../scientific/04_dependence_models.md) | a singular method run in Section 5 or a composite in Section 6 |
| interpret fitting diagnostics, scenarios, or proper scores | [Dependence fitting, sampling, and scoring](../scientific/05_training_sampling_scoring.md) | output inspection in Section 10 |
| understand the declaration-to-evidence chain | [Scientific workflow](../scientific/06_scientific_workflow.md) | cache, run, resume, and notebook procedures below |
| prepare a table, figure, or claim from a completed study | [Experiments, reporting, and result interpretation](../scientific/07_experiments_and_results.md) | report locations and inspection in Section 10 |

The [configuration reference](configuration_reference.md) is the authoritative
field-level companion: it specifies which YAML choices are admissible once the
scientific choice has been made. The [artifact reference](artifact_reference.md)
identifies the records that preserve it.

## 1. Choose the appropriate interface

| Objective | Interface | Main input | Main output |
|---|---|---|---|
| Inspect data, frozen marginals, PITs, or one method interactively | notebook | selected base and, where relevant, method YAML | explanatory calculations and figures |
| Construct only frozen Chronos marginals and PITs | cache command | one base YAML | `artifacts/cache/<fingerprint>/` |
| Fit and evaluate one copula hypothesis | singular command | one base and one method YAML | `runs/singular/.../` |
| Reproduce a declared comparison or sensitivity study | composite command | one venue YAML | `runs/<venue>/.../`, `reports/<venue>/.../` |

## 2. Scientific objects

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

The reasons for holding the base fixed are developed in the
[research problem](../scientific/01_research_problem.md) and
[scientific workflow](../scientific/06_scientific_workflow.md). The
configuration reference translates those roles into valid fields; it is not a
second definition of the statistical estimand.

Implementation: `BaseExperimentConfig`, the discriminated `MethodConfig`
union, and `CompositeExperimentConfig` are defined in
`src/simcast/config.py`. Unknown or cross-method fields are errors.

## 3. Environment, data, and pinned sources

Create the locked environment from the repository root:

```bash
uv sync --group dev
bash scripts/setup_chronos.sh
```

The second command installs the pinned Chronos source with the minimal
representation-output patch used by the experiment. It does not modify the
forecast quantiles.

Chronos weights are retrieved through the Hugging Face cache. An optional
`HF_TOKEN` improves Hub rate limits; never store it in YAML or Git:

```bash
export HF_TOKEN=...
```

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

## 4. Frozen-marginal construction

Construct the frozen Chronos marginal record without fitting a dependence
model:

```bash
uv run python -m simcast.cli.build_cache \
  --base configs/bases/liander2024/transformer.yaml
```

This creates the Chronos native quantile grids, output-patch representations,
and finite-quantile PIT observations declared by the base. The compatible
canonical cache location is determined by the base's marginal fingerprint.
Before interpreting or changing a PIT-related option, read
[Chapter 3](../scientific/03_chronos_and_pit.md): the cache stores the
configured finite-quantile law, which may be discretized or piecewise linear.
Choose a specific directory only when needed:

```bash
uv run python -m simcast.cli.build_cache \
  --base configs/bases/liander2024/solar_park.yaml \
  --output-dir artifacts/cache/solar_park
```

The command will not replace an existing cache. Add `--overwrite` only when
you intentionally wish to reconstruct that exact output directory. No
dependence fitting, scenario generation, or report generation occurs in this
step.

Implementation: `simcast.cli.build_cache` and
`build_cache_from_config` in `src/simcast/cli/build_cache.py`.

## 5. Singular experiment

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
The mathematical M4 hypothesis is derived in
[Chapter 4](../scientific/04_dependence_models.md); the likelihood, projection,
and scores are defined in [Chapter 5](../scientific/05_training_sampling_scoring.md).

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

## 6. Composite experiment

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

Treat this command as the execution of a predeclared comparison, rather than
as a generic model sweep. [Chapter 6](../scientific/06_scientific_workflow.md)
explains the resulting evidence chain and
[Chapter 7](../scientific/07_experiments_and_results.md) specifies which paired
comparisons and uncertainty statements are scientifically interpretable.

## 7. Laboratory budgets

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

## 8. Cache compatibility

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

## 9. Venues, identifiers, and resume

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

## 10. Outputs and interpretation

A singular directory records the two resolved roles, fitted method(s), one
evaluation, and its manifest. A composite directory records the resolved
composite, exact expansion, role hashes, environment and Git metadata,
shared deterministic fits, seed-specific conditional fits, evaluations,
completion state, and log. Its report directory contains pooled per-origin
records, method summaries, paired effects, and figures. The exact cache, fit,
evaluation, and report schemas are listed in the
[artifact reference](artifact_reference.md).

For a current composite, begin in
`reports/<venue>/<composite>/<run-id>/`: `method_summary.csv` gives the primary
aggregate pinball summaries; `paired_effects.csv` contains the declared
moving-block contrasts; `per_origin_metrics.parquet` is the origin-level input
to those contrasts; and `method_comparison.png`/`.pdf` visualize the primary
summary. Each is a report of the resolved composite, not a free-standing
result. Their exact columns and retained identifiers are defined in the
[artifact reference](artifact_reference.md#composite-experiment-and-report-roots).

### Reading a completed report

Begin with the resolved declaration and completion manifest, then inspect the
evidence in the following order:

1. Confirm the base fingerprint, ordered entity IDs, $K_g$, complete-case mask,
   and marginal diagnostics. These determine the common forecast experiment.
2. For M2--M4, examine each seed's fitting curve and validation pseudo-NLL
   before aggregating its test scores.
3. Read pooled score summaries together with the per-origin paired effects and
   moving-block intervals. For negatively oriented scores, a negative effect
   relative to the declared reference favours the method.
4. Use lead-wise records to test whether an overall result hides horizon
   heterogeneity. Use correlation and scenario figures to explain behaviour,
   not to select a method after the fact.
5. Separate principal comparisons from explicitly declared sensitivities and
   record any limitation of the fixed marginal, finite-quantile, same-lead
   Gaussian-copula design.

This is the operational version of the reporting discipline in
[Chapter 7](../scientific/07_experiments_and_results.md). That chapter states
what must accompany a scientific claim; this guide tells you where to find the
supporting records.

Lower scores are better. Interpret paired effects with their temporal
moving-block intervals, not only point estimates. Inspect marginal diagnostics
before attributing aggregate error to dependence. A model comparison is valid
only when the base fingerprint and complete entity ordering coincide.

Historical artifacts can be inspected with
`simcast.reporting.read_evaluation_artifacts`. The reader returns original JSON
payloads without validation migration or rewriting. Historical execution
commands and monolithic configurations are intentionally unsupported.

## 11. Notebooks

Open notebooks through the project environment:

```bash
uv run jupyter lab
```

Start with the [Notebook guide](../../notebooks/README.md). Method-specific
notebooks are in `notebooks/dependence_methods/`. Notebook 02 accepts one
selected `BASE_FILE`, constructs that cache when absent, and uses that same
cache throughout. If the selected cache has no quantile crossings, it reports
that result rather than attempting to load a different group.

If the Simcast environment is not available as a Jupyter kernel, install it
once:

```bash
uv run python -m ipykernel install --user --name simcast --display-name "Python (simcast)"
```

## 12. Advanced configuration and temporary overrides

Most users can select a supplied base, method, or composite YAML file without
changing its structure. The following mechanisms are useful when maintaining
new configuration files or making one temporary change.

### Inheriting a common configuration

At the top of a YAML file, `extends` names one or more parent YAML files. The
child starts from the parent settings and then replaces only the fields it
states explicitly:

```yaml
extends: common.yaml
kind: base
id: liander2024_transformer
data:
  entity_type: transformer
```

Here the transformer base inherits the common data, Chronos, PIT, and sampling
settings, while supplying its own entity type and complete ordered group. A
relative parent path is interpreted relative to the child file. When several
parents are listed, they are applied in order. Nested mappings combine by
field; lists, such as `ordered_entity_ids`, are replaced as whole lists.

### Reading a local value from the environment

YAML can refer to an environment variable when a setting depends on the local
machine rather than the scientific design:

```yaml
local_dir: ${SIMCAST_DATA_DIR:-data/liander2024}
```

This means “use `SIMCAST_DATA_DIR` if it has been set; otherwise use
`data/liander2024`.” It is why setting `SIMCAST_DATA_DIR` before a command or
Jupyter session changes the data location without editing the base YAML.

### Making one temporary command-line change

`--set` changes one nested YAML value for one command; it never edits the YAML
file. The field name follows the YAML nesting with dots, and the value is read
as YAML. For example, `pit.dependence_transform=training_frequency` means the
same setting as the nested YAML block shown below:

```bash
uv run python -m simcast.cli.build_cache \
  --base configs/bases/liander2024/transformer.yaml \
  --set pit.dependence_transform=training_frequency
```

```yaml
pit:
  dependence_transform: training_frequency
```

Examples:

```bash
--set chronos.device=cpu
--set pit.mode=linear_interpolation
--set sampling.num_samples=2048
--set evaluation.energy_score=false
--set covariates.future_weather_source=oracle
```

`pit.mode=linear_interpolation` changes both historical PIT construction and
entity-scenario projection, so it changes the marginal fingerprint and uses a
different cache. It cannot be combined with
`pit.dependence_transform=training_frequency`, which is defined only for the
default discretized cells. The mathematical distinction is derived in
[Chapter 3](../scientific/03_chronos_and_pit.md#4-two-finite-quantile-pit-constructions),
and every PIT option is enumerated in the
[configuration reference](configuration_reference.md#25-finite-quantile-marginal-law-and-pit).

For a method field, use a command or composite entry that resolves that method;
for example, `model.latent_rank=8` is valid for M2/M3 and invalid for M4.
Unknown fields and cross-method fields fail validation.

## 13. Common situations

| Situation | Interpretation and action |
|---|---|
| `ModuleNotFoundError: simcast` in Jupyter | select/install `Python (simcast)` and start Jupyter with `uv run jupyter lab` |
| Dataset file missing | set `SIMCAST_DATA_DIR` or run `download_data` for the selected base |
| Cache file or `.zmetadata` missing | the cache is absent or incomplete; build it, adding `--overwrite` only for a deliberate replacement |
| Hugging Face unauthenticated warning | public retrieval is permitted but rate-limited; optionally set `HF_TOKEN` |
| No crossing figure in notebook 02 | the selected cache has no representative crossing; this is a valid result |
| Configuration rejected | consult [Configuration reference](configuration_reference.md); unknown or cross-method fields are invalid |

## 14. Scientific validation without experiments

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
```

These commands test schemas, matrix properties, complete-vector invalidation,
permutation equivariance, finite PIT/projection laws, cache identity, expansion,
resume, and reporting on synthetic temporary data. They do not run Chronos,
fit the empirical experiments, or alter existing `runs/` and `reports/`.
