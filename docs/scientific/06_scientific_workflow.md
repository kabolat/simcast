# Scientific workflow

**Technical counterparts.** This chapter states the scientific sequence and
the evidence it must leave behind. The [usage guide](../technical/usage_guide.md)
turns that sequence into commands and notebooks; the
[configuration reference](../technical/configuration_reference.md) defines the
declarations used below; and the
[artifact reference](../technical/artifact_reference.md) defines the resulting
records. The interpretation of a completed comparison belongs in
[Chapter 7](07_experiments_and_results.md).

## 1. Objects and transformations

The experiment is a sequence of statistical transformations:

$$
\mathcal D_g
\xrightarrow{\text{information rule}}
\{\mathcal I^{(i)},\mathbf y_g^{(i)}\}_{i=1}^N
\xrightarrow{\text{frozen Chronos-2}}
\{\widehat y_{k,\tau,q_j}^{(i)}\}
\xrightarrow{f^{\mathrm{PIT}}}
\{\mathbf z_{g,\tau}^{(i)}\}
\xrightarrow{\text{dependence fit}}
\{R_{g,\tau}^{(i)}\}
\xrightarrow{\text{copula sampling}}
\{\widetilde A_{g,\tau}^{(i,m)}\}
\xrightarrow{\text{scores}}
\widehat{\mathcal R}.
$$

The base declaration fixes every transformation through the pseudo-PIT. The
method declaration changes only the map to $R_{g,\tau}^{(i)}$. A composite
declaration makes the repeated fits explicit; an evaluation declaration fixes
the sampling design and scores; and a report declaration fixes the reference
and the uncertainty analysis of paired differences.
The mathematical definitions of the information set, PIT, copulas, and scores
are given in Chapters [2](02_data_and_information_set.md),
[3](03_chronos_and_pit.md), [4](04_dependence_models.md), and
[5](05_training_sampling_scoring.md), respectively.

## 2. Scientific declaration and operational evidence

The five configuration objects are not merely an interface convention. They
separate quantities that must remain common from the hypothesis that is allowed
to vary, and both from the way the outcome is scored and compared. This
correspondence makes a completed result auditable.

| Scientific object | What it fixes or varies | Operational declaration | Evidence retained |
|---|---|---|---|
| Base | population, ordered $\mathcal E_g$, $\mathcal I^{(i)}$, marginal grid, PIT rule, and split | a `base` YAML file; [base fields](../technical/configuration_reference.md#2-base-configuration) | fingerprinted PIT library and resolved base |
| Method | one map from features or historical scores to $R_{g,\tau}^{(i)}$ | a `method` YAML file; [method fields](../technical/configuration_reference.md#3-method-configuration) | resolved method, fitted parameters or checkpoint, fitting diagnostics |
| Composite | declared bases, method variants, and seeds | a `composite` YAML file; [composite fields](../technical/configuration_reference.md#4-composite-configuration) | expansion manifest and run status |
| Evaluation | scenario count and seed, cross-entity statistic $T$, quantile and interval levels, and scores | an `evaluation` YAML file; [evaluation fields](../technical/configuration_reference.md#7-evaluation-and-report-configuration) | per-method evaluation records and manifest |
| Report | reference method and block-bootstrap design | a `report` YAML file; [report fields](../technical/configuration_reference.md#7-evaluation-and-report-configuration) | method summaries, paired effects, and figures |

The [usage guide's interface map](../technical/usage_guide.md#1-choose-the-appropriate-interface)
shows which command creates each record. The separation prevents a method
choice, such as M2 rank, from changing the upstream marginal forecast, and
prevents a reporting preference from being mistaken for a change in the
data-generating experiment.

## 3. Complete cases

For group $g$, every tensor has the fixed entity axis
$\mathcal E_g=[k_1,\ldots,k_{K_g}]$. Define entity validity
$V_{k,\tau}^{(i)}\in\{0,1\}$ and group validity

$$V_{g,\tau}^{(i)}=\prod_{k\in\mathcal E_g}V_{k,\tau}^{(i)}.$$

Only cases with $V_{g,\tau}^{(i)}=1$ enter dependence fitting or evaluation.
The dataset object enumerates these complete $(i,\tau)$ pairs; the collator
stacks them without selecting entities; and the fitter asserts an entity
dimension of exactly $K_g$. Evaluation checks that every supplied correlation
is $K_g\times K_g$. The formal full-group protocol is introduced in
[Chapter 1](01_research_problem.md); its configuration safeguards are listed
in the [base reference](../technical/configuration_reference.md#21-complete-group-and-data).

Example: if one of 15 transformer observations is missing at lead 32 of origin
$i$, all 15 components of that one case are excluded. The other leads remain
eligible if complete. This is a missing-vector rule, not an opportunity to
forecast a smaller group.

Implementation: `DependenceDataset`, `DependenceCollator`,
`ConditionalTrainer.expected_num_entities`, and `evaluate_from_config` enforce
the rule independently. The resulting `pit_valid` mask and declared $K_g$ are
stored as described in the [artifact reference](../technical/artifact_reference.md#pit-library-directory).

## 4. Frozen-marginal record

Chronos is called separately for each physical entity. The record stores named
axes for origins, entities, leads, native quantiles, output patches, and hidden
coordinates. Test realizations and derived test pseudo-PIT values are stored in
a sealed Zarr group inaccessible through training mode.

The record is identified by the base fingerprint, not by a manually chosen
name. The fingerprint covers the complete ordered group, pinned data, temporal
protocol, information set, chronological split, Chronos revisions, and PIT
construction. This allows one expensive marginal computation to support all
five dependence methods without confusing scientifically different forecasts.
The exact cache-compatibility rule and command are documented in the
[usage guide](../technical/usage_guide.md#4-frozen-marginal-construction) and
[configuration reference](../technical/configuration_reference.md#6-cache-fingerprint).

Example: changing M2 rank does not rebuild Chronos. Changing vintage weather to
oracle changes $\mathcal I^{(i)}$ and therefore requires a different cache.

Implementation: `build_cache_from_config` constructs the record;
`base_fingerprint` and `locate_compatible_cache` govern reuse.

## 5. Dependence fitting

M0 writes the identity specification. M1 estimates shrinkage correlation from
training pseudo-scores. M2--M4 minimize Gaussian-copula negative
pseudo-log-likelihood over all complete training cases and choose a checkpoint
by chronological validation pseudo-NLL. Test outcomes do not influence this
choice. The likelihood and each parameterization are derived in Chapters
[4](04_dependence_models.md) and [5](05_training_sampling_scoring.md).

The public term is **dependence fitting** because M0 is specified rather than
trained and M1 has a closed-form estimator. Inside gradient-based code,
training and optimization retain their ordinary mathematical meanings.

M4 follows exactly the same partition and stopping semantics as M2 and M3. A
small preliminary budget is an explicit composite override applicable to any
of these three methods. The accepted fields and their scientific roles are in
the [method reference](../technical/configuration_reference.md#3-method-configuration).

## 6. Singular transformation

`run_singular` combines one base with one method, obtains the compatible
marginal record, fits that method, and evaluates it. An optional explicit
reference is fitted and evaluated with common random numbers. The saved base
and method remain separate, preserving the logic of controlled comparison.

```bash
uv run singular \
  --base configs/bases/liander2024/transformer.yaml \
  --method configs/methods/m3_set_aware_low_rank.yaml \
  --reference-method configs/methods/m0_independent.yaml
```

This is appropriate for studying one conditional hypothesis, one group, or a
single diagnostic. The [singular-experiment instructions](../technical/usage_guide.md#5-singular-experiment)
state the created files and the [reporting chapter](07_experiments_and_results.md)
states what a paired result can support scientifically. A singular run is not a
substitute for the explicitly declared repetitions and uncertainty analysis of
a composite.

## 7. Composite transformation

The `composite` command resolves only explicitly listed entries. Each fit is the tuple

$$c=(\text{base entry},\text{method entry},\text{seed or deterministic}).$$

The expansion manifest is an auditable enumeration of these tuples. Identical
deterministic fits are shared. Conditional fits remain seed-specific. Each
evaluation scores every selected fit independently under one declared design,
with identical cases and common random numbers across methods, so any two
evaluated methods can later be paired. The composite schema makes this claim
checkable rather than implicit; see the
[configuration reference](../technical/configuration_reference.md#4-composite-configuration).

```bash
uv run composite --config configs/venues/<venue>/main.yaml
```

A report then chooses a reference $b$. The paired origin-level loss for method
$a$ is

$$d_{a,b,g}^{(i)}=S_{a,g}^{(i)}-S_{b,g}^{(i)},$$

where valid leads are averaged within origin and seeds are averaged within
method and origin. A moving-block bootstrap samples
chronological blocks of $d^{(i)}$ to retain short-range temporal dependence.
The command, venue paths, and repeatable-resume procedure are given in the
[composite section of the usage guide](../technical/usage_guide.md#6-composite-experiment).

## 8. Resume and immutability

Before a composite starts, the fully resolved base, method, override, and seed
definitions are hashed. Resume succeeds only if that digest equals the stored
manifest digest. A cell marked complete is skipped only when its recorded
outputs pass completion checks. A changed composite must use a new run
identifier. The exact venue and resume interface is specified in the
[usage guide](../technical/usage_guide.md#9-venues-identifiers-and-resume).

## 9. From completed computation to reported evidence

A result is a chain of records, rather than a number copied from a console.
The report must preserve the connection

$$
\text{declared comparison}
\longrightarrow
\text{fitted dependence law}
\longrightarrow
\text{common-case scores}
\longrightarrow
\text{paired effect and uncertainty}.
$$

First inspect the resolved base and method files, the entity ordering, and the
marginal fingerprint; next inspect fitting diagnostics for every conditional
seed; then inspect score summaries, lead-wise patterns, and paired
origin-level effects. The report root contains the consolidated records,
method summaries, paired effects, and figures. Its schema is described in the
[artifact reference](../technical/artifact_reference.md#composite-experiment-and-report-roots),
while the practical inspection order is in the
[usage guide](../technical/usage_guide.md#10-outputs-and-interpretation).

This order matters. A lower aggregate score is not evidence about a dependence
method until the comparison is known to share the same fixed marginal grid,
complete-case mask, sampling design, and test origins. Chapter
[7](07_experiments_and_results.md) defines the estimands and claim discipline
for that evidence.

## 10. Relation between CLI and notebooks

The CLI is the authoritative route for full reproduction. Notebooks use the
same role-specific loaders, fingerprint lookup, runtime adapter, dependence
fitter, and evaluator. They expose intermediate equations and arrays for
interpretation, while write-producing cells default to disabled. The
[notebook section of the usage guide](../technical/usage_guide.md#11-notebooks)
explains how to open them and the [notebook guide](../../notebooks/README.md)
maps them to the scientific chapters.

Thus the two interfaces differ in exposition, not in statistical
transformation. A notebook calculation is comparable with a CLI result only
when its resolved base, method, seed, and cache fingerprint coincide.
