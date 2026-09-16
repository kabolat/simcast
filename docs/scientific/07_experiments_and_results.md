# Experiments, reporting, and result interpretation

**Technical counterparts.** The [usage guide](../technical/usage_guide.md#6-composite-experiment)
explains how to execute a declared composite, the
[configuration reference](../technical/configuration_reference.md#4-composite-configuration)
defines its bases, repetitions, reference, and bootstrap settings, and the
[artifact reference](../technical/artifact_reference.md#composite-experiment-and-report-roots)
defines the records from which a report is made. The scientific workflow in
[Chapter 6](06_scientific_workflow.md) explains why these records must remain
connected.

## 1. Controlled comparison

For each base, all methods receive the same fixed FM-derived marginal quantile
grid. The comparison varies only the Gaussian-copula correlation
$R_{g,\tau}^{(i)}$. The principal hierarchy asks:

1. Does residual spatial dependence matter? Compare M1--M4 with M0.
2. Is a persistent lead-specific structure sufficient? Compare conditional
   M2--M4 with M1.
3. Does full-group contextualization help within the low-rank family? Compare
   M3 with M2.
4. How does a learned nonnegative RBF similarity restriction compare with the
   signed low-rank families? Compare M4 with M2/M3.
5. How heterogeneous are these effects across physical entity groups and
   forecast leads?

None of these questions claims novelty for joint probabilistic energy
forecasting itself. The contribution is a controlled assessment of dependence
layers over fixed foundation-model-derived marginals.

## 2. From declared composite to scientific report

The scientific report is the final interpretation of a predeclared sequence,
not a separate numerical exercise. The evidence path is:

| Stage | Scientific question | Required declaration or record | Technical guide |
|---|---|---|---|
| Fixed comparison | What remains common across methods? | resolved base, ordered $\mathcal E_g$, marginal fingerprint, complete-case rule | [base configuration](../technical/configuration_reference.md#2-base-configuration) and [cache construction](../technical/usage_guide.md#4-frozen-marginal-construction) |
| Dependence hypothesis | Which $C_{g,\tau}^{(i)}$ is being tested? | resolved method file and, where applicable, seed-specific checkpoint | [method configuration](../technical/configuration_reference.md#3-method-configuration) |
| Declared repetition | Which groups, variants, seeds, and reference form the study? | resolved composite and expansion manifest | [composite configuration](../technical/configuration_reference.md#4-composite-configuration) |
| Common evaluation | Are the methods scored on the same outcomes and random draws? | evaluation manifest, validity mask, per-origin records | [evaluation schema](../technical/artifact_reference.md#evaluation-directory) |
| Uncertainty and interpretation | How large and how stable is the paired difference? | paired effects, block-bootstrap intervals, lead-wise and group-wise summaries | [outputs and interpretation](../technical/usage_guide.md#10-outputs-and-interpretation) |

This structure distinguishes a result from a convenient plot. A figure may
illustrate a correlation matrix or a scenario fan, but the principal numerical
claim comes from the declared common-case score and its paired origin-level
uncertainty. Reporting must therefore state the base, method entry, seed
aggregation, reference, score orientation, and validity population.

## 3. Principal methods

M0, M1, M2, M3, and M4 are all principal declared methods. The supplied main
composite lists all five bases and all five methods. M2--M4 each use the same
ten seeds and the same default 100-epoch maximum with patience 12. M4 has no
origin truncation, hidden epoch cap, special opt-in flag, or report exclusion.

Example expansion for five groups:

| Method | Fits per group | Total cells |
|---|---:|---:|
| M0 | 1 deterministic | 5 |
| M1 | 1 deterministic | 5 |
| M2 | 10 seeds | 50 |
| M3 | 10 seeds | 50 |
| M4 | 10 seeds | 50 |

The complete composite therefore contains 160 declared cells. This count is
determined directly by its YAML and is checked before execution.

## 4. What constitutes a comparable result

A row can enter one comparison only if the following are identical:

- base fingerprint and ordered entity IDs;
- test origin and lead identities;
- complete-vector validity mask;
- repaired or raw marginal quantile grid;
- scenario count, evaluation seed, and projection rule;
- score estimator.

Neural seed is a repetition of parameter estimation, not a new observation.
For uncertainty over forecast instances, seed-specific losses are first
aggregated within method and origin. The temporal bootstrap then resamples
origins in blocks.

## 5. Primary estimands

Let $S_{m,g}^{(i)}$ denote an origin-level score for method $m$, obtained after
averaging valid leads. Against reference $r$, define

$$d_{m,r,g}^{(i)}=S_{m,g}^{(i)}-S_{r,g}^{(i)}.$$

For negatively oriented scores, $d<0$ favors $m$. The report presents the
sample mean and moving-block confidence interval. Lead-wise tables reveal
whether an overall mean hides forecast-horizon heterogeneity.

Aggregate pinball loss and CRPS assess the distribution of
$A_{g,\tau}^{(i)}$. Coverage must be interpreted with interval width and proper
interval scores. Energy Score uses the empirical all-pairs estimator on the
selected 512-member joint ensemble; ensemble selection is distinct from the
formula. Variogram Score emphasizes pairwise entity contrasts.

The current generic composite report designates mean aggregate pinball as its
primary paired metric: `method_summary.csv` and `paired_effects.csv` therefore
summarize that score. The per-method evaluation records retain the other
configured scores, but a pooled paired interval for CRPS, Energy Score, or
Variogram Score must not be claimed unless the composite reporting procedure is
explicitly extended to aggregate that score with the same declared case and
bootstrap rules. See the [evaluation schema](../technical/artifact_reference.md#evaluation-directory)
and [report schema](../technical/artifact_reference.md#composite-experiment-and-report-roots).

## 6. Sensitivity and ablation declarations

Sensitivity analyses are separate explicit composites:

- feature ablation repeats M2, M3, and M4 entries with named feature overrides;
- PIT sensitivity changes only a base-level dependence transform;
- rank sensitivity varies rank only for M2/M3 and retains M4 as a declared
  non-rank reference;
- static sensitivity compares lead-specific and pooled M1;
- laboratory composites apply transparent reduced budgets to M2--M4 equally.

Strict method validation prevents nonsensical contrasts such as assigning a
factor rank to M4 or conditional features to M0.

## 7. Status of historical results

Any existing `runs/` and `reports/` directories that predate this
configuration migration are untouched, read-only history, not evidence for the
current composites. The tracked `results/` directory of pre-migration
exploratory summaries has been retired: some of its neural results were
produced with random entity-subset augmentation and were never full-group
evidence. It remains recoverable from Git history for audit purposes, but it
is not restored, relabeled, or regenerated.

No empirical experiment or report was run as part of the configuration
refactor. Consequently this chapter does not invent a new numerical headline
table. A current table becomes scientifically admissible only after the
explicit composite is run to completion and its manifest verifies the current
schema and full-group design.

Implementation: `read_evaluation_artifacts` provides read-only historical
inspection. `run_composite` creates new venue-scoped artifacts and includes M4
in its generic summaries, paired effects, and figures.

## 8. Reading and reporting a completed composite

Proceed in this order:

1. Verify `composite_manifest.json` has `status: complete` and the expected
   expansion/hash.
2. Verify each base's entity ordering and marginal fingerprint.
3. Inspect marginal calibration and crossing diagnostics before dependence
   scores.
4. Inspect convergence and validation pseudo-NLL for each M2--M4 seed.
5. Compare per-origin paired effects and their block-bootstrap intervals.
6. Examine lead-wise and group-wise heterogeneity.
7. Use representative correlation matrices for mechanism interpretation, not
   method selection.

An unfavorable or nonconvergent M4 result remains part of the declared
experiment. Excluding it after observing test performance would change the
scientific question and must be represented by a new composite, not by editing
the completed report.

### What a scientific report should contain

At minimum, report the following in the order that lets a reader assess the
claim rather than merely reproduce a number:

1. **Scope of the comparison.** Name the physical group, its ordered entity
   list and $K_g$, the forecast-origin population, and the complete-vector
   validity rule. These establish what the result describes.
2. **Fixed marginal experiment.** State the Chronos revision, the weather
   information policy, any isotonic repair, configured finite-quantile PIT law, and
   common scenario design. These are properties of the base, not of the
   winning method.
3. **Dependence alternatives.** State the exact M0--M4 entries, conditional
   feature/model choices, and the number of independently fitted seeds. A
   conditional result without its seed aggregation is incomplete evidence.
4. **Primary paired result.** For the predeclared primary score—currently mean
   aggregate pinball in the generic composite report—give the reference, the
   mean $d_{m,r,g}^{(i)}$, its orientation, the number of retained origins, and
   the moving-block interval with its configured block length. Report a second
   score as a paired result only when its aggregation and interval procedure
   have also been declared and generated.
5. **Heterogeneity and diagnostics.** Present lead-wise and group-wise patterns
   alongside marginal diagnostics and conditional-fitting diagnostics. Treat a
   correlation illustration as a mechanism diagnostic, not as proof of better
   forecasts.
6. **Sensitivity and limits.** Identify every separately declared sensitivity,
   distinguish it from the principal comparison, and state limitations such as
   finite native quantile support, Gaussian same-lead copulas, and the scope of
   the observed data population.

The [usage guide's reporting instructions](../technical/usage_guide.md#10-outputs-and-interpretation)
show how to find these records; the
[artifact reference](../technical/artifact_reference.md#evaluation-directory)
names their schemas. Do not merge rows across bases with different
fingerprints or present legacy subset-trained neural artifacts as a result of
this protocol.
