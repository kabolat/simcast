# Experimental questions and result interpretation

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

## 2. Principal methods

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

## 3. What constitutes a comparable result

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

## 4. Primary estimands

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

## 5. Sensitivity and ablation declarations

Sensitivity analyses are separate explicit composites:

- feature ablation repeats M2, M3, and M4 entries with named feature overrides;
- PIT sensitivity changes only a base-level dependence transform;
- rank sensitivity varies rank only for M2/M3 and retains M4 as a declared
  non-rank reference;
- static sensitivity compares lead-specific and pooled M1;
- laboratory composites apply transparent reduced budgets to M2--M4 equally.

Strict method validation prevents nonsensical contrasts such as assigning a
factor rank to M4 or conditional features to M0.

## 6. Status of historical results

Existing `runs/`, `reports/`, and tracked result files predate this
configuration migration and are intentionally unchanged. Some historical
neural results were produced with random entity-subset augmentation. They are
legacy exploratory evidence and must not be presented as results of the
current complete-group composites. Their original manifests remain readable,
but they are not silently relabeled or regenerated.

No empirical experiment or report was run as part of the configuration
refactor. Consequently this chapter does not invent a new numerical headline
table. A current table becomes scientifically admissible only after the
explicit composite is run to completion and its manifest verifies the current
schema and full-group design.

Implementation: `read_evaluation_artifacts` provides read-only historical
inspection. `run_composite` creates new venue-scoped artifacts and includes M4
in its generic summaries, paired effects, and figures.

## 7. Reading a completed composite

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
