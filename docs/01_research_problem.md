# Research question, estimand, and assumptions

## Scientific objective

Consider a fixed physical group of `K` electricity-producing or consuming
entities. At forecast origin `t_i`, the quantity of operational interest at
lead `tau` is the spatial aggregate

$$
A_{i,\tau}=\sum_{k=1}^{K}Y_{i,k,\tau}.
$$

A foundation model can issue useful marginal probabilistic forecasts
$F_{i,k,\tau}$ for each component $Y_{i,k,\tau}$ without determining their
joint distribution. The aggregate distribution nevertheless depends strongly
on cross-entity error co-occurrence. For example, independent positive errors
partly cancel in a sum, while spatially correlated positive errors do not.

The primary estimand is therefore the out-of-sample change in proper scoring
rules for $A_{i,\tau}$ when its joint scenarios are generated with a learned
cross-entity copula rather than independence, holding every
$F_{i,k,\tau}$ fixed.

The core comparison is:

- M0: independent Gaussian copula;
- M1: static, lead-specific Gaussian copula;
- M2: conditionally predicted low-rank Gaussian copula;
- M3: set-aware conditionally predicted low-rank Gaussian copula.

M4, a conditional RBF-kernel copula, is deliberately a bounded smoke test and
is not a fully trained or tuned competitor.

## Notation

| Symbol | Meaning | Code axis or object |
|---|---|---|
| $i\in\{1,\ldots,N\}$ | forecast-origin index | `origin` |
| $t_i$ | UTC forecast-origin timestamp | `origin_timestamp` |
| $k\in\{1,\ldots,K\}$ | physical entity index | `entity` |
| $\tau\in\{1,\ldots,H\}$ | one-based forecast lead | `lead` |
| $j\in\{1,\ldots,Q\}$ | native Chronos quantile index | `quantile` |
| $m\in\{1,\ldots,M\}$ | Monte Carlo scenario index | in-memory sample axis |
| $p(\tau)$ | Chronos output patch containing lead $\tau$ | `patch` |
| $Y_{i,k,\tau}$ | realized target | `true_y` |
| $\widehat q_{i,k,\tau,j}$ | Chronos value at probability $q_j$ | `quantile_prediction` |
| $u_{i,k,\tau}$ | discretized PIT pseudo-observation | `pit_u` |
| $z_{i,k,\tau}=\Phi^{-1}(u_{i,k,\tau})$ | Gaussianized PIT score | `pit_z` |
| $v_{i,k,\tau}$ | fixed feature vector for a conditional copula | constructed in memory |
| $R_{i,\tau}$ | $K\times K$ same-lead correlation matrix | model output |
| $A_{i,\tau}$ | observed spatial sum | `true_y.sum(entity)` |

Arrays in the repository use order `[origin, entity, lead, ...]` in the cache.
Training flattens valid `(origin, lead)` cases and presents a network with
`[batch, entity, feature]`. Evaluation stores correlations as
`[origin, lead, entity, entity]`.

## Factorization being studied

At a single origin and lead, Sklar's theorem motivates the decomposition

$$
P(Y_1\le y_1,\ldots,Y_K\le y_K\mid\mathcal I_i)
=C_{i,\tau}\!\left(
F_{i,1,\tau}(y_1),\ldots,F_{i,K,\tau}(y_K)
\right),
$$

where $\mathcal I_i$ is the information available at origin $t_i$. Simcast
approximates $C_{i,\tau}$ with a Gaussian copula parameterized by
$R_{i,\tau}$, and approximates each marginal $F_{i,k,\tau}$ by Chronos's
finite native quantile grid.

The experiment changes $R_{i,\tau}$ and nothing in the quantile grid. This
creates a controlled intervention on spatial dependence: a method can improve
aggregate forecasts only by changing joint rank co-occurrence, not by changing
an entity's values or probability-cell frequencies.

## What is and is not modeled

Modeled:

- forecast-error dependence across entities in one homogeneous physical group;
- variation of that dependence with forecast origin and lead for M2--M4;
- robustness to evaluating smaller prefixes of the fitted group;
- aggregate uncertainty induced by the resulting spatial scenarios.

Not modeled:

- temporal covariance between lead $\tau$ and lead $\tau'\ne\tau$;
- a trajectory-level copula across the full 96-lead day;
- relationships between different Liander entity types in one joint group;
- improvements to the Chronos marginal model;
- interpolation or tail extrapolation beyond native Chronos quantile values;
- causal effects or operational dispatch decisions.

Scenarios at different leads are sampled and scored as separate same-lead
experiments. Concatenating them does **not** produce scientifically valid daily
trajectories.

## Scientific assumptions

1. **PIT scores identify residual dependence.** Conditional on useful
   marginals, co-movement in $z$ is treated as forecast-error dependence. Raw
   target correlation is not used as a substitute.
2. **A Gaussian copula is an adequate first dependence family.** It captures
   rank association through a correlation matrix but not asymmetric or
   tail-specific dependence.
3. **Historical pseudo-observations transfer chronologically.** Dependence
   learned in the training period is assumed informative for validation and
   test periods.
4. **The static group is meaningful.** Every method sees all physical members
   of the selected entity category. Missing values invalidate a case; they do
   not redefine the group.
5. **Discrete marginals are acceptable for controlled comparison.** Scenario
   values have finite support at Chronos's native quantiles. This introduces
   ties and discretization, but ensures exact method-to-method marginal parity.
6. **Weather vintages and reporting delays encode the information set.** Only
   data knowable at $t_i$ are used.

## Primary scientific questions

The generated `scientific_summary.json` frames six descriptive questions:

1. Is residual PIT correlation present?
2. Does independence give calibrated aggregate intervals?
3. Does a static PIT copula improve over independence?
4. Does context conditioning improve over the static copula?
5. Does set-aware conditioning improve over entity-wise conditioning?
6. Do conclusions persist at smaller group cardinalities?

Answers are descriptive for the current test set. A rigorous publication
should add paired uncertainty intervals or block-bootstrap tests over origins,
repeat neural training over multiple seeds, and predeclare the primary score
and model comparison.

## Interpretation discipline

Lower pinball loss, CRPS, WIS, interval score, Energy Score, and Variogram Score
is better. Coverage is not an optimization score: it must be read jointly with
nominal coverage and interval width. A wider method can have coverage closer to
nominal while being less sharp. Absolute scores cannot be compared across
entity types because load scales and units differ.

The term “best” in this repository means lowest mean aggregate pinball loss
among M0--M3 on the single held-out test split. It does not mean statistically
significant, universally best, or operationally optimal.
