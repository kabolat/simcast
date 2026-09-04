# Research question, estimand, and assumptions

## Scientific objective

Simcast studies whether spatial aggregate forecasts improve when residual
cross-entity forecast dependence is modeled explicitly while the
foundation-model-derived marginal quantile grids are held fixed.

Chronos-2 remains frozen. For most Liander groups the fixed grid is the raw
Chronos-2 native quantile output. For solar parks, raw crossings are substantial,
so a deterministic isotonic monotonicity repair is first applied. The resulting
repaired solar grid is then fixed across every dependence method. No dependence
model can change any marginal quantile value.

This is a controlled predictive comparison of copulas. It is not a causal
experiment and does not claim novelty for joint probabilistic energy
forecasting as a general topic.

## Indices and static groups

For a positive integer $n$, write $[n]=\{1,\ldots,n\}$.

Forecast instances are indexed by

$$
i\in[N],
$$

with actual UTC forecast-origin time $t^{(i)}$ and chronological ordering

$$
t^{(1)}<t^{(2)}<\cdots<t^{(N)}.
$$

Static physical entity groups are indexed by $g\in[G]$. Group $g$ has the
complete ordered entity set

$$
\mathcal E_g=\{k_1,\ldots,k_{K_g}\},
\qquad K_g=|\mathcal E_g|.
$$

Each current Liander homogeneous entity type is one separate group. The entity
set never varies with forecast instance. If any member is invalid at a case,
the case is discarded for the complete group; $\mathcal E_g$ is not reduced.

The one-based forecast lead is $\tau\in[H]$. Its physical target time is

$$
t^{(i)}+\tau\Delta,
\qquad \Delta=15\text{ minutes}
$$

in the present experiments.

## Random variables and observations

Use

$$
Y_{k,\tau}^{(i)}
$$

for the future target **random variable** of entity $k$ at forecast instance
$i$ and lead $\tau$, and

$$
y_{k,\tau}^{(i)}
$$

for its observed realization. The distinction is maintained throughout the
scientific documentation.

Let $\mathcal I^{(i)}$ be the information available at origin $t^{(i)}$. The
entity-wise predictive marginal is

$$
F_{k,\tau}^{(i)}(y)
=P\!\left(Y_{k,\tau}^{(i)}\le y\mid\mathcal I^{(i)}\right).
$$

Chronos-2 approximates it with the finite native quantile grid

$$
\left\{
\left(\hat y_{k,\tau,q_j}^{(i)},q_j\right)
\right\}_{j=1}^{Q},
\qquad 0<q_1<\cdots<q_Q<1.
$$

The random spatial aggregate and its observation are respectively

$$
A_{g,\tau}^{(i)}
=\sum_{k\in\mathcal E_g}Y_{k,\tau}^{(i)},
\qquad
a_{g,\tau}^{(i)}
=\sum_{k\in\mathcal E_g}y_{k,\tau}^{(i)}.
$$

$\Lambda$ is reserved for low-rank factor loadings; $A$ is used only for the
random aggregate.

## Central controlled comparison

For one group, forecast instance, and lead, the modeled joint distribution is

$$
P\!\left(
Y_{k,\tau}^{(i)}\le y_k,\;k\in\mathcal E_g
\mid\mathcal I^{(i)}
\right)
=C_{g,\tau}^{(i)}\!\left(
\left\{F_{k,\tau}^{(i)}(y_k)\right\}_{k\in\mathcal E_g}
\right).
$$

The methods differ only in $C_{g,\tau}^{(i)}$, currently represented by a
Gaussian-copula correlation matrix

$$
R_{g,\tau}^{(i)}\in\mathbb R^{K_g\times K_g}.
$$

All methods use the same complete ordered $\mathcal E_g$, fixed FM-derived
quantile grid, test cases, and base random draws. The core hierarchy is:

- M0: independent-copula baseline;
- M1: static lead-specific Gaussian copula;
- M2: conditional low-rank Gaussian copula with a shared entity-wise network;
- M3: set-aware conditional low-rank Gaussian copula that contextualizes the
  complete group with self-attention.

M4 is an optional conditional kernel diagnostic and is not central to the
PowerTech comparison.

## PIT pseudo-observations

For an observed realization, the deterministic finite-quantile pseudo-PIT is

$$
u_{k,\tau}^{(i)}
=f^{\mathrm{PIT}}\!\left(
y_{k,\tau}^{(i)},
\left\{
\left(\hat y_{k,\tau,q_j}^{(i)},q_j\right)
\right\}_{j=1}^{Q}
\right).
$$

The implementation assigns the observation to one of $Q+1$ cells. It does not
interpolate or reconstruct a continuous predictive CDF. Gaussianized scores are

$$
z_{k,\tau}^{(i)}=\Phi^{-1}\!\left(u_{k,\tau}^{(i)}\right),
$$

and the complete group vector is

$$
\mathbf z_{g,\tau}^{(i)}
=\left[z_{k,\tau}^{(i)}\right]_{k\in\mathcal E_g}
\in\mathbb R^{K_g}.
$$

This vector exists for training/evaluation only when every group member is
valid at $(i,\tau)$.

## Primary estimand

For a proper aggregate score $S$, the descriptive method contrast over the
held-out full-group cases is

$$
\frac{1}{|\mathcal D_{g,\mathrm{test}}|}
\sum_{(i,\tau)\in\mathcal D_{g,\mathrm{test}}}
\left[
S\!\left(C_{g,\tau}^{(i)},a_{g,\tau}^{(i)}\right)
-S\!\left(C_{g,\tau}^{(i),\mathrm{M0}},a_{g,\tau}^{(i)}\right)
\right],
$$

where notation suppresses the shared fixed marginals. Mean aggregate pinball
loss is the headline descriptive score; CRPS, interval scores, WIS, coverage,
Energy Score, and Variogram Score provide complementary evidence.

## What is and is not modeled

Modeled:

- same-lead forecast-error dependence across all entities in one static group;
- origin- and lead-dependent correlations for M2/M3;
- aggregate uncertainty induced by full-group spatial scenarios.

Not modeled:

- random, prefix, or reduced-cardinality entity sets;
- entity dropout or subset augmentation;
- covariance between different forecast leads;
- trajectory-level temporal copulas;
- mixed-type Liander groups;
- Chronos fine-tuning or new foundation models;
- continuous CDF interpolation or tail extrapolation;
- causal effects or operational decisions.

M2/M3 remain mathematically capable of accepting other cardinalities, but this
is an architectural property only and is not evaluated in the PowerTech study.

## Scientific assumptions

1. Gaussianized finite-cell PIT scores provide a useful approximation to
   residual rank dependence.
2. A Gaussian copula is an adequate first family despite excluding asymmetric
   and tail-specific dependence.
3. Training-period dependence transfers to the chronological test period.
4. Each homogeneous metadata-defined entity group is scientifically meaningful.
5. Complete-case filtering does not invalidate the intended full-group
   interpretation; its possible selection effect remains a limitation.
6. The discrete fixed marginal approximation is acceptable for a controlled
   method comparison.
7. Reporting delays and weather vintages correctly represent
   $\mathcal I^{(i)}$.

## PowerTech scientific questions

1. How much residual dependence remains after the fixed FM-derived marginal
   forecasts?
2. How well does the independent-copula baseline represent full-group aggregate
   uncertainty?
3. Does a static lead-specific PIT copula improve aggregate forecasts?
4. Does context-conditioned low-rank dependence improve over static dependence?
5. Does full-group set-aware contextualization improve over entity-wise
   conditioning?
6. How heterogeneous is the benefit of dependence modeling across physical
   entity groups?

These are predictive questions. “Best” means lowest held-out mean aggregate
pinball among M0--M3 under the declared full-group protocol, not causal,
universally superior, or statistically significant.

## Interpretation discipline

Lower pinball, CRPS, WIS, interval score, Energy Score, and Variogram Score is
better. Coverage must be considered with nominal coverage and interval width;
coverage alone is not a proper score. Absolute values are not comparable across
entity types because scales and units differ. Confirmatory claims require
paired temporal uncertainty intervals and multiple neural-training seeds.
