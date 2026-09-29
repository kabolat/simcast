# Full-group dependence fitting, scenario generation, and scoring

**Operational realization.** Sampling and evaluation arguments are defined in
the [configuration reference](../technical/configuration_reference.md), while
saved fitted methods and evaluation records are described in the
[artifact reference](../technical/artifact_reference.md).

## Dependence-fitting cases

For group $g$, conditional training converts each valid $(i,\tau)$ into

$$
\left(X_{g,\tau}^{(i)},\mathbf z_{g,\tau}^{(i)}\right),
$$

where

$$
X_{g,\tau}^{(i)}\in\mathbb R^{K_g\times F},
\qquad
\mathbf z_{g,\tau}^{(i)}\in\mathbb R^{K_g}.
$$

A case exists only if the binary validity indicator
$V_{g,\tau}^{(i)}=1$ and all features are finite. $X$ is the feature matrix;
it must not be confused with $V$. The code axis
is `[origin, entity, lead, feature]`; dataset construction flattens only the
valid `(origin, lead)` cases. Origin and lead indices are metadata rather than
direct network inputs. Lead information enters through the frozen forecast,
patch representation, and within-patch position.

The entity dimension is checked again immediately before every M2--M4 model
call.
Every training and validation batch therefore contains exactly the complete
ordered $\mathcal E_g$. Missing one entity invalidates the case instead of
shrinking it.

M1 uses complete training vectors directly and does not use validation. M2,
M3, and M4 fit on the chronological training partition and use chronological
validation pseudo-likelihood for checkpoint selection. `DependenceCollator`
only stacks complete vectors.

## Gaussian-copula pseudo-likelihood

For one complete score vector $\mathbf z$ and predicted correlation $R$, the
Gaussian-copula log density relative to independent standard normals is

$$
\log c_R(\mathbf u)
=-\frac12\log|R|
-\frac12\mathbf z^{\mathsf T}(R^{-1}-I)\mathbf z.
$$

The minimized negative pseudo-log-likelihood is

$$
\mathcal L(\mathbf z,R)
=\frac12\left[
\log|R|+\mathbf z^{\mathsf T}(R^{-1}-I)\mathbf z
\right].
$$

It is a pseudo-likelihood because the marginal transformation uses
deterministic finite-quantile PIT values and because $R$ may be predicted
from the same forecast information. The loss may be negative; M0 gives zero
before stabilization for every vector.

No explicit inverse is formed. For $R=LL^{\mathsf T}$, Cholesky solves compute
$R^{-1}\mathbf z$ and
$\log|R|=2\sum_k\log L_{kk}$. Optimization averages batch losses. Validation
sums then divides by the number of full-group cases.

## Optimization and checkpoints

The baseline defaults are AdamW, batch size 64, learning rate $10^{-3}$, weight
decay $10^{-4}$, global gradient-norm cap one, at most 100 epochs, and early
stopping after 12 epochs without a strictly lower validation loss.

The trainer writes `best.pt` on each validation improvement, `final.pt` at the
actual last epoch, CSV/JSON histories, `training_summary.json`, and a training
curve. Evaluation loads `best.pt`. Conditional checkpoints include constructor
arguments, the feature-builder configuration and training-only
standardization, complete ordered entity IDs, method, epoch, and state dict.

Python, NumPy, PyTorch, and CUDA generators are seeded. When
`runtime.deterministic: true`, PyTorch deterministic algorithms are enforced
strictly and CUDA flash/memory-efficient attention kernels are disabled in
favour of the deterministic mathematical attention backend. The CUDA BLAS
workspace is fixed before training begins. A same-seed M3 check on the study
host reproduced every learned tensor exactly. Reproducibility is still scoped
to the recorded software and hardware environment; it is not asserted across
different PyTorch, CUDA, or GPU versions.

## Full-group scenario generation

For every valid $(g,i,\tau)$ and method:

1. obtain $R_{g,\tau}^{(i)}\in\mathbb R^{K_g\times K_g}$ and assert its exact
   full-group shape;
2. stabilize it and compute $L_{g,\tau}^{(i)}$;
3. draw $M=4096$ base normals $\eta^{(m)}\in\mathbb R^{K_g}$;
4. form copula uniforms
   $u_{k,\tau}^{(i,m)}=\Phi([L_{g,\tau}^{(i)}\eta^{(m)}]_k)$;
5. project each uniform through the configured fixed marginal law, producing
   $\widetilde Y_{k,\tau}^{(i,m)}$;
6. apply the evaluation's cross-entity statistic to the complete group:

$$
\widetilde T_{g,\tau}^{(i,m)}
=T_g\!\left(\widetilde{\mathbf Y}_{g,\tau}^{(i,m)}\right).
$$

`evaluation.cross_entity_statistic: sum` uses
$T_g(\mathbf y)=\sum_k y_k$; `absolute_sum` uses
$T_g(\mathbf y)=\sum_k |y_k|$; `max` uses $\max_k y_k$; and
`absolute_max` uses $\max_k |y_k|$. The same function is applied to the
observed entity vector before aggregate quantile, interval, CRPS, and pinball
scoring. Energy and Variogram Scores continue to score the full entity vector,
independent of this choice.

Cases are processed in batches of `evaluation.scenario_batch_size`. Base normals are keyed by the separately
recorded evaluation seed and flattened `(origin, lead)` index. They therefore
remain identical for the same case across methods, method order, and batch
layout. `sampling.common_random_numbers: false` is an explicit diagnostic mode
that adds a stable method-specific seed offset.

The test Gaussian-copula pseudo-NLL is also evaluated case by case under each
predicted correlation. It is a direct dependence diagnostic, not an aggregate
forecast score, and remains a pseudo-likelihood because PITs are inferred from
a finite native quantile grid rather than an analytic predictive CDF.

Confirmatory uncertainty is computed after averaging valid leads within each
origin. Neural seeds are averaged within method/origin before the primary
non-circular moving-block bootstrap. Daily block length seven is primary;
lengths three and fourteen are sensitivity analyses.

## Finite-quantile marginal projection

Scenario projection is the inverse-direction counterpart of historical PIT
construction. `pit.mode` fixes both operations. Under the default
`discretized` mode, nearest-level boundaries are

$$
b_0=0,
\quad b_j=\frac{q_j+q_{j+1}}2\;(j=1,\ldots,Q-1),
\quad b_Q=1.
$$

Each uniform is assigned to exactly one native quantile value
$\hat y_{k,\tau,q_j}^{(i)}$. There is no value interpolation or tail
extrapolation.

Under `linear_interpolation`, the inverse marginal is constant below $q_1$,
linear between adjacent pairs $(q_j,\hat y_{q_j})$, and constant above $q_Q$.
Consequently, it retains endpoint atoms of masses $q_1$ and $1-q_Q$ and does
not invent unsupported tails. Equal adjacent values, including ties introduced
by isotonic repair, also form atoms.

Because Gaussian-copula components are marginally uniform, M0--M4 have the
same configured entity-wise marginal law. For any given base, the same raw or
isotonic-repaired grid and the same projection mode are held fixed across all
methods. See [Chapter 3](03_chronos_and_pit.md#7-probability-space-projection-for-scenarios)
for the formal maps and a numerical comparison.

Cross-entity-statistic quantiles are nearest empirical order statistics of
the $M$ full-group scenario statistics. For the simple sum, they are not sums
of equally labeled marginal quantiles:

$$
Q_\alpha\!\left(\sum_kY_k\right)
\ne\sum_kQ_\alpha(Y_k)
$$

in general.

## Aggregate scores

All metrics pool the same valid full-group origin-lead cases unless reported by
lead. Lower loss/score is better. The aggregate scores evaluate the scalar
cross-entity statistic $a=T(\mathbf y)$ of the observation against the $M$
scenario statistics $x_m=T(\mathbf x_m)$; see the
[references](#references) at the end of this chapter.

### Pinball loss

For aggregate quantile forecast $q_\alpha$ and observation $a$, the quantile
(pinball) loss of Koenker and Bassett (1978) is

$$
\rho_\alpha(a-q_\alpha)
=\max\{\alpha(a-q_\alpha),(\alpha-1)(a-q_\alpha)\}.
$$

`mean_pinball` averages over the configured `evaluation.quantile_levels`
(default $0.05,0.10,0.25,0.50,0.75,0.90,0.95$) and all cases; the per-level
values are kept as `pinball_q<level>`. The supplied main report presents
`mean_pinball` as its headline score. Aggregate quantiles are nearest
empirical order statistics of the $M$ scenario statistics (Hyndman and Fan,
1996).

The loss is consistent for the $\alpha$-quantile (Gneiting, 2011): its expected
value is minimized by reporting the true $\alpha$-quantile. It is asymmetric.
An observation above $q_\alpha$ costs $\alpha$ per unit of excess, one below
costs $1-\alpha$ per unit. At $\alpha=0.95$, under-forecasting is penalized
nineteen times more heavily than over-forecasting, so the optimal $q_{0.95}$
is exceeded in only 5% of cases. This makes the per-level values diagnostic:

- a high `pinball_q0.05` or `pinball_q0.95` locates the problem in the lower
  or upper tail of the aggregate distribution;
- a high `pinball_q0.5` is a location error (at $\alpha=0.5$ the loss is half
  the absolute error of the median);
- `summary_quantile_calibration.png` shows the empirical frequency of
  $a\le q_\alpha$ against $\alpha$. A calibrated forecast lies on the diagonal,
  and the pinball loss rewards sharpness only among calibrated quantiles.

### Ensemble CRPS

The continuous ranked probability score of Matheson and Winkler (1976) is

$$
\operatorname{CRPS}(F,a)=\int_{-\infty}^{\infty}\bigl(F(x)-\mathbf 1\{a\le x\}\bigr)^2\,dx .
$$

For $F$ with a finite mean it has the kernel (energy) representation of
Gneiting and Raftery (2007),

$$
\operatorname{CRPS}(F,a)=\mathbb E_F|X-a|-\tfrac12\,\mathbb E_F|X-X'|,
\qquad X,X'\overset{\text{iid}}{\sim}F .
$$

Simcast evaluates this for the empirical distribution $\widehat F_M$ of the
ensemble $x_1,\ldots,x_M$, which replaces both expectations by averages:

$$
\operatorname{CRPS}(\widehat F_M,a)
=\frac1M\sum_m|x_m-a|
-\frac1{2M^2}\sum_{m,n}|x_m-x_n|.
$$

This is the exact integral CRPS of the ensemble's step-function CDF, not an
approximation of it. Three equivalent views are useful:

- **Order statistics.** With sorted members $x_{(1)}\le\cdots\le x_{(M)}$,
  $\sum_{m,n}|x_m-x_n|=2\sum_{i=1}^M(2i-M-1)\,x_{(i)}$. The implementation uses
  this identity, so the pair term costs $O(M\log M)$ instead of $O(M^2)$.
- **Quantile decomposition.** $\operatorname{CRPS}(\widehat F_M,a)
  =\frac2M\sum_{i=1}^M\rho_{\tau_i}(a-x_{(i)})$ with $\tau_i=(2i-1)/(2M)$, that
  is, twice the mean pinball loss over $M$ evenly spaced levels (Laio and
  Tamea, 2007; Bröcker, 2012). This is the bridge to WIS below.
- **Fair version.** Replacing $1/(2M^2)$ by $1/(2M(M-1))$ gives the fair CRPS
  of Ferro (2014), an unbiased estimator of $\operatorname{CRPS}(F,a)$ for the
  distribution $F$ that generated the members. The version used here exceeds
  it by about $\mathbb E|X-X'|/(2M)$, which is negligible at $M=4096$. Zamo
  and Naveau (2018) compare these estimators.

### Central intervals

For central coverage $c=1-\alpha$ with bounds $(l,u)$ set to the empirical
$\alpha/2$ and $1-\alpha/2$ scenario quantiles, the interval score of Winkler
(1972) (see also Gneiting and Raftery, 2007) is

$$
\operatorname{IS}_\alpha(l,u;a)
=(u-l)+\frac{2}{\alpha}(l-a)\mathbf1(a<l)
+\frac{2}{\alpha}(a-u)\mathbf1(a>u).
$$

Coverage is the fraction with $l\le a\le u$; width is $u-l$. The default
evaluation reports central 0.50, 0.80, and 0.90 intervals from
`evaluation.interval_levels`, independently of `quantile_levels`. Coverage
must be read with width and a proper score.

### Weighted interval score

For $K$ central intervals with miscoverages $\alpha_1,\ldots,\alpha_K$ and the
empirical aggregate median $m$, Simcast uses the weighted interval score of
Bracher et al. (2021) with $w_0=\tfrac12$ and $w_k=\alpha_k/2$:

$$
\operatorname{WIS}
=\frac{1}{K+\tfrac12}\left(
\tfrac12|a-m|+
\sum_{k=1}^K\tfrac{\alpha_k}{2}\operatorname{IS}_{\alpha_k}
\right).
$$

Because $\tfrac{\alpha}{2}\operatorname{IS}_\alpha=\rho_{\alpha/2}(a-l)+\rho_{1-\alpha/2}(a-u)$
and $\tfrac12|a-m|=\rho_{1/2}(a-m)$,

$$
\operatorname{WIS}=\frac{2}{2K+1}\sum_{j=1}^{2K+1}\rho_{\tau_j}(a-q_{\tau_j}),
\qquad \tau\in\{\alpha_k/2,\ \tfrac12,\ 1-\alpha_k/2\},
$$

twice the mean pinball loss over the $2K+1$ levels implied by the intervals.

### Calibration summaries

The report includes two descriptive calibration plots, with one panel per
base and one curve per evaluated method. They use the valid rows of
`per_origin_lead_metrics.parquet`, not origin-averaged values. Thus each
complete test origin-lead case contributes a hit or miss; each fitted seed
contributes its own cases. The curves pool forecast leads and seeds within a
base/method. They are not paired comparisons and have no confidence bands.

For interval level $c$ with scenario-derived bounds $[l_{c,v},u_{c,v}]$,
`summary_coverage.png` plots nominal coverage $c$ against the empirical hit
rate

$$
\widehat C(c)=\frac1N\sum_{v\in\mathcal V}
\mathbf1\{l_{c,v}\le a_v\le u_{c,v}\},
$$

where $\mathcal V$ is the set of valid test origin-lead-seed records and $N$
is its size. A calibrated interval lies on the diagonal. Below the diagonal
means undercoverage; above means overcoverage. For example, a nominal 90%
interval that covers only 75% of cases is too narrow, shifted, or both.

For quantile probability $\alpha$ with forecast $q_{\alpha,v}$,
`summary_quantile_calibration.png` plots nominal $\alpha$ against

$$
\widehat Q(\alpha)=\frac1N\sum_{v\in\mathcal V}
\mathbf1\{a_v\le q_{\alpha,v}\}.
$$

This is the empirical frequency with which observations fall at or below the
forecast quantile. Calibration means $\widehat Q(\alpha)\approx\alpha$. A curve
above the diagonal indicates the forecasts are too high at those levels; a
curve below indicates they are too low. Ties are counted as hits, matching the
quantile definition used by the evaluator. Finite test samples and nearest
empirical scenario quantiles make departures from the diagonal inevitable.

These plots diagnose **calibration only**, not sharpness or overall forecast
quality. An excessively wide interval can attain or exceed nominal coverage;
similarly, the quantile-calibration curve says nothing about how far the
forecast quantiles are from the observations. Read the interval curve with
`interval_width_*`, and read the quantile curve with `pinball_q<level>`.
CRPS, pinball loss, and WIS are proper scores that reward calibration while
also penalizing unnecessary spread; WIS additionally decomposes interval
width and misses. The curves summarize reliability, while the proper scores
support method ranking. Different bases have different cross-entity scales,
so these calibration plots are faceted by base rather than pooled across
groups.

### How the aggregate scores relate

`mean_pinball`, CRPS, and WIS are not three independent criteria. All three
are averages of the same quantile loss $\rho_\tau$ and differ only in which
levels $\tau$ they average over and in a factor of two:

| Score | Levels averaged | Formula | Scale |
|---|---|---|---|
| `mean_pinball` | the $J$ declared `quantile_levels` | $\frac1J\sum_j\rho_{\tau_j}$ | half |
| WIS | the $2K+1$ levels implied by `interval_levels` | $\frac{2}{2K+1}\sum_j\rho_{\tau_j}$ | full |
| CRPS | all levels in $(0,1)$ with equal weight | $2\int_0^1\rho_\tau\,d\tau$ | full |

"Full scale" means that each score reduces to the absolute error for a point
forecast, so it is measured in the units of the observation and reads as an
uncertainty-aware absolute error. `mean_pinball` is on half that scale:
$2\times$`mean_pinball` is the number to set beside CRPS and WIS. With the
default configuration the three are tied even more closely: `quantile_levels`
equal the levels implied by the default `interval_levels`, so
$\operatorname{WIS}=2\times$`mean_pinball` exactly, case by case.

The substantive difference is the weighting over $\tau$. Bracher et al.
(2021) show that WIS approaches CRPS for many, roughly evenly spaced levels;
the same holds for $2\times$`mean_pinball`. Short or uneven grids depart from
it:

1. **The default grid emphasizes the tails.** The default levels
   $\{0.05,0.10,0.25,0.50,0.75,0.90,0.95\}$ place four of seven points in the
   outer 10% tails and only three between 0.25 and 0.75. `mean_pinball` and WIS
   therefore weight tail errors more than CRPS does. A set of central intervals
   such as 0.10--0.80 does the opposite: it weights the centre and ignores the
   tails beyond the 0.10 and 0.90 quantiles.
2. **Grid-based values depend on the declared levels.** Changing
   `quantile_levels` or `interval_levels` changes what `mean_pinball` or WIS
   measures, so their values from evaluations with different levels are not
   comparable. CRPS does not depend on any declared level.
3. **CRPS uses the whole ensemble.** Because the complete scenario ensemble is
   available, CRPS is the exact, level-free score of the aggregate
   distribution. `mean_pinball` and WIS summarize it at chosen quantiles.

### Choosing and reading the aggregate scores

Choose the score that matches the question, and fix the primary one before
looking at test results (see [Chapter 7](07_experiments_and_results.md)).

| Question | Score |
|---|---|
| How accurate is the whole aggregate distribution? | CRPS |
| How accurate is a quantile that a decision uses, e.g. a 95% capacity or reserve level? | `pinball_q<level>` at that level |
| How accurate is the aggregate on a declared quantile grid? | `mean_pinball` |
| Are the intervals calibrated and how sharp are they? | `coverage_*` with `interval_width_*`, summarized by WIS |
| How does the result compare with interval- or quantile-format benchmarks? | WIS, or `mean_pinball` on the benchmark's levels |

WIS adds most when it is read through its components. Each $\operatorname{IS}_\alpha$
splits into the width $u-l$ (sharpness) and the penalties for observations
below $l$ or above $u$ (calibration), and `interval_width_*`, `coverage_*`, and
`interval_score_*` are stored per level for this purpose. One half of this WIS,
as an average pinball loss, was the GEFCom2014 score (Hong et al., 2016).

When reading results:

1. **Agreement is the robust case.** If CRPS, `mean_pinball`, and WIS favour
   the same method with paired intervals that exclude zero, the improvement
   holds regardless of how the quantile levels are weighted.
2. **Disagreement locates the difference.** If `mean_pinball` improves but
   CRPS does not, the gain is concentrated at the declared levels, usually the
   tails under the default grid. Inspect `pinball_q<level>` to find which.
3. **Coverage explains the pinball tails.** Coverage below nominal with narrow
   intervals means under-dispersion; coverage above nominal with wide
   intervals means over-dispersion. Either raises the corresponding tail
   pinball losses.
4. **In Simcast, dependence acts through dispersion.** Marginals are fixed, so
   aggregate scores can change only because the copula changes the joint
   distribution of the entities. For the sum,
   $\operatorname{Var}(\sum_kY_k)=\sum_k\operatorname{Var}(Y_k)+2\sum_{k<l}\operatorname{Cov}(Y_k,Y_l)$:
   positive dependence widens the aggregate distribution. When forecast errors
   are positively correlated, M0 produces aggregate intervals that are too
   narrow. This shows up as coverage below nominal and inflated tail pinball
   losses, while the median, and hence `pinball_q0.5`, is barely affected. For
   `max` and `absolute_max`, positive dependence instead lowers the upper tail
   of the statistic.
5. **Magnitudes are group-specific.** All three are in the units of $T$, so
   compare them within a base, or use relative paired effects across bases.

## Joint full-group scores

Joint scores use the first `min(M,512)` members of the already generated
full-group ensemble. This is an ensemble-size selection, not entity
subsampling. They evaluate the entity vector $\mathbf y\in\mathbb R^{K_g}$,
not the cross-entity statistic.

### Energy Score

The Energy Score generalizes the CRPS kernel form to vectors (Gneiting and
Raftery, 2007; Gneiting et al., 2008). For selected ensemble
$x_1,\ldots,x_{M_J}\in\mathbb R^{K_g}$ and observed vector $\mathbf y$, the
baseline uses the empirical all-pairs estimator

$$
\widehat{\operatorname{ES}}
=\frac1{M_J}\sum_{m=1}^{M_J}\|x_m-\mathbf y\|_2
-\frac1{2M_J^2}\sum_{m=1}^{M_J}\sum_{n=1}^{M_J}
\|x_m-x_n\|_2.
$$

The pair sum is exact for the selected 512-member ensemble. `torch.cdist` is
chunked over 128 first-sample rows solely to limit memory; no cyclic pairing or
pair subsampling remains.

Interpretation:

1. **Accuracy minus spread.** The first term is the mean Euclidean distance from
   the scenarios to the observed vector; the second rewards ensemble spread.
   As with the CRPS, a sharper ensemble scores better only if it still
   contains the observation. For $K_g=1$ the Energy Score equals the CRPS.
2. **Relation to the aggregate scores.** Since
   $\|\mathbf x\|_2=c_{K}\int_{S^{K-1}}|\theta^{\mathsf T}\mathbf x|\,d\sigma(\theta)$
   for a constant $c_K$ and the uniform distribution $\sigma$ on unit
   directions, the Energy Score is proportional to the CRPS of the projection
   $\theta^{\mathsf T}\mathbf Y$ averaged over all directions $\theta$. The
   simple-sum aggregate CRPS scores one direction,
   $\theta\propto(1,\ldots,1)$. The Energy Score therefore evaluates the joint
   law in every direction, and a gain in the sum direction is diluted among
   all the others. A method can improve the aggregate CRPS without improving
   the Energy Score, and the reverse.
3. **Weak sensitivity to dependence.** The Energy Score is strictly proper, but
   location and scale errors dominate it, and its ability to detect
   misspecified correlations is limited (Scheuerer and Hamill, 2015; Pinson
   and Tastu, 2013). Because Simcast holds marginals fixed, differences
   between methods are entirely due to dependence but are typically small
   relative to the score. Judge them by their paired intervals, not by their
   size relative to the score; the Variogram Score is reported alongside for
   this reason.
4. **Scale.** The score is in the units of $\mathbf y$, and the Euclidean norm
   lets large or volatile entities dominate. It grows with $K_g$ and is not
   comparable across groups.
5. **Estimator.** Like the ensemble CRPS, the all-pairs estimator exceeds its
   fair counterpart by about $\mathbb E\|X-X'\|/(2M_J)$; with $M_J=512$ this
   is small relative to the score.

### Variogram Score

Simcast uses the unit-weight Variogram Score of order $p$ (Scheuerer and
Hamill, 2015):

$$
\operatorname{VS}_p
=\sum_{i=1}^{K_g}\sum_{j=1}^{K_g}\left(
|y_i-y_j|^p
-\mathbb E|X_i-X_j|^p
\right)^2,
$$

where the expectation is estimated by the mean over the $M_J$ selected
members and $p=$ `evaluation.variogram_power` (default $0.5$). The diagonal
terms are zero and the $(i,j)$ and $(j,i)$ terms are equal, so the
implementation evaluates each unordered pair once and doubles the sum; the
result is exactly the double sum above.

Interpretation requires care:

1. **Proper, not strictly proper.** The score depends on the forecast only
   through the expected pairwise variogram $\mathbb E|X_i-X_j|^p$. A shift
   common to all components leaves every difference unchanged, and two
   forecasts with the same pairwise variograms score identically. VS
   therefore does not assess marginal accuracy. In Simcast marginals are
   fixed across methods, so VS differences between methods isolate
   dependence, but a good VS says nothing about the fixed marginals.
2. **Discrimination.** Scheuerer and Hamill (2015) find VS distinctly more
   discriminative than the Energy Score with respect to correlation
   structure. This is why both are reported.
3. **Scale and group size.** VS has units of $|y|^{2p}$ and sums
   $K_g(K_g-1)$ non-zero terms. Its magnitude grows with entity scale and
   with $K_g$, so it is not comparable across groups. Because every pair has
   equal weight, pairs of large or volatile entities dominate.
4. **Order $p$.** Smaller $p$ gives less weight to large pairwise differences
   and is more robust to outliers. scoringRules uses $p=0.5$ by default and
   lists $p=0.5$ and $p=1$ as standard choices (Jordan et al., 2019).
5. **Monte Carlo estimate.** Squaring the difference between the observed term
   and an ensemble mean biases the estimate upward by
   $\operatorname{Var}(|X_i-X_j|^p)/M_J$ per term. With $M_J=512$ this is
   small, and it is the same selected ensemble size for every method.

## Full-group output scope

The evaluator produces only complete-group metrics, lead tables, correlation
matrices, and figures. The dependence networks share weights across entities,
so one architecture serves groups of different $K_g$; tests with several
group sizes check this property.

## Worked scoring example and implementation guidance

Suppose three aggregate scenarios are $(48,55,63)$ and the observation is
$a=58$. The first CRPS term is
$(|48-58|+|55-58|+|63-58|)/3=6$. The pairwise distances are $7$, $15$, and $8$,
so the nine ordered pairs sum to $60$ and the second term is
$60/(2\cdot 9)=3.33$. The CRPS is $6-3.33=2.67$: the ensemble's spread earns
back a little over half of its mean absolute error. The median scenario is
$55$, so `pinball_q0.5` is $0.5\cdot|58-55|=1.5$. Energy Score applies the
same accuracy-minus-spread principle to vectors of entity values and
therefore evaluates the joint spatial law.

For M2--M4, `method.optimization.*` controls optimization. In an evaluation
document, `sampling.num_samples`, `sampling.evaluation_seed`, and
`sampling.common_random_numbers` control joint draws, and `evaluation.*`
declares the cross-entity statistic, score levels, variogram order, and
joint-ensemble size.
Training is in `training/trainer.py`, sampling in
`sampling/gaussian_copula.py`, finite projection in
`sampling/quantile_projection.py`, and scoring in `evaluation/metrics.py` and
`evaluation/aggregate.py`, all below `src/simcast/`.

## References

- Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021). Evaluating
  epidemic forecasts in an interval format. *PLoS Computational Biology*,
  17(2), e1008618. https://doi.org/10.1371/journal.pcbi.1008618
- Bröcker, J. (2012). Evaluating raw ensembles with the continuous ranked
  probability score. *Quarterly Journal of the Royal Meteorological Society*,
  138(667), 1611--1617. https://doi.org/10.1002/qj.1891
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *Quarterly
  Journal of the Royal Meteorological Society*, 140(683), 1917--1923.
  https://doi.org/10.1002/qj.2270
- Gneiting, T. (2011). Quantiles as optimal point forecasts. *International
  Journal of Forecasting*, 27(2), 197--207.
  https://doi.org/10.1016/j.ijforecast.2009.12.015
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *Journal of the American Statistical
  Association*, 102(477), 359--378. https://doi.org/10.1198/016214506000001437
- Gneiting, T., Stanberry, L. I., Grimit, E. P., Held, L., & Johnson, N. A.
  (2008). Assessing probabilistic forecasts of multivariate quantities, with
  an application to ensemble predictions of surface winds. *TEST*, 17,
  211--235. https://doi.org/10.1007/s11749-008-0114-x
- Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., & Hyndman, R. J.
  (2016). Probabilistic energy forecasting: Global Energy Forecasting
  Competition 2014 and beyond. *International Journal of Forecasting*, 32(3),
  896--913. https://doi.org/10.1016/j.ijforecast.2016.02.001
- Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical packages.
  *The American Statistician*, 50(4), 361--365.
  https://doi.org/10.1080/00031305.1996.10473566
- Jordan, A., Krüger, F., & Lerch, S. (2019). Evaluating probabilistic
  forecasts with scoringRules. *Journal of Statistical Software*, 90(12),
  1--37. https://doi.org/10.18637/jss.v090.i12
- Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*,
  46(1), 33--50. https://doi.org/10.2307/1913643
- Laio, F., & Tamea, S. (2007). Verification tools for probabilistic forecasts
  of continuous hydrological variables. *Hydrology and Earth System Sciences*,
  11(4), 1267--1277. https://doi.org/10.5194/hess-11-1267-2007
- Matheson, J. E., & Winkler, R. L. (1976). Scoring rules for continuous
  probability distributions. *Management Science*, 22(10), 1087--1096.
  https://doi.org/10.1287/mnsc.22.10.1087
- Pinson, P., & Tastu, J. (2013). *Discrimination ability of the Energy
  score*. Technical report, Technical University of Denmark.
- Scheuerer, M., & Hamill, T. M. (2015). Variogram-based proper scoring rules
  for probabilistic forecasts of multivariate quantities. *Monthly Weather
  Review*, 143(4), 1321--1334. https://doi.org/10.1175/MWR-D-14-00269.1
- Winkler, R. L. (1972). A decision-theoretic approach to interval
  estimation. *Journal of the American Statistical Association*, 67(337),
  187--191. https://doi.org/10.1080/01621459.1972.10481224
- Zamo, M., & Naveau, P. (2018). Estimation of the continuous ranked
  probability score with limited information and applications to ensemble
  weather forecasts. *Mathematical Geosciences*, 50(2), 209--234.
  https://doi.org/10.1007/s11004-017-9709-7
