# Training, scenario generation, and scoring

## Training examples

Conditional training begins after the feature builder has fitted its scalar
statistics on training origins. A `DependenceDataset` turns each complete
$(i,\tau)$ case into

$$
(V_{i,:,\tau},z_{i,:,\tau}),
$$

where $V\in\mathbb R^{K\times F}$ and $z\in\mathbb R^K$. A case is retained
only when all entity scores and all feature components are finite. Origin index
and one-based lead are carried as metadata but are not explicit network inputs;
lead information enters only through Chronos forecasts, embeddings, and
within-patch position.

M1 uses complete training vectors directly and does not use the validation
partition. M2--M4 train on the training partition and use chronological
validation pseudo-likelihood for checkpoint selection and early stopping.

## Gaussian-copula pseudo-likelihood

For one pseudo-score vector $z$ and predicted correlation $R$, the Gaussian
copula log density relative to independent standard normals is

$$
\log c_R(u)
=-\frac12\log|R|
-\frac12 z^{\mathsf T}(R^{-1}-I)z.
$$

The minimized negative pseudo-log-likelihood is therefore

$$
\mathcal L(z,R)
=\frac12\left[\log|R|+z^{\mathsf T}(R^{-1}-I)z\right].
$$

It is “pseudo” because the marginal CDFs are represented by deterministic
discrete PIT locations and because $R$ can be conditionally predicted. The
normalizing constants cancel. The loss can be negative; zero is the M0 value
for every vector, and a fitted copula density may assign more density than
independence.

No matrix inverse is formed. With $R=LL^{\mathsf T}$, the implementation uses
Cholesky solves for $R^{-1}z$ and computes
$\log|R|=2\sum_k\log L_{kk}$. Batch losses are averaged for optimization and
summed then divided by case count for validation reporting.

## Random entity-subset training

The default conditional collator trains on multiple cardinalities. For each
batch with full size $K$:

- with probability 0.25, use all $K$ entities;
- otherwise draw subset size $K'$ uniformly as an integer from
  $\{4,\ldots,K-1\}$ and sample a fresh random entity permutation/subset for each
  case in the batch;
- all cases in that batch share $K'$ so they can be stacked, but need not share
  the same entity identities.

When $K\le4$, the full group is always used. Validation always uses the full
group. Subset training is compatible with M2/M3/M4's shared parameterization;
it does not change the target physical group or impute missing entities. M4's
bounded high-level smoke run explicitly disables this mechanism.

## Optimizer and checkpoint selection

Defaults are AdamW, batch size 64, learning rate $10^{-3}$, weight decay
$10^{-4}$, global gradient-norm clipping at 1, at most 100 epochs, and early
stopping after 12 consecutive epochs without a strictly lower validation loss.

The trainer saves:

- `best.pt` whenever validation pseudo-NLL improves;
- `final.pt` at the actual last epoch;
- CSV and JSON loss histories;
- `training_summary.json` with best epoch/loss and epochs completed;
- `training_curve.png`.

After fitting, the in-memory model is restored to the best state. A checkpoint
also contains method name, constructor arguments, fitted feature-standardizer
state, entity IDs, schema version, and epoch.

Python, NumPy, PyTorch, and all CUDA generators are seeded. Deterministic
PyTorch algorithms are requested with warnings rather than hard failure. This
improves replay but does not promise bitwise equality across GPU hardware,
CUDA/PyTorch versions, or nondeterministic kernels.

## Gaussian-copula scenario algorithm

For each valid test $(i,\tau)$ and method:

1. stabilize $R_{i,\tau}$ and compute its Cholesky factor $L$;
2. draw $M=4096$ base normal vectors $\eta^{(m)}$;
3. compute $u^{(m)}=\Phi(L\eta^{(m)})$;
4. map each component to its nearest native probability cell;
5. obtain entity scenario values
   $\widetilde Y_{i,k,\tau}^{(m)}$;
6. sum entities:
   $\widetilde A_{i,\tau}^{(m)}=\sum_k\widetilde Y_{i,k,\tau}^{(m)}$.

Cases are processed in batches of 16. Each method recreates a generator with
seed `config.seed + batch_offset`, so methods receive the same base normal
array for the same case batch. This common-random-number design reduces Monte
Carlo noise in pairwise score differences.

Implementation note: the evaluator currently always uses common random
numbers. `sampling.common_random_numbers` records the intended protocol but is
not consulted as a switch. Since its only accepted experimental value is not
type-restricted, setting it false would currently have no effect and should not
be described as an independent-draw experiment.

## Aggregate quantiles

Evaluation levels are $0.05,0.10,0.25,0.50,0.75,0.90,0.95$. Aggregate
quantiles are nearest empirical order statistics of the $M$ aggregate samples,
using PyTorch's `interpolation="nearest"`. They are not obtained by summing
entity quantiles. Indeed,

$$
Q_\alpha\left(\sum_kY_k\right)
\ne\sum_k Q_\alpha(Y_k)
$$

in general. `sum_marginal_quantiles` exists only as an explicitly labelled
comonotonic-style diagnostic helper and is not the evaluated method forecast.

## Aggregate scoring rules

All reported aggregate metrics pool valid origin-lead cases unless marked
“by lead.” Lower loss/score is better.

### Pinball loss

For quantile forecast $q_\alpha$ and realization $y$,

$$
\rho_\alpha(y-q_\alpha)
=\max\{\alpha(y-q_\alpha),(\alpha-1)(y-q_\alpha)\}.
$$

`mean_pinball` averages over all seven configured quantile levels and all valid
cases. `pinball_q...` averages each level separately. This is the repository's
descriptive ranking statistic.

### Ensemble CRPS

For empirical ensemble $x_1,\ldots,x_M$,

$$
\operatorname{CRPS}
=\frac1M\sum_m|x_m-y|
-\frac1{2M^2}\sum_{m,n}|x_m-x_n|.
$$

The implementation evaluates the pair term in $O(M\log M)$ through sorted
order-statistic coefficients rather than materializing all $M^2$ pairs.

### Central interval score and coverage

For central coverage $c=1-\alpha$ with bounds $(l,u)$,

$$
\operatorname{IS}_\alpha(l,u;y)
=(u-l)+\frac{2}{\alpha}(l-y)\mathbf1(y<l)
+\frac{2}{\alpha}(y-u)\mathbf1(y>u).
$$

Coverage is the fraction satisfying $l\le y\le u$; width is $u-l$. The study
reports nominal coverages 0.50, 0.80, and 0.90. Because discrete empirical
quantiles and tied native values are used, exact nominal coverage is not
generally achievable.

### Weighted interval score

For $J$ configured central intervals with miscoverage $\alpha_j$, the
implemented WIS is

$$
\operatorname{WIS}
=\frac{
\tfrac12|y-m|+\sum_{j=1}^{J}\tfrac{\alpha_j}{2}
\operatorname{IS}_{\alpha_j}
}{
\tfrac12+\sum_{j=1}^{J}\tfrac{\alpha_j}{2}
},
$$

where $m$ is the empirical median.

## Multivariate entity-level scores

Joint scores assess the $K$-dimensional entity vector before aggregation. For
runtime control, only the first `min(M,512)` scenarios are used.

### Energy Score used by the experiment

The population/ensemble form is

$$
\operatorname{ES}(F,y)
=\mathbb E\|X-y\|_2-rac12\mathbb E\|X-X'\|_2.
$$

The final evaluator estimates the first expectation with all retained samples
and the second with cyclic adjacent pairs:

$$
\widehat{\operatorname{ES}}_{\mathrm{cyclic}}
=\frac1M\sum_{m=1}^{M}\|x_m-y\|_2
-\frac1{2M}\sum_{m=1}^{M}\|x_m-x_{m-1}\|_2,
$$

where index zero wraps to $M$. It is a paired Monte Carlo approximation, not
the usual all-pairs V-statistic. The standalone `energy_score` utility in
`evaluation/metrics.py` does implement the all-pairs expression in chunks, but
that utility is **not** what writes experimental `metrics.json`.

### Variogram Score

With power $p=0.5$ and unit weights over unique pairs,

$$
\operatorname{VS}_p
=\sum_{a<b}\left(
|y_a-y_b|^p-rac1M\sum_m|x_{m,a}-x_{m,b}|^p
\right)^2.
$$

This is sensitive to pairwise spatial contrast. Its magnitude grows with scale
and the number of entity pairs, so it cannot be compared directly between
groups of different size or units.

## Variable-cardinality diagnostic

For requested sizes 3, 7, and 15 that do not exceed group cardinality, the
evaluator selects the **first** $K'$ entity IDs in metadata order. It does not
average over random subsets. For $K'<K$, it recomputes each correlation on the
subset and uses at most 1,024 scenarios; for full $K$, it reuses the main
4,096-scenario result. Only mean pinball and the widest configured coverage are
written to `variable_k.csv`.

This tests executable cardinality behavior but confounds cardinality with
which specific entities occupy the prefix. A publication-quality sensitivity
study should sample or enumerate multiple subsets and report uncertainty.

## Missing and invalid test cases

A test case enters all method comparisons only if all $K$ truths are finite,
all forecast quantiles are finite, and none cross in the persisted marginal
grid. The same boolean mask is reused for every method and stored beside the
aggregate predictions. Thus method scores are paired over identical cases.
