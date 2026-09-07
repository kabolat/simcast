# Full-group training, scenario generation, and scoring

## Training cases

For group $g$, conditional training converts each valid $(i,\tau)$ into

$$
\left(V_{g,\tau}^{(i)},\mathbf z_{g,\tau}^{(i)}\right),
$$

where

$$
V_{g,\tau}^{(i)}\in\mathbb R^{K_g\times F},
\qquad
\mathbf z_{g,\tau}^{(i)}\in\mathbb R^{K_g}.
$$

A case exists only if all entity scores and features are finite. The code axis
is `[origin, entity, lead, feature]`; dataset construction flattens only the
valid `(origin, lead)` cases. Origin and lead indices are metadata rather than
direct network inputs. Lead information enters through the frozen forecast,
patch representation, and within-patch position.

The full-group protocol rejects subset training at configuration validation and
checks the entity dimension again immediately before every M2/M3 model call.
Every training and validation batch therefore contains exactly the complete
ordered $\mathcal E_g$. Missing one entity invalidates the case instead of
shrinking it.

M1 uses complete training vectors directly and does not use validation. M2 and
M3 train on the chronological training partition and use chronological
validation pseudo-likelihood for checkpoint selection. M4 follows the same
full-group rule when run as an optional diagnostic.

The generic collator still contains legacy subset functionality so old
exploratory artifacts can be understood, but no full-group config can enable it.

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
deterministic finite-cell pseudo-PIT values and because $R$ may be predicted
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

Python, NumPy, PyTorch, and CUDA generators are seeded. Deterministic PyTorch
algorithms are requested with warnings. Bitwise equality is not guaranteed
across hardware, CUDA/PyTorch versions, or kernels.

## Full-group scenario generation

For every valid $(g,i,\tau)$ and method:

1. obtain $R_{g,\tau}^{(i)}\in\mathbb R^{K_g\times K_g}$ and assert its exact
   full-group shape;
2. stabilize it and compute $L_{g,\tau}^{(i)}$;
3. draw $M=4096$ base normals $\eta^{(m)}\in\mathbb R^{K_g}$;
4. form copula uniforms
   $u_{k,\tau}^{(i,m)}=\Phi([L_{g,\tau}^{(i)}\eta^{(m)}]_k)$;
5. project each uniform to one fixed FM-derived quantile value, producing
   $\widetilde Y_{k,\tau}^{(i,m)}$;
6. sum the complete group:

$$
\widetilde A_{g,\tau}^{(i,m)}
=\sum_{k\in\mathcal E_g}\widetilde Y_{k,\tau}^{(i,m)}.
$$

Cases are processed in batches of 16. Each method recreates its generator with
seed `config.seed + batch_offset`, so all methods receive the same base normals
for the same cases. The current evaluator always uses these common random
numbers; the recorded configuration flag is not an independent-draw switch.

## Finite marginal projection

Scenario projection is not the historical PIT mapping. Given native levels
$q_1<\cdots<q_Q$, it defines nearest-level boundaries

$$
b_0=0,
\quad b_j=\frac{q_j+q_{j+1}}2\;(j=1,\ldots,Q-1),
\quad b_Q=1.
$$

Each uniform is assigned to exactly one native quantile value
$\hat y_{k,\tau,q_j}^{(i)}$. There is no value interpolation or tail
extrapolation. Because Gaussian-copula components are marginally uniform, M0,
M1, M2, and M3 have the same discrete entity-wise marginal masses. For solar,
the values are the persisted isotonic-repaired grid; for other groups they are
the raw native grid.

Aggregate quantiles are nearest empirical order statistics of the $M$
full-group aggregate samples. They are not sums of equally labeled marginal
quantiles:

$$
Q_\alpha\!\left(\sum_kY_k\right)
\ne\sum_kQ_\alpha(Y_k)
$$

in general.

## Aggregate scores

All metrics pool the same valid full-group origin-lead cases unless reported by
lead. Lower loss/score is better.

### Pinball loss

For aggregate quantile forecast $q_\alpha$ and observation $a$,

$$
\rho_\alpha(a-q_\alpha)
=\max\{\alpha(a-q_\alpha),(\alpha-1)(a-q_\alpha)\}.
$$

`mean_pinball` averages over evaluation levels
$0.05,0.10,0.25,0.50,0.75,0.90,0.95$ and all cases. It is the headline
descriptive ranking score.

### Ensemble CRPS

For aggregate ensemble $x_1,\ldots,x_M$,

$$
\operatorname{CRPS}
=\frac1M\sum_m|x_m-a|
-\frac1{2M^2}\sum_{m,n}|x_m-x_n|.
$$

The implementation evaluates the pair term in $O(M\log M)$ through sorted
order-statistic coefficients.

### Central intervals and WIS

For central coverage $c=1-\alpha$ with bounds $(l,u)$,

$$
\operatorname{IS}_\alpha(l,u;a)
=(u-l)+\frac{2}{\alpha}(l-a)\mathbf1(a<l)
+\frac{2}{\alpha}(a-u)\mathbf1(a>u).
$$

Coverage is the fraction with $l\le a\le u$; width is $u-l$. The baseline reports
central 0.50, 0.80, and 0.90 intervals. Coverage must be read with width and a
proper score.

For interval miscoverages $\alpha_j$, implemented WIS is

$$
\operatorname{WIS}
=\frac{
\tfrac12|a-m|+
\sum_j\tfrac{\alpha_j}{2}\operatorname{IS}_{\alpha_j}
}{
\tfrac12+\sum_j\tfrac{\alpha_j}{2}
},
$$

where $m$ is the empirical aggregate median.

## Joint full-group scores

Joint scores use the first `min(M,512)` members of the already generated
full-group ensemble. This is an ensemble-size selection, not entity
subsampling.

### Energy Score

For selected ensemble $x_1,\ldots,x_{M_J}\in\mathbb R^{K_g}$ and observed
vector $\mathbf y$, the baseline uses the empirical all-pairs estimator

$$
\widehat{\operatorname{ES}}
=\frac1{M_J}\sum_{m=1}^{M_J}\|x_m-\mathbf y\|_2
-\frac1{2M_J^2}\sum_{m=1}^{M_J}\sum_{n=1}^{M_J}
\|x_m-x_n\|_2.
$$

The pair sum is exact for the selected 512-member ensemble. `torch.cdist` is
chunked over 128 first-sample rows solely to limit memory; no cyclic pairing or
pair subsampling remains.

### Variogram Score

With $p=0.5$ and unit weights over unique entity pairs,

$$
\operatorname{VS}_p
=\sum_{a<b}\left(
|y_a-y_b|^p
-\frac1{M_J}\sum_m|x_{m,a}-x_{m,b}|^p
\right)^2.
$$

Its magnitude depends on scale and the number of entity pairs, so it is not
comparable across differently scaled groups.

## Full-group output scope

The evaluator produces only complete-group metrics, lead tables, correlation
matrices, and figures. It does not create prefix-$K$ tables,
variable-cardinality CSVs, or reduced-group figures. Architecture-level
variable-size tests remain only to ensure that shared-weight mathematics has
not been accidentally hard-coded to one $K_g$; they are not experiments.
