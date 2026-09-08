# Frozen Chronos forecasts, fixed marginal grids, features, and PIT construction

## Why Chronos is frozen

The experiment is designed to isolate the contribution of spatial dependence.
Chronos-2 is therefore a frozen marginal forecaster:

- source commit: `8589d1988e9676817548e9626738ff06b6ca6370`;
- model revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`;
- model ID: `amazon/chronos-2`;
- default inference dtype: `bfloat16`;
- all model parameters have `requires_grad=False`;
- the model remains in evaluation mode and inference runs under `no_grad`;
- each physical entity remains a separate Chronos group (`cross_learning=false`).

The wrapper uses the official `Chronos2Dataset` preprocessing, instance
scaling, patching, future-covariate alignment, and `target_idx_ranges`. Each
physical entity must have exactly one target row. It never replaces group IDs
with a common zero, which would make Chronos cross-learn between entities and
change the experimental intervention.

Only a single direct model forward pass is allowed. A request exceeding
`max_output_patches * output_patch_size` raises an error because autoregressive
unrolling would not yield one coherent set of output-patch representations.

## The minimal Chronos patch

The upstream model already computes a forecast-side representation immediately
before the quantile head but does not return it. `scripts/setup_chronos.sh`
checks out the exact source commit, applies
`patches/chronos2_forecast_embeds.patch`, and installs that source editable with
`uv`. The patch adds a `forecast_embeds` field to `Chronos2Output` and exposes
the existing tensor. It does not alter attention, scaling, loss, encoder
states, quantile values, or inference logic.

The setup is idempotent: it detects an already-applied patch before attempting
to apply it again.

## Returned marginal and representation tensors

For every forecast instance $i$ and group $g$, the wrapper returns:

$$
\widehat y^{(i)}\in\mathbb R^{K_g\times H\times Q},
\qquad
e^{(i)}\in\mathbb R^{K_g\times P\times D}.
$$

In the observed model configuration, $Q=21$, output patch size $S=16$,
$H=96$, $P=\lceil H/S\rceil=6$, and hidden width $D=768$. Quantile predictions
are retained in float32. Embeddings are created in float32, stored in float16
to reduce cache size, then converted back to float32 for adapter training.

The patch index for one-based lead $\tau$ is

$$
p(\tau)=\left\lfloor\frac{\tau-1}{S}\right\rfloor.
$$

All 16 leads in an output patch reuse the same stored representation. The
within-patch position feature described below restores lead position within
that block.

## Conditional-model feature vector

The feature vector is evidence available at $t^{(i)}$, not a new target model.
It parameterizes a candidate conditional copula $C_{g,\tau}^{(i)}$ while
leaving $F_{k,\tau}^{(i)}$ unchanged. Thus an M2/M3 improvement is evidence
that frozen forecast context is associated with residual spatial rank
dependence; it is not evidence that a dependence model improved an entity
marginal.

No additional temporal model is learned. For each $(i,k,\tau)$, the
deterministic vector $v_{k,\tau}^{(i)}$ concatenates enabled components in this
exact order:

1. raw forecast-patch embedding $e_{k,p(\tau)}^{(i)}$;
2. normalized native-quantile shape;
3. marginal median;
4. log absolute 80% spread;
5. normalized within-patch position;
6. latitude and longitude.

Let $m=\hat y_{k,\tau,0.5}^{(i)}$ and
$s=\hat y_{k,\tau,0.9}^{(i)}-\hat y_{k,\tau,0.1}^{(i)}$. Quantile
shape component $j$ is

$$
r_j=\frac{\hat y_{k,\tau,q_j}^{(i)}-m}{|s|+\epsilon_s},
\qquad \epsilon_s=10^{-6}.
$$

The scalar distribution components are $m$ and
$\log(|s|+\epsilon_s)$. The within-patch component is

$$
w_\tau=\frac{(\tau-1)\bmod S}{\max(S-1,1)}.
$$

With the native $Q=21$ grid and all defaults enabled, the feature dimension is
$D+Q+1+1+1+2=794$.

All non-embedding components are concatenated into one scalar block. For each
scalar feature $f$, mean $\mu_f$ and population standard deviation $s_f$ are
estimated over **all origin, entity, and lead positions in the training
partition only**. The transformed value is $(x_f-\mu_f)/s_f$. If the standard
deviation is at most $10^{-6}$, scale is set to one. These statistics are
frozen for validation and test and stored in every conditional checkpoint.

`layer_normalize_embedding` is slightly historical naming: it controls the
input `LayerNorm` of M2, while M3 and M4 always begin with layer normalization.
No trainable entity-ID embedding is supported; enabling the configuration flag
raises `NotImplementedError`.

## Quantile crossings

A quantile row crosses when

$$
\exists j:\hat y_{k,\tau,q_j}^{(i)}>\hat y_{k,\tau,q_{j+1}}^{(i)}.
$$

Crossing frequency is computed on the **raw** Chronos output overall, by
entity, and by lead before any repair. With `monotone_repair: none`, a crossing
invalidates that entity row and therefore the complete spatial vector.

With `monotone_repair: isotonic`, each finite row is replaced by the solution

$$
\widetilde{\boldsymbol q}
=\arg\min_{x_1\le\cdots\le x_Q}
\sum_{j=1}^{Q}\left(x_j-\hat y_{k,\tau,q_j}^{(i)}\right)^2.
$$

The repaired values are used both for PIT construction and scenario
projection, and they are persisted in the cache. Raw crossing diagnostics are
still retained. For most groups, the FM-derived grid is the raw Chronos grid;
for solar it is this deterministic repair. In both cases the resulting grid is
fixed across M0--M4, and no dependence model can modify it.

Solar is the only supplied configuration that enables repair. Its raw crossing
rate is 25.290%, mainly at zero-output hours. Without repair, 15,477 of 33,312
origin-lead vectors are invalid and eight leads have no complete training
vectors, making lead-wise M1 inestimable. With repair, only three test cases are
dropped for missing truth.

## Discretized PIT

Chronos supplies values only at increasing native probability levels
$0<q_1<\cdots<q_Q<1$. Simcast intentionally does not invent a continuous CDF.
Define probability edges

$$
a=(0,q_1,\ldots,q_Q,1)
$$

and cell locations

$$
c_j=\frac{a_j+a_{j+1}}2,\qquad j=0,\ldots,Q.
$$

For observation $y_{k,\tau}^{(i)}$, let $J$ be the index of the first predicted
quantile value greater than or equal to the observation; if none exists,
$J=Q$. Then

$$
u_{k,\tau}^{(i)}=c_J,
\qquad
z_{k,\tau}^{(i)}=\Phi^{-1}\!\left(
\min(1-\epsilon,\max(\epsilon,u_{k,\tau}^{(i)}))
\right),
\quad\epsilon=10^{-7}.
$$

Implementation uses left-sided `searchsorted`, so equality with a predicted
quantile belongs to the lower value interval ending at that quantile. Tied
isotonic values are handled deterministically by the first equal entry.

This is a pseudo-PIT, not a randomized PIT. Even under a perfectly calibrated
forecast it is supported only on $Q+1$ locations and its Gaussianized scores
are correspondingly discrete. Gaussian-copula fitting is therefore a
pseudo-likelihood procedure.

The two maps in the study answer different questions and must not be
conflated. The historical map $y\mapsto u$ allocates an observation to a finite
PIT cell so that dependence can be learned from realized outcomes. The forward
map $U\mapsto\hat y$ allocates a simulated uniform to a finite native-quantile
value so that scenarios retain the frozen marginal law. Neither is a continuous
CDF reconstruction.

### Training-frequency PIT sensitivity

The confirmatory primary analysis retains the nominal finite-cell midpoint
above. A separate sensitivity fits empirical cell probabilities using training
origins only for every entity and lead. Cell $c$ is mapped to the midpoint of
its frozen empirical mass,

$$
\tilde u_{k,\tau,c}
=\sum_{r<c}\hat p_{k,\tau,r}+\tfrac12\hat p_{k,\tau,c},
\qquad
\tilde z_{k,\tau,c}=\Phi^{-1}(\tilde u_{k,\tau,c}).
$$

Validation and test frequencies never enter this map. The sensitivity changes
only dependence scores used for fitting and pseudo-NLL diagnosis. It does not
interpolate the CDF or alter persisted/repaired Chronos quantiles and scenario
projection.

## Complete spatial vectors

An $(i,\tau)$ case for group $g$ is valid only if every
$k\in\mathcal E_g$ has a finite observation, every FM-derived quantile value is
finite, and every quantile row is noncrossing after the configured repair.
Formally,

$$
V_{g,\tau}^{(i)}=\prod_{k\in\mathcal E_g}V_{k,\tau}^{(i)}.
$$

When $V_{g,\tau}^{(i)}=0$, all $u_{k,\tau}^{(i)}$ and
$z_{k,\tau}^{(i)}$ for $k\in\mathcal E_g$ are stored as
`NaN`. This prevents each model from learning a differently composed group.

## Marginal invariance during scenario generation

The PIT cells and scenario projection use related but deliberately different
partitions of probability space. Scenario uniforms are mapped to the nearest
native probability level using boundaries

$$
b_0=0,\quad b_j=\frac{q_j+q_{j+1}}2\ (j=1,\ldots,Q-1),\quad b_Q=1.
$$

If $U\in[b_j,b_{j+1})$ (with the implemented right-boundary convention), the
scenario value is exactly $\hat y_{k,\tau,q_{j+1}}^{(i)}$ under one-based indexing. There
is no interpolation. Since every Gaussian-copula component has a uniform
marginal, each method assigns the same probability mass $b_{j+1}-b_j$ to the
same fixed FM-derived quantile value. Only the joint indices across entities
change.

Consequences include finite scenario support, ties, no values below the lowest
or above the highest native quantile, and aggregate quantiles that are Monte
Carlo order statistics rather than sums of equally labelled marginal
quantiles.
