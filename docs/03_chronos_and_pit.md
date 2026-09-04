# Frozen Chronos marginals, features, and PIT construction

## Why Chronos is frozen

The experiment is designed to isolate the contribution of spatial dependence.
Chronos-2 is therefore an immutable marginal forecaster:

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

For every origin, the wrapper returns:

$$
\widehat q\in\mathbb R^{K\times H\times Q},
\qquad
e\in\mathbb R^{K\times P\times D}.
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

No additional temporal model is learned. For each $(i,k,\tau)$, the
deterministic vector $v_{i,k,\tau}$ concatenates enabled components in this
exact order:

1. raw forecast-patch embedding $e_{i,k,p(\tau)}$;
2. normalized native-quantile shape;
3. marginal median;
4. log absolute 80% spread;
5. normalized within-patch position;
6. latitude and longitude.

Let $m=\widehat q_{0.5}$ and $s=\widehat q_{0.9}-\widehat q_{0.1}$. Quantile
shape component $j$ is

$$
r_j=\frac{\widehat q_j-m}{|s|+\epsilon_s},
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
\exists j:\widehat q_j>\widehat q_{j+1}.
$$

Crossing frequency is computed on the **raw** Chronos output overall, by
entity, and by lead before any repair. With `monotone_repair: none`, a crossing
invalidates that entity row and therefore the complete spatial vector.

With `monotone_repair: isotonic`, each finite row is replaced by the solution

$$
\widetilde{\boldsymbol q}
=\arg\min_{x_1\le\cdots\le x_Q}
\sum_{j=1}^{Q}(x_j-\widehat q_j)^2.
$$

The repaired values are used both for PIT construction and scenario
projection, and they are persisted in the cache. Raw crossing diagnostics are
still retained. This consistency is essential: fitting dependence with a
repaired CDF and evaluating against an unrepaired quantile grid would describe
different marginals.

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

For observation $y$, let $J$ be the index of the first predicted quantile
value greater than or equal to $y$; if none exists, $J=Q$. Then

$$
u=c_J,
\qquad
z=\Phi^{-1}\!\left(\min(1-\epsilon,max(\epsilon,u))\right),
\quad\epsilon=10^{-7}.
$$

Implementation uses left-sided `searchsorted`, so equality with a predicted
quantile belongs to the lower value interval ending at that quantile. Tied
isotonic values are handled deterministically by the first equal entry.

This is a pseudo-PIT, not a randomized PIT. Even under a perfectly calibrated
forecast it is supported only on $Q+1$ locations and its Gaussianized scores
are correspondingly discrete. Gaussian-copula fitting is therefore a
pseudo-likelihood procedure.

## Complete spatial vectors

An $(i,\tau)$ case is valid only if every entity has finite truth, every native
forecast value is finite, and every quantile row is noncrossing after the
configured repair. Formally,

$$
V_{i,\tau}=\prod_{k=1}^{K}V_{i,k,\tau}.
$$

When $V_{i,\tau}=0$, all $u_{i,k,\tau}$ and $z_{i,k,\tau}$ are stored as
`NaN`. This prevents each model from learning a differently composed group.

## Marginal invariance during scenario generation

The PIT cells and scenario projection use related but deliberately different
partitions of probability space. Scenario uniforms are mapped to the nearest
native probability level using boundaries

$$
b_0=0,\quad b_j=\frac{q_j+q_{j+1}}2\ (j=1,\ldots,Q-1),\quad b_Q=1.
$$

If $U\in[b_j,b_{j+1})$ (with the implemented right-boundary convention), the
scenario value is exactly $\widehat q_{j+1}$ under one-based indexing. There
is no interpolation. Since every Gaussian-copula component has a uniform
marginal, each method assigns the same probability mass $b_{j+1}-b_j$ to the
same Chronos value. Only the joint indices across entities change.

Consequences include finite scenario support, ties, no values below the lowest
or above the highest native quantile, and aggregate quantiles that are Monte
Carlo order statistics rather than sums of equally labelled marginal
quantiles.
