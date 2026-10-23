# Dependence models M0--M4

**Operational realization.** Valid method YAML fields for M0--M4 are listed in
the [configuration reference](../technical/configuration_reference.md). The
[usage guide](../technical/usage_guide.md) shows singular and composite
commands that fit these methods.

## Shared Gaussian-copula construction

For forecast instance $i$, lead $\tau$, and static group $g$, each method
supplies an effective correlation on the case's active set. Its dimension is
$K_g$ in fixed mode and $|\mathcal A_{g,\tau}^{(i)}|$ in variable mode:

$$
R_{g,\tau}^{(i)}\in\mathbb R^{K_{\mathrm{active}}\times K_{\mathrm{active}}},
\qquad K_{\mathrm{active}}=|\mathcal A_{g,\tau}^{(i)}|.
$$

In variable mode the active matrix is the principal submatrix indexed by
$\mathcal A_{g,\tau}^{(i)}$. The canonical universe and entity IDs remain
fixed; cached correlation artifacts are scattered back to that full axis with
inactive entries marked by identity rows and columns.

For scenario $m$,

$$
\eta^{(m)}\sim\mathcal N(0,I_{K_{\mathrm{active}}}),\qquad
x^{(i,m)}=L_{g,\tau}^{(i)}\eta^{(m)},
$$

$$
L_{g,\tau}^{(i)}L_{g,\tau}^{(i)\mathsf T}=R_{g,\tau}^{(i)},
\qquad
u_{k,\tau}^{(i,m)}=\Phi(x_k^{(i,m)}).
$$

The uniforms are projected through the same configured finite-quantile
marginal law for every method. Its knots are the raw Chronos-2 native grid for
most groups and the deterministically repaired Chronos grid where isotonic
repair is declared. The base selects discretized or piecewise-linear
projection; no dependence model changes that selection or any marginal
quantile value.

Every effective correlation matrix is symmetric, positive definite after
stabilization, and has unit diagonal.

## M0: independent Gaussian copula

M0 is the independent-copula or null-dependence baseline:

$$
R_{g,\tau}^{(i)}=I_{K_{\mathrm{active}}}.
$$

It estimates no parameters; `fit` records only entity IDs and lead count.
Independence concerns scenario ranks conditional on the fixed forecasts. It
does not assert that raw loads or empirical forecast errors are independent.

## M1: static lead-specific Gaussian copula

M1 estimates one training-only matrix for each lead:

$$
\widehat\Sigma_{g,\tau}^{\mathrm{LW}}
=\mathrm{LedoitWolf}(
\{\mathbf z_{g,\tau}^{(i)}:
i\in\mathcal D_{\mathrm{train}},V_{g,\tau}^{(i)}=1\}
).
$$

For complete fixed-mode data, Ledoit--Wolf shrinkage combines empirical
covariance with a well-conditioned structured target using a data-estimated
shrinkage intensity. This exact complete-data branch is retained. In variable
mode, cases below `protocol.min_entities_per_case` are excluded; remaining
entity pairs use only joint finite PIT observations. Pairs with fewer than two
overlaps default to zero correlation, and the resulting matrix is projected to
positive semidefinite form and normalized. Per-pair overlap counts and the
estimator mode are saved with the fit.

Covariance is converted to correlation by

$$
R_{ab}=\frac{\Sigma_{ab}}{\sqrt{\Sigma_{aa}\Sigma_{bb}}}.
$$

The implementation symmetrizes, clips negative eigenvalues to zero,
reconstructs, normalizes the diagonal, adds positive jitter, normalizes again,
and fixes the diagonal exactly at one. Numerically nonpositive variances are
treated as independent from other entities before projection. At least two
observations are required for each pairwise estimate.

With `share_across_leads: true`, eligible vectors are pooled over leads and one
matrix is repeated across $H$. The baseline uses the default lead-specific form.

## Shared low-rank parameterization

M2 and M3 output loading $\lambda_{k,\tau}^{(i)}\in\mathbb R^r$ and positive
uniqueness scale $\sigma_{k,\tau}^{(i)}$ for every active entity
$k\in\mathcal A_{g,\tau}^{(i)}$. Their latent representation is

$$
z_{k,\tau}^{(i)}
=\lambda_{k,\tau}^{(i)\mathsf T}\xi
+\sigma_{k,\tau}^{(i)}\epsilon_k,
\qquad
\xi\sim\mathcal N(0,I_r),
\quad \epsilon_k\overset{\mathrm{iid}}\sim\mathcal N(0,1).
$$

Writing the active-set loading matrix as $\Lambda_{g,\tau}^{(i)}$, the implied
covariance is

$$
\Sigma_{g,\tau}^{(i)}
=\Lambda_{g,\tau}^{(i)}\Lambda_{g,\tau}^{(i)\mathsf T}
+\mathrm{diag}(
\{\sigma_{k,\tau}^{(i)2}\}_{k\in\mathcal A_{g,\tau}^{(i)}}
),
$$

which is normalized to correlation. Positivity is enforced with

$$
\sigma_{k,\tau}^{(i)}
=\mathrm{softplus}(b_{k,\tau}^{(i)})+\sigma_{\min},
\qquad \sigma_{\min}=10^{-3}.
$$

The shared component has default rank $r=4$; uniqueness preserves full matrix
rank. Factors are rotation non-identifiable: $\Lambda$ and $\Lambda Q$ imply
the same covariance for any orthogonal $Q$. Individual loading coordinates
therefore have no standalone physical interpretation.

## M2: conditional low-rank Gaussian copula

M2 applies one shared entity-wise network:

$$
(\lambda_{k,\tau}^{(i)},b_{k,\tau}^{(i)})
=f_\theta(v_{k,\tau}^{(i)}).
$$

The default architecture is:

```text
LayerNorm(F)
Linear(F,256) -> GELU -> Dropout(0.1)
Linear(256,128) -> GELU -> Dropout(0.1)
Linear(128,r+1), r=4
```

Weights and parameter count do not depend on $K_g$. Reordering entities
reorders the outputs and the resulting correlation. Entity $k$ does not inspect
other entity features before choosing its parameters; interaction occurs
through $\lambda_a^{\mathsf T}\lambda_b$.

Location and forecast features can indirectly distinguish entities, but no
trainable entity-ID lookup exists. Variable-mode cases are compacted to their
active canonical entities; batches pad only to the largest active count in
that batch and carry an explicit padding mask.

## M3: set-aware conditional low-rank Gaussian copula

M3 first contextualizes every member of the active set:

$$
h_{g,\tau}^{(i)}
=\mathrm{TransformerEncoder}_\theta(
\mathrm{Linear}(\mathrm{LayerNorm}(v_{g,\tau}^{(i)}))
).
$$

A shared head maps each contextual representation to $(\lambda_k,b_k)$.
Defaults are width 128, two encoder layers, four attention heads, pre-norm,
GELU feed-forward width 512, dropout 0.1, and rank four.

There are no positional encodings or entity-ID embeddings. For permutation
matrix $P$,

$$
f_\theta(PV)=P f_\theta(V),
\qquad
R_\theta(PV)=P R_\theta(V)P^{\mathsf T}
$$

at evaluation. Tests verify this numerically. Unlike M2, each entity can use
all other active-set features through self-attention.

Padded rows are masked as attention keys and values and cannot alter the active
entities' correlations. The active set itself is the same PIT-valid subset used
by every method.

## M4: conditional RBF-kernel copula

M4 embeds each active entity with a shared MLP,

$$
h_{k,\tau}^{(i)}=g_\theta(v_{k,\tau}^{(i)})\in\mathbb R^{16},
$$

then constructs

$$
K_{ab}=\exp(
-\frac{\|h_a-h_b\|_2^2}{2\ell^2}
)+\delta_{ab}\nu,
$$

where $\ell=\mathrm{softplus}(\ell_{\mathrm{raw}})$ is learned from one
and nugget $\nu=10^{-3}$. Correlation normalization and jitter follow. The RBF
Gram matrix is positive semidefinite and permutation equivariant, but its
off-diagonal correlations are nonnegative, a substantive restriction.

M4 is fitted on every admitted training case and selected by
chronological validation pseudo-NLL, using the same partition and optimization
semantics as M2 and M3. Its supplied method file uses the same 100-epoch
maximum and patience 12. Preliminary composites may explicitly reduce these
budgets for M2, M3, and M4 alike; no M4-specific data truncation exists.

## Model hierarchy

| Method | Learns from PIT | Changes by instance | Context before $R$ | Parameters independent of $K_g$ | Study status |
|---|---|---|---|---|---|
| M0 independent | no | no | no | yes | core baseline |
| M1 static | yes | no | empirical group estimate | no | core baseline |
| M2 conditional low-rank | yes | yes | no | yes | core method |
| M3 set-aware low-rank | yes | yes | self-attention | yes | core method |
| M4 conditional kernel | yes | yes | pairwise embedded distance | yes | core method |

## Numerical-stabilization audit

Stabilization is intentionally repeated at different interfaces:

1. M1 adds jitter after PSD projection and renormalizes. M2/M3 add jitter while
   converting low-rank covariance to correlation. M4 normalizes its
   nugget-augmented kernel and applies model-stage jitter.
2. M2--M4 training then symmetrizes, restores unit diagonal, adds
   likelihood-stage jitter, renormalizes, and performs Cholesky factorization.
3. Scenario generation for every method again symmetrizes, normalizes, adds
   sampling-stage jitter, renormalizes, and performs Cholesky factorization.

Conditional training therefore encounters model-stage and likelihood-stage
jitter; scenario generation encounters model-stage (except M0) and
sampling-stage jitter. Since every addition is followed by diagonal
renormalization, this is not equivalent to simply adding the constants.

The cleanup does not consolidate these operations because doing so would alter
the learned objective or scenario law and make old checkpoints numerically
incomparable. Tests verify unit diagonal, symmetry, and strict positive
definiteness after repeated stabilization. Cholesky failure remains visible
rather than silently substituting independence.

## Worked comparisons and implementation guidance

For $K_g=3$, M0 uses $I_3$: a high sampled rank for one entity conveys no
information about the others. M1 may instead estimate

$$
R=\begin{pmatrix}1&0.7&0.2\\0.7&1&0.1\\0.2&0.1&1\end{pmatrix},
$$

and use it at every origin for that lead. M2 and M3 may produce a different
matrix at each origin; M3 additionally lets one entity's context affect all
contextual representations. M4 makes correlations decay with learned
embedding distance and cannot represent negative off-diagonal values.

| Method | Configuration | Controlling class |
|---|---|---|
| M0 | method file with `family: independent` | `IndependentCopula` |
| M1 | `method.model.*` in the static file | `StaticGaussianCopula` |
| M2 | `method.features/model/optimization` | `ConditionalLowRankGaussianCopula` |
| M3 | `method.features/model/optimization` | `SetAwareLowRankGaussianCopula` |
| M4 | `method.features/model/optimization` | `ConditionalKernelGaussianCopula` |

These classes are in `src/simcast/dependence/`. Conditional method files place
the information in `features` and numerical estimation in `optimization`.
