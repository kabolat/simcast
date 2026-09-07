# Dependence models M0--M4

## Shared Gaussian-copula construction

For forecast instance $i$, lead $\tau$, and static group $g$, each method
supplies

$$
R_{g,\tau}^{(i)}\in\mathbb R^{K_g\times K_g}.
$$

For scenario $m$,

$$
\eta^{(m)}\sim\mathcal N(0,I_{K_g}),\qquad
x^{(i,m)}=L_{g,\tau}^{(i)}\eta^{(m)},
$$

$$
L_{g,\tau}^{(i)}L_{g,\tau}^{(i)\mathsf T}=R_{g,\tau}^{(i)},
\qquad
u_{k,\tau}^{(i,m)}=\Phi(x_k^{(i,m)}).
$$

The uniforms are projected through the same fixed FM-derived discrete marginal
grid for every method. The grid is the raw Chronos-2 native grid for most
groups and the deterministically repaired Chronos grid for solar. No dependence
model changes any marginal quantile value.

Every correlation matrix is symmetric, positive definite after stabilization,
has unit diagonal, and covers the complete ordered $\mathcal E_g$.

## M0: independent Gaussian copula

M0 is the independent-copula or null-dependence baseline:

$$
R_{g,\tau}^{(i)}=I_{K_g}.
$$

It estimates no parameters; `fit` records only entity IDs and lead count.
Independence concerns scenario ranks conditional on the fixed forecasts. It
does not assert that raw loads or empirical forecast errors are independent.

## M1: static lead-specific Gaussian copula

M1 estimates one training-only matrix for each lead:

$$
\widehat\Sigma_{g,\tau}^{\mathrm{LW}}
=\operatorname{LedoitWolf}\!\left(
\left\{\mathbf z_{g,\tau}^{(i)}:
i\in\mathcal D_{\mathrm{train}},V_{g,\tau}^{(i)}=1\right\}
\right).
$$

Ledoit--Wolf shrinkage combines the empirical covariance with a
well-conditioned structured target using a data-estimated shrinkage intensity.
This is safer than an unregularized covariance when $K_g$ is not tiny relative
to the number of complete cases.

Covariance is converted to correlation by

$$
R_{ab}=\frac{\Sigma_{ab}}{\sqrt{\Sigma_{aa}\Sigma_{bb}}}.
$$

The implementation symmetrizes, clips negative eigenvalues to zero,
reconstructs, normalizes the diagonal, adds positive jitter, normalizes again,
and fixes the diagonal exactly at one. Numerically nonpositive variances are
treated as independent from other entities before projection. At least two
complete full-group vectors are required per estimated matrix.

With `share_across_leads: true`, complete vectors are pooled over leads and one
matrix is repeated across $H$. The baseline uses the default lead-specific form.
M1 never applies entity selection, regardless of the legacy subset flag that
appeared in some old resolved configurations.

## Shared low-rank parameterization

M2 and M3 output loading $\lambda_{k,\tau}^{(i)}\in\mathbb R^r$ and positive
uniqueness scale $\sigma_{k,\tau}^{(i)}$ for every
$k\in\mathcal E_g$. Their latent representation is

$$
z_{k,\tau}^{(i)}
=\lambda_{k,\tau}^{(i)\mathsf T}\xi
+\sigma_{k,\tau}^{(i)}\epsilon_k,
\qquad
\xi\sim\mathcal N(0,I_r),
\quad \epsilon_k\overset{\mathrm{iid}}\sim\mathcal N(0,1).
$$

Writing the complete loading matrix as $\Lambda_{g,\tau}^{(i)}$, the implied
covariance is

$$
\Sigma_{g,\tau}^{(i)}
=\Lambda_{g,\tau}^{(i)}\Lambda_{g,\tau}^{(i)\mathsf T}
+\operatorname{diag}\!\left(
\left\{\sigma_{k,\tau}^{(i)2}\right\}_{k\in\mathcal E_g}
\right),
$$

which is normalized to correlation. Positivity is enforced with

$$
\sigma_{k,\tau}^{(i)}
=\operatorname{softplus}(b_{k,\tau}^{(i)})+\sigma_{\min},
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
trainable entity-ID lookup exists. Although the model can process another
cardinality as a software property, full-group training and evaluation assert
that its input contains exactly the complete ordered $\mathcal E_g$.

## M3: set-aware conditional low-rank Gaussian copula

M3 first contextualizes every member of the complete group:

$$
h_{g,\tau}^{(i)}
=\operatorname{TransformerEncoder}_\theta\!\left(
\operatorname{Linear}(\operatorname{LayerNorm}(v_{g,\tau}^{(i)}))
\right).
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
all other full-group features through self-attention.

A padding-mask interface remains as a possible future software capability.
The full-group protocol never pads, drops, or selects entities; M3 always contextualizes all
$K_g$ members.

## M4: optional conditional RBF-kernel copula

M4 embeds each full-group entity with a shared MLP,

$$
h_{k,\tau}^{(i)}=g_\theta(v_{k,\tau}^{(i)})\in\mathbb R^{16},
$$

then constructs

$$
K_{ab}=\exp\!\left(
-\frac{\|h_a-h_b\|_2^2}{2\ell^2}
\right)+\delta_{ab}\nu,
$$

where $\ell=\operatorname{softplus}(\ell_{\mathrm{raw}})$ is learned from one
and nugget $\nu=10^{-3}$. Correlation normalization and jitter follow. The RBF
Gram matrix is positive semidefinite and permutation equivariant, but its
off-diagonal correlations are nonnegative, a substantive restriction.

The bounded smoke config uses at most 32 train and validation origins, five
epochs, patience two, and no entity selection. The full M4 config uses complete
partitions and the normal optimizer budget, also without entity selection.
Legacy smoke results remain diagnostic. The previous full transformer M4 run
used random subset augmentation and is also legacy; neither is a core
full-group result.

## Model hierarchy

| Method | Learns from PIT | Changes by instance | Full-group context before $R$ | Parameters independent of $K_g$ | Study status |
|---|---|---|---|---|---|
| M0 independent | no | no | no | yes | core baseline |
| M1 static | yes | no | empirical group estimate | no | core baseline |
| M2 conditional low-rank | yes | yes | no | yes | core method |
| M3 set-aware low-rank | yes | yes | self-attention | yes | core method |
| M4 conditional kernel | yes | yes | pairwise embedded distance | yes | optional diagnostic |

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
