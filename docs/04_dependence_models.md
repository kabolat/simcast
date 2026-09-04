# Dependence models M0--M4

## Shared Gaussian-copula construction

For a valid origin and lead, every method supplies a correlation matrix
$R_{i,\tau}\in\mathbb R^{K\times K}$. A scenario is generated as

$$
\eta^{(m)}\sim\mathcal N(0,I_K),\qquad
x^{(m)}=L_{i,\tau}\eta^{(m)},\quad
L_{i,\tau}L_{i,\tau}^{\mathsf T}=R_{i,\tau},
$$

$$
u_k^{(m)}=\Phi(x_k^{(m)}).
$$

The uniform vector is projected independently through the same fixed discrete
Chronos marginals for all methods. Valid model matrices are symmetric,
positive definite after numerical jitter, and have unit diagonal.

## M0: independent copula

M0 sets

$$
R_{i,\tau}=I_K
$$

for every origin and lead. `fit` records only entity IDs and the number of
leads; it estimates no parameters. It is the causal-control-like baseline for
the experiment: aggregate differences relative to M0 are attributable to
cross-entity dependence assumptions.

Independence refers to scenario ranks conditional on the frozen forecasts. It
does not claim that raw loads or forecast errors are empirically independent.

## M1: static Gaussian copula

M1 estimates correlation from training-only Gaussianized PIT vectors. By
default, one matrix is fitted per lead:

$$
\widehat\Sigma_\tau^{\mathrm{LW}}
=\operatorname{LedoitWolf}\left(
\{z_{i,:,\tau}:i\in\mathcal D_{\mathrm{train}},V_{i,\tau}=1\}
\right).
$$

Ledoit--Wolf shrinkage combines an empirical covariance matrix with a
well-conditioned structured target using an estimated shrinkage intensity.
This is preferable to an unregularized sample covariance when $K$ is not tiny
relative to the number of complete cases.

Covariance is converted to correlation as

$$
R_{ab}=\frac{\Sigma_{ab}}{\sqrt{\Sigma_{aa}\Sigma_{bb}}}.
$$

The implementation then symmetrizes, clips negative eigenvalues to zero,
reconstructs the matrix, normalizes its diagonal, adds positive diagonal
jitter, normalizes again, and sets the diagonal exactly to one. Zero or
numerically nonpositive covariance variances are treated as independent from
other entities before the projection. At least two complete samples are
required for every estimated matrix.

With `share_across_leads: true`, valid vectors from all leads are pooled and one
estimate is repeated across $H$. The supplied experiments use the default
lead-specific variant.

Properties:

- no origin conditioning;
- $H K(K-1)/2$ effective off-diagonal values after shrinkage;
- requested entity subsets are principal submatrices in requested order;
- no neural optimization or validation-based early stopping.

## Shared low-rank covariance parameterization

M2 and M3 output one loading vector $\lambda_k\in\mathbb R^r$ and positive
uniqueness scale $\sigma_k$ per entity. Their latent representation is

$$
z_k=\lambda_k^{\mathsf T}\xi+\sigma_k\epsilon_k,qquad
\xi\sim\mathcal N(0,I_r),\quad
\epsilon_k\overset{\mathrm{iid}}\sim\mathcal N(0,1).
$$

Hence

$$
\Sigma=\Lambda\Lambda^{\mathsf T}+\operatorname{diag}(\sigma_1^2,\ldots,\sigma_K^2),
$$

followed by covariance-to-correlation normalization and jitter. Positivity is
enforced by

$$
\sigma_k=\operatorname{softplus}(a_k)+\sigma_{\min},
\qquad\sigma_{\min}=10^{-3}.
$$

This construction is positive semidefinite before uniqueness and strictly
positive definite under positive uniqueness/jitter. Rank $r=4$ constrains the
shared covariance component but the diagonal uniqueness preserves full matrix
rank. Factor orientation is not identifiable: $\Lambda$ and $\Lambda Q$ yield
the same covariance for any orthogonal $Q$. Individual loading coordinates
must therefore not be given standalone physical interpretations.

## M2: conditional low-rank copula

M2 applies the same entity-wise network $f_\theta$ independently to every
feature vector:

$$
(\lambda_{i,k,\tau},a_{i,k,\tau})
=f_\theta(v_{i,k,\tau}).
$$

The default architecture is:

```text
LayerNorm(F)
Linear(F,256) -> GELU -> Dropout(0.1)
Linear(256,128) -> GELU -> Dropout(0.1)
Linear(128,r+1), r=4
```

The same weights are used for every entity and the output size is independent
of $K$. M2 is permutation equivariant in the limited element-wise sense: if
entity features are permuted, the loadings and resulting matrix are permuted
accordingly. However, entity $k$ cannot inspect the other entities before
choosing $\lambda_k$; cross-entity interaction occurs only through the inner
products $\lambda_a^{\mathsf T}\lambda_b$.

Location and forecast features can implicitly distinguish entities, but there
is no learned ID lookup. A previously unseen entity can technically be passed
if its feature semantics match, although the current checkpoint loader and
evaluator deliberately require evaluation IDs to be a subset of fitted IDs.

## M3: set-aware low-rank copula

M3 first maps each entity to a shared latent width and applies a transformer
encoder across the current entity set:

$$
h_{i,:,\tau}
=\operatorname{TransformerEncoder}_\theta
\left(\operatorname{Linear}(\operatorname{LayerNorm}(v_{i,:,\tau}))\right).
$$

A shared linear head maps each contextual representation to
$(\lambda_k,a_k)$. Defaults are model width 128, two encoder layers, four
attention heads, pre-norm layers, GELU feed-forward blocks of width 512,
dropout 0.1, and rank four.

There are no positional encodings and no entity-ID embeddings. Self-attention
is therefore permutation equivariant:

$$
f_\theta(PV)=P f_\theta(V),
\qquad
R_\theta(PV)=P R_\theta(V)P^{\mathsf T}
$$

for any permutation matrix $P$ when dropout is disabled at evaluation. Tests
assert this property numerically. Unlike M2, an entity's factor parameters can
change with the features of other members in the current set.

An optional padding mask exists for batched variable-cardinality sets. Masked
loadings are zero, masked uniqueness is one, and the caller must still avoid
interpreting padded rows as physical members. Current training uses
fixed-cardinality batches or randomly selected same-size subsets, so no padding
is needed in the main experiment.

## M4: conditional RBF-kernel copula

M4 embeds each entity feature with a shared MLP:

$$
h_k=g_\theta(v_k)\in\mathbb R^{16},
$$

using LayerNorm, hidden widths 256 and 128, GELU, and dropout 0.1. It constructs

$$
K_{ab}=\exp\left(-\frac{\|h_a-h_b\|_2^2}{2\ell^2}\right)
+\delta_{ab}\nu,
$$

where $\ell=\operatorname{softplus}(\ell_{\mathrm{raw}})$ is learned from an
initial value of one and nugget $\nu=10^{-3}$. Correlation normalization and
jitter follow. The RBF Gram matrix is positive semidefinite by construction.

M4 is permutation equivariant because the embedding network is shared and
pairwise distances do not depend on order. Its correlations are nonnegative
before numerical normalization, a substantive restriction compared with the
factor models.

The repository enforces `smoke_only: true`. The high-level runner uses at most
32 training origins and 32 validation origins that contain at least one
complete lead, disables subset training, caps optimization at five epochs, and
uses patience at most two. It still evaluates the resulting checkpoint on the
full test set. These intentionally unequal training resources mean M4 numbers
must be read only as a systems/feasibility smoke result.

## Model comparison

| Method | Learns from PIT | Changes by origin | Set interaction before $R$ | Parameter count independent of $K$ | Intended status |
|---|---|---|---|---|---|
| M0 independent | no | no | no | yes | core baseline |
| M1 static | yes | no | empirical group estimate | no, matrix grows with $K$ | core baseline |
| M2 conditional low-rank | yes | yes | no | yes | core method |
| M3 set-aware low-rank | yes | yes | self-attention | yes | core method |
| M4 conditional kernel | yes | yes | pairwise embedded distance | yes | bounded smoke only |

## Numerical stabilization

Several layers intentionally repeat stabilization because matrices pass through
different public interfaces. Correlations are symmetrized, forced to unit
diagonal, given jitter, then renormalized to unit diagonal before Cholesky
factorization. Adding jitter without the final normalization would change
Gaussian marginal variances and violate the copula construction. A Cholesky
failure is allowed to surface rather than silently replacing the model matrix.
