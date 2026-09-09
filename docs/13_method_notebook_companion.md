# Scientific method-notebook companion

The notebooks in `notebooks/dependency_methods/` form an independent reading
path. They examine the same cached scientific objects as registered
command-line experiments, but each restarts the notation and derivation so no
earlier notebook is prerequisite.

The shared transformation is

$$
\{(\hat y_{k,\tau,q_j}^{(i)},q_j),y_{k,\tau}^{(i)}\}
\longrightarrow u_{k,\tau}^{(i)}
\longrightarrow z_{k,\tau}^{(i)}
\longrightarrow R_{g,\tau}^{(i)}
\longrightarrow \widetilde A_{g,\tau}^{(i,m)}.
$$

The methods differ only in the construction of the same-lead correlation
matrix. Every notebook preserves the complete ordered group: one invalid
entity invalidates $(i,\tau)$ and never produces a smaller fitted group.

| Notebook | Scientific question | Correlation construction |
|---|---|---|
| `00_m0_independent_copula.ipynb` | What remains under independent residual ranks? | $R=I_{K_g}$ |
| `01_m1_static_gaussian_copula.ipynb` | What persistent training-period rank association exists? | lead-specific PIT correlation |
| `02_m2_conditional_low_rank_copula.ipynb` | Can frozen entity context predict dependence? | $\Lambda\Lambda^\mathsf T+\operatorname{diag}(\sigma^2)$ |
| `03_m3_set_aware_low_rank_copula.ipynb` | Does full-group contextualization add information? | self-attention, then the same low-rank form |
| `04_m4_conditional_kernel.ipynb` | What follows from a smooth nonnegative kernel restriction? | RBF Gram matrix |

## Organization of each monograph

Each notebook defines $i,t^{(i)},g,\mathcal E_g,K_g,\tau,Y,y,F,A$ and the
complete-group indicator $V$ before using them. It then states the statistical
hypothesis, gives a numerical or graphical example, identifies the relevant
configuration arguments and mathematical implementation, and only then offers
disabled estimation/evaluation cells. No displayed equation depends on a
symbol introduced in another notebook.

## Read-only default and worked example

`TRAIN_IF_MISSING = False` and `RUN_EVALUATION = False` make a monograph
read-only. For example, opening M2 reads the selected frozen marginal cache and
visualizes an illustrative low-rank correlation; it does not fit parameters.
Changing `TRAIN_IF_MISSING` deliberately calls the same `train_from_config`
transformation as the command-line experiment. Changing `RUN_EVALUATION`
opens test labels and writes scores to the explicitly chosen new directory.

## Implementation guidance and interpretation

`BASE_FILE` selects the frozen-marginal experiment and `METHOD_FILE` selects
one dependence hypothesis. `BASE_OVERRIDES` and `METHOD_OVERRIDES` apply only
to their corresponding role. Cache
compatibility is determined from scientific metadata, not a convenient name.
Notebook diagrams are explanatory. Formal comparisons require predeclared
multi-seed runs, origin-level aggregation, and temporally blocked uncertainty
intervals. A single visualized origin must never select a method, and absolute
scores must not be compared across entity types with different scales.
