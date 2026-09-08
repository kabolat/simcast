# Scientific method-notebook companion

The notebooks in `notebooks/05` through `09` are not an alternative
implementation of Simcast. They are a transparent reading of the same cached
scientific objects used by the command-line experiments. A selected YAML
configuration determines the static group $\mathcal E_g$, its ordering, the
frozen Chronos marginal grid, and the cache identity.

Each notebook makes the following chain explicit:

$$
\{(\hat y_{k,\tau,q_j}^{(i)},q_j),y_{k,\tau}^{(i)}\}
\longrightarrow u_{k,\tau}^{(i)}
\longrightarrow z_{k,\tau}^{(i)}
\longrightarrow R_{g,\tau}^{(i)}
\longrightarrow \widetilde A_{g,\tau}^{(i,m)}.
$$

The methods differ only at the middle arrow, where they specify the same-lead
Gaussian-copula correlation matrix. The notebooks preserve the complete ordered
group: a missing entity invalidates an entire $(i,\tau)$ vector and never leads
to a smaller displayed or fitted group.

| Notebook | Scientific question | Correlation construction |
|---|---|---|
| `05_m0_independent_copula.ipynb` | What does aggregate uncertainty look like when residual ranks are independent? | $R=I_{K_g}$ |
| `06_m1_static_gaussian_copula.ipynb` | What persistent same-lead rank association is present in the training period? | lead-specific training PIT correlation |
| `07_m2_conditional_low_rank_copula.ipynb` | Can frozen per-entity forecast context predict dependence? | $\Lambda\Lambda^\mathsf T+\operatorname{diag}(\sigma^2)$ |
| `08_m3_set_aware_low_rank_copula.ipynb` | Does contextualizing the full group improve that conditional representation? | self-attention followed by the same low-rank form |
| `09_m4_conditional_kernel_diagnostic.ipynb` | What does a restricted smooth, non-negative kernel dependence family imply? | RBF Gram matrix, diagnostic only |

## Read-only by default

Every method notebook has `TRAIN_IF_MISSING = False` and
`RUN_EVALUATION = False`. In this state it only opens an existing cache and,
when available, an existing checkpoint/evaluation. It does not download data,
call Chronos, train a model, overwrite a run, or create a result artifact.

To turn a notebook into a self-contained didactic experiment, change the two
controls deliberately and set a new `OUTPUT_DIR`. Those cells call the same
public training and evaluation functions as the CLI; they do not reproduce
their mathematical steps with notebook-only code. The CLI remains the primary
route for registered paper experiments, while the notebooks are intended for
inspection, teaching, and interpretation.

## How to read results

The notebook plots are descriptive. For a formal paper comparison, use the
predeclared multi-seed runs, origin-level aggregation, and moving-block
bootstrap in the confirmatory report. In particular, do not choose a method
from one visualized origin, and do not compare absolute scores across entity
types with different scales.
