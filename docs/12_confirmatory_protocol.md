# Confirmatory experimental protocol

## 1. Scientific scope

The study asks whether explicitly modelling same-lead cross-entity residual
rank dependence improves probabilistic forecasts of the complete spatial
aggregate when FM-derived marginal quantile grids are fixed.

Chronos-2 remains frozen. The study does not fine-tune the foundation model,
model dependence between different leads, or alter physical group membership.
M0, M1, M2, M3, and M4 are all primary declared methods. Test observations
previously inspected during exploratory development are acknowledged; the
protocol is fixed from its declaration onward rather than claimed to be
historically untouched.

## 2. Frozen base design

Each group base records the exact ordered entity set $\mathcal E_g$ and
$K_g$. The shared design fixes:

- $\Delta=15$ minutes, $L=672$, and $H=96$;
- daily origins at 23:45 UTC;
- chronological train, validation, and test partitions with overlap purging;
- vintage weather available at $t^{(i)}$ and the documented calendar features;
- deterministic finite-cell pseudo-PITs without CDF interpolation;
- deterministic isotonic repair for solar before the marginal grid is fixed;
- 4,096 aggregate scenarios and a 512-member joint-score ensemble;
- evaluation seed 2027 and common random numbers;
- the complete-vector validity rule.

The base fingerprint makes this design machine-verifiable. Any change to
entity order, revisions, information set, forecast origin protocol, split,
Chronos, or PIT construction yields a different marginal identity.

## 3. Method design

M0 is identity correlation. M1 is the lead-specific Ledoit--Wolf estimate with
positive stabilization. M2 and M3 use rank four in the primary comparison. M4
uses a learned 16-dimensional RBF geometry. M2--M4 use AdamW, a maximum of 100
epochs, patience 12, and complete training/validation partitions.

The ten optimization seeds are 11, 23, 37, 42, 59, 71, 83, 97, 101, and 131.
`best.pt` is selected only by chronological validation pseudo-NLL. The best test
seed is never selected. M0/M1 are deterministic conditional on the cache.

These choices are encoded in separate method files and repeated explicitly in
the main composite. There is no method-specific truncation or opt-in path for
M4.

## 4. Primary and secondary estimands

For every complete valid $(i,\tau)$ case, aggregate pinball loss is averaged
over the reported aggregate quantiles. The origin-level score is

$$
S_{m,g}^{(i)}=
\frac{1}{|\mathcal V_{g}^{(i)}|}
\sum_{\tau\in\mathcal V_g^{(i)}}S_{m,g,\tau}^{(i)},
$$

where $\mathcal V_g^{(i)}$ is the set of valid leads. Secondary aggregate
estimands are CRPS, WIS, central interval coverage, width, and interval score.
Joint estimands are Energy and Variogram Scores. Gaussian-copula pseudo-NLL of
the observed finite-cell score vector is a direct dependence diagnostic.

Energy Score is the empirical all-pairs estimator on the selected 512-member
joint ensemble. Selecting those members is ensemble subsampling; chunked
distance computation is only a memory strategy.

## 5. Paired Monte Carlo design

The base-normal array is a deterministic function of evaluation seed and case
identity. Every method receives the same $\eta_{g,\tau}^{(i,m)}$ before applying
its own correlation factor. Scenario uniforms differ by method, but source
randomness is paired. The valid mask and fixed marginal projection are also
identical.

This design reduces simulation noise in paired score differences without
treating scenarios as independent empirical observations.

## 6. Temporal and optimization uncertainty

For methods $a$ and $b$,

$$d_{a,b,g}^{(i)}=S_{a,g}^{(i)}-S_{b,g}^{(i)}.$$

Negative values favor $a$. Conditional-method seeds are first averaged within
method, group, and origin. A non-circular moving-block bootstrap then resamples
contiguous origins. The primary block length is seven daily origins; lengths
three and fourteen are sensitivities; the final calculation uses 10,000
replicates.

This separates optimization variation across M2--M4 seeds from test-period
variation across origins. Leads and neural seeds are not counted as independent
observational replications.

## 7. Explicit sensitivity families

Separate composites declare:

- four feature profiles for each of M2, M3, and M4;
- nominal-cell versus training-frequency dependence scores;
- M2/M3 ranks 2, 4, and 8, with M4 retained as a non-rank reference;
- lead-specific versus lead-pooled M1.

Every variant has a distinct experiment ID and a role-valid override. No
command-line flag generates extra variants implicitly.

## 8. Reproduction and status

The main composite is executed by:

```bash
uv run python -m simcast.cli.run_composite \
  --config configs/venues/<venue>/main.yaml
```

The venue name is intentionally supplied through the path rather than embedded
in generic scientific documentation or Python symbols. Its run and report
trees retain the exact expansion, role hashes, environment, Git state, fitted
methods, evaluations, and completion status.

No empirical run or report is performed by the configuration migration itself.
Existing historical artifacts remain unchanged and may include earlier
exploratory designs. Only a newly completed current-schema composite supports
claims under this protocol.
