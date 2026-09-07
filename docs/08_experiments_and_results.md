# Experiments and results

## Full-group baseline status

### Input-protocol revision status

The saved full-group runs in this chapter predate the later target/covariate
input correction: they used the former solar/wind target-delay rule and
day-of-year calendar channels. Current code instead treats every target as
available when measured, replaces day-of-year with `is_weekend`, and offers an
optional future-weather oracle. At the requested scope, these experiments were
not rerun. Their tables therefore remain provenance for the earlier
full-group/subset-corrected protocol, not headline evidence for the revised
input protocol.

The full-group baseline uses one complete, ordered, static physical
entity group per Liander entity type. Every training, validation, and test case
for group $g$ has entity dimension $K_g=|\mathcal E_g|$. If any entity is
invalid at a forecast instance and lead, the complete $(i,\tau)$ case is
excluded. The entity set is never reduced.

The corrected full-group M0--M3 results are generated in new
`runs/full_group_*_m0_m3/` directories. They must not overwrite the legacy
artifacts described below. The tracked tables in this chapter are updated only
from runs whose manifests state all of the following:

- `full_group_only: true`;
- `subset_training: false`;
- `entity_selection_augmentation_enabled: false`;
- the complete ordered entity ID list and its exact $K_g$.

## Audit of earlier results

The earlier five-group runs under `runs/real_liander2024_*_m0_m4/` inherited
`subset_training.enabled: true` from the old base configuration. Consequently:

| Method | Actual earlier training protocol | Study status |
|---|---|---|
| M0 independent | no learned entity-selection step; always evaluated on the complete group | scientifically unaffected, but old metadata is not protocol-clean |
| M1 static | fitted from complete PIT vectors; always evaluated on the complete group | scientifically unaffected, but old metadata is not protocol-clean |
| M2 conditional low-rank | random-cardinality entity subsets were sampled during training | legacy/exploratory only |
| M3 set-aware | random-cardinality entity subsets were sampled during training | legacy/exploratory only |
| M4 bounded smoke | its method-specific configuration disabled subset training | optional diagnostic only |

The later transformer-only full-duration M4 run under
`runs/real_liander2024_transformer_m4_full/` also inherited random subset
training. It is therefore a legacy exploratory run despite using the full
training-origin budget. Its prefix-cardinality evaluation is not a full-group
result and is not reproduced here.

No numerical table from those neural runs is retained as a headline
full-group result. Their directories may remain available for provenance, but
their values must not be mixed with the corrected full-group experiment.
The per-run resolved-configuration evidence is summarized in
`results/legacy_experiment_audit.md`.

## Corrected protocol

All groups use the frozen Chronos-2 cache, 15-minute resolution, a seven-day
lookback, a one-day horizon, daily 23:45 UTC forecast origins, 21 native
quantile levels, 4,096 aggregate scenarios per valid test case, and a selected
512-member joint ensemble for joint scores. Solar uses its predeclared
deterministic isotonic repair. The resulting FM-derived marginal quantile grid
is fixed across M0--M3.

The core comparison changes only the Gaussian-copula dependence model:

- M0: independent Gaussian copula;
- M1: static lead-specific Gaussian copula;
- M2: conditional low-rank Gaussian copula;
- M3: set-aware conditional low-rank Gaussian copula.

M4 remains optional and diagnostic. No random subsets, prefixes,
reduced-cardinality evaluations, or entity-selection augmentations belong to
this experiment.

Energy Score is evaluated with the empirical all-pairs estimator on the
selected 512-member joint ensemble. Selecting those 512 members is ensemble
subsampling; chunked `torch.cdist` changes only memory use, not the estimator.

## Corrected full-group results

All five M0--M3 experiments completed under the protocol above. The primary
ranking uses mean aggregate pinball loss; lower is better. `Change vs M0` is

$$
100\frac{S_{\mathrm{best}}-S_{\mathrm{M0}}}{S_{\mathrm{M0}}},
$$

so a negative value is an improvement. These are descriptive, single-seed
results.

| Entity group | $K_g$ | Mean absolute off-diagonal training PIT correlation | Valid test cases | Best method | Change vs M0 | Best-method 90% coverage |
|---|---:|---:|---:|---|---:|---:|
| Transformer | 15 | 0.0760 | 6,439 | M2 conditional low-rank | -1.511% | 0.9219 |
| Solar park | 5 | 0.0794 | 6,717 | M1 static | -0.633% | 0.8733 |
| Wind park | 5 | 0.5466 | 6,664 | M1 static | -8.456% | 0.8950 |
| MV feeder | 15 | 0.0902 | 6,347 | M1 static | -14.715% | 0.8927 |
| Station installation | 15 | 0.1468 | 6,427 | M2 conditional low-rank | -8.873% | 0.9115 |

Dependence modelling improves mean pinball over independence for every group,
but its benefit and the preferred model are heterogeneous. M1 wins three
groups and M2 wins two. M3 does not win any group in this corrected single-seed
experiment. In particular, the former transformer M3 headline does not survive
removal of subset training.

### Complete metric table

| Entity group | Method | Mean pinball | WIS | 90% coverage | 90% width | Energy Score |
|---|---|---:|---:|---:|---:|---:|
| Transformer | M0 independent | 3,946,312.5 | 30,693,540 | 0.9258 | 62,003,436 | 12,806,960 |
|  | M1 static | 3,944,830.25 | 30,682,010 | 0.9144 | 59,698,028 | 12,739,834 |
|  | **M2 conditional low-rank** | **3,886,698** | **30,229,870** | **0.9219** | **59,714,272** | **12,725,114** |
|  | M3 set-aware | 3,904,673 | 30,369,680 | 0.9230 | 59,424,064 | 12,730,929 |
| Solar park | M0 independent | 0.019742087 | 0.15354955 | 0.8446 | 0.25837108 | 0.037339915 |
|  | **M1 static** | **0.019617192** | **0.15257819** | **0.8733** | **0.30911481** | **0.037203334** |
|  | M2 conditional low-rank | 0.020104649 | 0.15636951 | 0.8921 | 0.35972953 | 0.037245993 |
|  | M3 set-aware | 0.020085147 | 0.15621781 | 0.8860 | 0.36205295 | 0.037256483 |
| Wind park | M0 independent | 0.22512449 | 1.7509683 | 0.6955 | 1.6298919 | 0.32283098 |
|  | **M1 static** | **0.20608784** | **1.6029054** | **0.8950** | **2.5734615** | **0.32034478** |
|  | M2 conditional low-rank | 0.20611842 | 1.6031435 | 0.8969 | 2.6567435 | 0.32047984 |
|  | M3 set-aware | 0.20619807 | 1.6037629 | 0.9034 | 2.7380993 | 0.32067937 |
| MV feeder | M0 independent | 459,843.594 | 3,576,561.5 | 0.6342 | 2,713,431.25 | 527,885.562 |
|  | **M1 static** | **392,177.125** | **3,050,266.75** | **0.8927** | **5,173,175.5** | **522,559.031** |
|  | M2 conditional low-rank | 396,026.625 | 3,080,207.25 | 0.9450 | 6,425,532 | 521,631.375 |
|  | M3 set-aware | 397,109.031 | 3,088,625.75 | 0.9464 | 6,508,360 | 521,691 |
| Station installation | M0 independent | 5,860,416.5 | 45,581,016 | 0.7192 | 47,225,272 | 9,127,623 |
|  | M1 static | 5,348,543 | 41,599,780 | 0.8827 | 71,646,592 | 9,061,769 |
|  | **M2 conditional low-rank** | **5,340,423** | **41,536,624** | **0.9115** | **80,262,528** | **9,050,469** |
|  | M3 set-aware | 5,370,834.5 | 41,773,160 | 0.8859 | 73,881,952 | 9,052,517 |

Bold identifies the lowest mean pinball within a group. It does not assert
that the same method is optimal for every secondary metric.

## Interpretation by group

### Transformer

Residual training PIT dependence is modest. M1 changes mean pinball by only
-0.038%, whereas full-group M2 improves it by 1.511% relative to M0. M3 also
improves on M0 but is 17,975 mean-pinball units worse than M2. This reverses the
legacy, subset-trained transformer ordering and removes the former evidence
for an M3 headline win. A paired uncertainty analysis is still needed before
calling the M2 improvement reliable.

### Solar park

The main complication is the marginal intervention: substantial raw Chronos
quantile crossing required deterministic isotonic repair. That repaired grid
was fixed across all four methods. M1 improves slightly on M0; M2 and M3 have
coverage closer to 0.9 but worse pinball and WIS. Coverage alone therefore
does not justify preferring a conditional model.

### Wind park

Wind has the strongest residual PIT dependence. M0 severely undercovers at the
90% level (0.6955), while M1 reaches 0.8950 and improves mean pinball by 8.456%.
M2 and M3 are extremely close to M1 but do not improve the primary score. The
large gain comes from modelling dependence, not from conditional complexity.

### MV feeder

M0 has the strongest undercoverage (0.6342). M1 improves mean pinball by
14.715% and reaches 0.8927 coverage. M2 and M3 create wider intervals and
overcoverage while scoring slightly worse on mean pinball and WIS. A modest
average pairwise correlation can still have a large effect on a 15-entity sum.

### Station installation

All dependence models materially improve over M0. M2 is best on mean pinball,
WIS, and Energy Score, although its 90% interval is the widest and covers at
0.9115. M3 is worse than both M1 and M2 on mean pinball, so set-aware attention
is not uniformly beneficial.

## Interpretation rules

The primary scientific question is whether modelling cross-entity dependence
improves the aggregate predictive distribution while holding every
FM-derived marginal quantile grid fixed. Results are interpreted separately by
physical entity group because units, scales, marginal repairs, and dependence
strength differ.

The study does not claim novelty for joint probabilistic energy forecasting in
general. Nor does a single-seed comparison establish statistical superiority.
Any observed ranking is descriptive until paired uncertainty analysis over
chronological forecast origins is performed.

The central cross-group question is:

> How heterogeneous is the benefit of dependence modelling across physical
> entity groups?

## Threats to validity retained for the corrected study

1. Neural optimization is observed for a single seed.
2. The evaluation uses one chronological test block.
3. Nearby origins are dependent, so naive IID uncertainty intervals are not
   appropriate.
4. The finite native-quantile construction is a discrete approximation with
   no interpolated predictive CDF or tail extrapolation.
5. Gaussian copulas exclude asymmetric and non-Gaussian tail dependence.
6. Complete-case selection may exclude systematically difficult cases.
7. Solar receives deterministic isotonic repair whereas the other groups use
   raw Chronos-2 native quantiles.
8. Repeated inspection of the test block can introduce researcher overfitting.

## Reproducible result source

The machine-readable tracked table is
`results/full_group_metrics.csv`; its generated narrative and complete
ordered entity lists are in `results/full_group_summary.md`. Regenerate both
with:

```bash
uv run python scripts/summarize_full_group_results.py
```

The summarizer refuses runs with incorrect protocol flags, inconsistent group
metadata, missing methods, method-specific valid-case counts, or any
reduced-cardinality output. Full artifacts remain under
`runs/full_group_*_m0_m3/` and are intentionally separate from legacy runs.
