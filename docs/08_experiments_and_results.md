# Experimental protocol and results

## Protocol snapshot

The completed real-data study evaluates all five Liander entity types with the
same pinned data and Chronos revisions. Each group uses:

- 15-minute resolution, seven-day lookback, one-day horizon;
- daily 23:45 UTC origins;
- 347 retained origins: 222 train, 55 validation, 70 test;
- 21 native Chronos quantile levels and six 16-step output patches;
- 4,096 aggregate scenarios per valid test origin/lead;
- 512 scenarios for joint scores;
- one global seed, 42;
- complete-vector missingness/crossing gates;
- M0--M3 as core methods and M4 as a bounded smoke result.

There are $70\times96=6,720$ possible test origin-lead cases per group. The
valid counts differ because all entities must have finite truth and valid
marginals. Solar uses predeclared isotonic repair in its supplied named config;
all other groups use raw noncrossing marginals.

## Main result

The table ranks only core methods M0--M3 by mean aggregate pinball loss.
`Change vs M0` is

$$
100\frac{S_{\mathrm{best}}-S_{\mathrm{M0}}}{S_{\mathrm{M0}}},
$$

so a negative number is an improvement.

| Entity type | $K$ | Mean absolute off-diagonal training PIT correlation | Raw crossing rate | Valid test cases | Best core method | Change vs M0 | Best-method 90% coverage |
|---|---:|---:|---:|---:|---|---:|---:|
| Transformer | 15 | 0.0760 | 0.764% | 6,439 | M3 set-aware | -1.278% | 0.8980 |
| Solar park | 5 | 0.0794 | 25.290% | 6,717 | M1 static | -0.635% | 0.8721 |
| Wind park | 5 | 0.5466 | 0.160% | 6,664 | M1 static | -8.457% | 0.8957 |
| MV feeder | 15 | 0.0902 | 0.469% | 6,347 | M1 static | -14.713% | 0.8907 |
| Station installation | 15 | 0.1468 | 0.769% | 6,427 | M2 conditional low-rank | -9.222% | 0.9192 |

Residual dependence is present in every group and is especially strong for
wind. A dependence model improves mean pinball over M0 for every entity type,
but increased neural complexity does not win uniformly: M1 wins three groups,
M2 one, and M3 one.

## Complete mean-pinball and calibration comparison

| Entity type | Method | Mean pinball | Change vs M0 | 90% coverage | WIS |
|---|---|---:|---:|---:|---:|
| Transformer | M0 independent | 3,945,888 | 0.000% | 0.9259 | 30,690,242 |
|  | M1 static | 3,943,344.75 | -0.064% | 0.9135 | 30,670,464 |
|  | M2 conditional low-rank | 3,920,982.75 | -0.631% | 0.8950 | 30,496,534 |
|  | **M3 set-aware** | **3,895,459** | **-1.278%** | **0.8980** | **30,298,014** |
|  | M4 bounded kernel | 4,416,806.5 | +11.934% | 0.9929 | 34,352,936 |
| Solar park | M0 independent | 0.0197566 | 0.000% | 0.8437 | 0.153663 |
|  | **M1 static** | **0.0196311** | **-0.635%** | **0.8721** | **0.152686** |
|  | M2 conditional low-rank | 0.0202271 | +2.382% | 0.8958 | 0.157322 |
|  | M3 set-aware | 0.0200586 | +1.529% | 0.8682 | 0.156012 |
|  | M4 bounded kernel | 0.0203446 | +2.976% | 0.8931 | 0.158236 |
| Wind park | M0 independent | 0.225179 | 0.000% | 0.6955 | 1.751392 |
|  | **M1 static** | **0.206136** | **-8.457%** | **0.8957** | **1.603282** |
|  | M2 conditional low-rank | 0.206677 | -8.217% | 0.8849 | 1.607485 |
|  | M3 set-aware | 0.206819 | -8.154% | 0.8869 | 1.608588 |
|  | M4 bounded kernel | 0.206423 | -8.330% | 0.9263 | 1.605510 |
| MV feeder | M0 independent | 459,931.69 | 0.000% | 0.6337 | 3,577,246.75 |
|  | **M1 static** | **392,263.44** | **-14.713%** | **0.8907** | **3,050,938** |
|  | M2 conditional low-rank | 398,596.69 | -13.336% | 0.9200 | 3,100,196.5 |
|  | M3 set-aware | 397,109.53 | -13.659% | 0.9379 | 3,088,630 |
|  | M4 bounded kernel | 396,489.19 | -13.794% | 0.9540 | 3,083,804.75 |
| Station installation | M0 independent | 5,861,676.5 | 0.000% | 0.7187 | 45,590,820 |
|  | M1 static | 5,348,282 | -8.758% | 0.8838 | 41,597,748 |
|  | **M2 conditional low-rank** | **5,321,089.5** | **-9.222%** | **0.9192** | **41,386,252** |
|  | M3 set-aware | 5,371,470 | -8.363% | 0.9046 | 41,778,100 |
|  | M4 bounded kernel | 5,373,569 | -8.327% | 0.9389 | 41,794,432 |

Bold marks the lowest core mean pinball, not necessarily the lowest value in
every metric column. M4 is displayed for transparency but excluded from model
selection because it received at most 32 origins and five epochs.

## Full M4 transformer follow-up

A subsequent transformer-only experiment removed the M4 smoke restriction. It
used all 222 training origins (18,088 complete origin-lead vectors), all 55
validation origins (5,206 complete vectors), the same maximum 100 epochs and
patience 12 as M2/M3, default subset training, and seed 42. Early stopping
completed after 13 epochs and selected epoch 1 at validation pseudo-NLL
-1.473613. No Chronos cache was rebuilt.

The test evaluation reused the original M0--M3 checkpoints and evaluated all
methods together on the same 6,439 valid transformer cases. It was regenerated
on CPU, so the common Monte Carlo draws produce very small numerical changes in
the M0--M3 values relative to the earlier GPU-generated table.

| Method | Mean pinball | CRPS | WIS | 90% coverage | 90% width | Energy Score |
|---|---:|---:|---:|---:|---:|---:|
| M0 independent | 3,946,312.5 | 9,757,837 | 30,693,540 | 0.9258 | 62,003,436 | 12,782,282 |
| M1 static | 3,944,830.25 | 9,760,598 | 30,682,010 | 0.9144 | 59,698,028 | 12,714,452 |
| M2 conditional low-rank | 3,920,756 | 9,727,597 | 30,494,772 | 0.8942 | 55,402,296 | 12,718,841 |
| **M3 set-aware** | **3,894,989.75** | **9,703,453** | **30,294,368** | 0.8969 | 55,244,292 | **12,701,536** |
| M4 full conditional kernel | 4,289,735.5 | 10,251,721 | 33,364,610 | 0.9854 | 88,623,736 | 12,754,201 |

Full training improves M4 mean pinball by 2.877% relative to its bounded smoke
checkpoint, but M4 remains 8.702% worse than M0 and 10.135% worse than M3. Its
90% coverage of 0.9854 comes with a much larger interval width, indicating
overdispersion rather than superior calibration/sharpness balance.

The fitted M4 test correlations are nonnegative by construction: their mean
off-diagonal value is 0.1600, ranging approximately from 0 to 0.9908. By
comparison, the transformer PIT dependence contains both signs and the core
models can represent negative correlations. This provides a plausible
model-based explanation for M4's overly wide full-group aggregate forecast,
but it is an inference from the fitted matrices rather than a causal diagnosis.
The best checkpoint's learned RBF length scale is 1.0143, close to its initial
value of one.

At the exploratory prefix-cardinality diagnostic, M4 mean pinball is 2,847,024
for $K=3$, 2,993,279 for $K=7$, and 4,289,735.5 for $K=15$. It is best among
the five methods at $K=7$ in that 1,024-scenario prefix calculation, but the
diagnostic uses only one deterministic entity prefix and is not evidence of a
general cardinality advantage.

The full checkpoint is stored in
`runs/real_liander2024_transformer_m4_full/`; its comparison is in
`runs/real_liander2024_transformer_m4_full_evaluation/`. These generated paths
are local experiment artifacts and may be excluded from version control.

## Interpretation by group

### Transformer

Residual PIT correlation is modest. M1 barely changes mean pinball (-0.064%),
while conditional models improve progressively and M3 reaches -1.278% with
90% coverage 0.898. This is the one group supporting the hypothesis that
set-level context adds value beyond entity-wise conditional factors. The gain
is small enough that paired uncertainty analysis is necessary before claiming
a reliable effect.

### Solar park

The primary scientific complication is marginal crossing, not dependence.
Isotonic repair makes the comparison executable while preserving raw crossing
diagnostics. M1 improves mean pinball slightly, whereas M2/M3 are worse than M0
on that score. The neural methods' wider/altered intervals can move coverage
closer to nominal without producing a better proper-score result. This warns
against using coverage alone as the selection criterion.

### Wind park

Mean absolute PIT correlation is 0.5466, far larger than any other group.
Independence severely undercovers at 90% (0.6955). All dependence methods
recover most of the aggregate improvement; the simple static M1 is marginally
best. The conditional methods' differences from M1 are tiny relative to the
large M0-to-M1 step, suggesting that estimating dependence is essential here
but context variation has not demonstrated added value.

### MV feeder

M0 has the strongest aggregate undercoverage (0.6337). M1 reduces mean pinball
by 14.713%, the largest relative improvement in the study, and brings coverage
to 0.8907. Neural methods produce wider intervals and coverage above nominal
but slightly worse pinball/WIS than M1. A low mean absolute correlation can
still matter materially for a 15-component sum; average absolute correlation
alone is not a sufficient predictor of aggregate impact.

### Station installation

All dependence methods improve substantially over independence. M2 is best on
mean pinball and WIS, providing the clearest positive evidence for conditional
dependence in this study. M3 is worse than M2, so set-aware complexity is not
uniformly beneficial.

## What the study supports

The evidence supports these descriptive conclusions:

- ignoring cross-entity forecast-error dependence can materially distort
  spatial aggregate uncertainty;
- a static shrinkage copula is a strong baseline and often sufficient;
- conditional low-rank modelling can help for some entity types;
- set-aware attention is promising for transformers but not a universal
  improvement;
- the value of dependence modelling is heterogeneous across physical groups;
- marginal validity, especially quantile crossings, can dominate feasibility.

It does **not** support claims that M3 is generally superior, that M4 has been
fairly compared, that results transfer beyond the selected year/groups, or that
the differences are statistically significant.

## Threats to validity

1. **Single seed.** Neural optimization and subset sampling are observed once.
2. **Single chronological test block.** Seasonality and regime changes may make
   this block unrepresentative.
3. **No uncertainty on score differences.** Cases within an origin and nearby
   days are dependent; naive IID standard errors would be inappropriate.
4. **Discrete truncated marginals.** No interpolation or tail extrapolation is
   used, limiting achievable calibration and scenario resolution.
5. **Gaussian copula.** Tail dependence and asymmetric co-errors are excluded.
6. **Complete-case selection.** Missing/crossing cases may be systematically
   harder than retained cases.
7. **Solar intervention differs.** Solar uses isotonic repair; cross-type score
   comparisons are inappropriate regardless, but protocol heterogeneity should
   remain visible.
8. **Variable-K prefixes.** Cardinality and entity composition are confounded.
9. **M4 resource inequality.** Its results are diagnostic only.
10. **Test reuse risk.** The repository enforces access separation, not a
    one-time cryptographic seal; repeated researcher inspection can overfit.

## Recommended confirmatory experiment

Predeclare mean aggregate pinball as primary, M1 as the main baseline, and
M2/M3 as paired alternatives. Repeat M2/M3 over multiple seeds. Estimate
confidence intervals for per-origin average score differences using a moving
block bootstrap over chronological origins, with block length justified by
dependence diagnostics. Report effect sizes, intervals, calibration/width, and
joint scores. Treat entity type as a stratification factor rather than pooling
absolute scores. Run sensitivity analyses for PIT repair, pooled-vs-lead M1,
factor rank, and several random entity subsets.

## Source artifacts

The compact generated summary is in `runs/all_entity_types_summary.csv` and its
narrative companion `runs/all_entity_types_summary.md`. Full metrics,
lead-specific tables, variable-cardinality tables, correlations, and figures
are under each `runs/real_liander2024_*_m0_m4/evaluation/` directory. These run
directories are generated artifacts and may be excluded from version control;
the table above preserves the headline scientific record in tracked docs.
