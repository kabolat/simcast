# PowerTech 2027 confirmatory protocol

## Status and scientific scope

This protocol freezes the next phase of the Simcast study. It asks:

> If frozen FM-derived entity marginals are held fixed, how much does explicitly
> modelling same-lead cross-entity forecast-error dependence improve
> probabilistic forecasts of the full spatial aggregate?

The test period was inspected during exploratory development. This is recorded
as a limitation: the protocol is confirmatory from its declaration onward, not
a claim that the test set was historically untouched. Hyperparameters and
checkpoint choices must not be revised in response to further test results.

Chronos-2 remains frozen. The experiment does not fine-tune the FM, model
dependence across forecast leads, or change the physical group. The four primary
methods are M0 independent, M1 lead-specific static Gaussian, M2 conditional
low-rank Gaussian, and M3 set-aware conditional low-rank Gaussian. M4 remains
diagnostic.

## Frozen configurations

The machine-validated group configurations are under `configs/powertech2027/`.
Each records the exact ordered entity IDs and $K_g$ in addition to inheriting
the pinned Liander and Chronos revisions. The common declaration fixes:

- $L=672$, $H=96$, and 15-minute resolution;
- one origin per day at 23:45 UTC;
- chronological train/validation/test splits with overlapping horizons purged;
- deterministic finite-cell PITs with no CDF interpolation;
- solar isotonic repair before all dependence methods;
- 4,096 aggregate scenarios and 512 selected joint-score scenarios;
- AdamW settings and validation pseudo-NLL checkpoint selection;
- evaluation seed 2027 and common random numbers;
- neural seeds 11, 23, 37, 42, 59, 71, 83, 97, 101, and 131;
- 10,000 moving-block bootstrap replicates with primary block length seven
  daily origins and sensitivity lengths three and fourteen;
- `full_group_only: true`, subset training disabled, and no variable-$K$ output.

Training and evaluation compare configured ordered IDs against the cache. A
confirmatory cache is also checked against the data, forecast, covariate,
Chronos, and marginal-PIT construction settings. This deliberately rejects old
caches containing day-of-year instead of the currently declared weekend
indicator.

## Primary and secondary estimands

For every valid complete $(i,\tau)$ case, aggregate pinball loss is first
averaged over the seven requested aggregate quantiles. The primary per-origin
score then averages those losses over valid leads:

$$
S_m^{(i)}=\frac{1}{|\mathcal V_i|}
\sum_{\tau\in\mathcal V_i}S_{m,\tau}^{(i)}.
$$

Secondary aggregate outputs are CRPS, WIS, central coverage, width, and interval
score. Joint outputs are Energy and Variogram Scores. The direct dependence
diagnostic is Gaussian-copula pseudo-NLL of the observed finite-cell PIT score
vector under the predicted correlation. It is called a pseudo-likelihood
because finite deterministic PITs need not be exactly uniform.

The evaluator writes one row per group, method, seed, origin, and lead, followed
by an origin-aggregated table. Invalidity of one entity invalidates the complete
group case; no entity is removed.

## Common random numbers

The base normal array is a deterministic function of the evaluation seed and
flattened `(origin, lead)` case index. Thus every Gaussian-copula method receives
the same $\eta^{(m)}$ for a fixed case, independent of batching or method order.
Each method applies its own Cholesky factor. Scenario uniforms therefore differ
while the source Monte Carlo draws are paired. Nearest-native-quantile
projection is identical for every method.

## Neural repetitions and checkpoint selection

M2 and M3 are trained for every declared optimization seed with identical
architecture and data. `best.pt` is selected solely by minimum validation
pseudo-NLL. Training curves, selected epoch, validation criterion, checkpoint
hash, Git SHA, config hash, revisions, ordered IDs, training seed, and evaluation
seed are stored. Reports retain individual seeds and summarize their mean,
standard deviation, minimum, and maximum. They never select the best test seed.

For the primary temporal interval, the per-origin score is averaged across
neural seeds before resampling origins. M0 and M1 are deterministic conditional
on the training cache.

## Paired moving-block bootstrap

For methods $a$ and $b$, chronological paired origin differences are

$$
d_{a,b}^{(i)}=S_a^{(i)}-S_b^{(i)}.
$$

Negative values favour method $a$. A non-circular moving-block bootstrap draws
contiguous blocks from the ordered origin sequence, concatenates enough blocks
to reach $N$, and truncates to $N$. The report gives the mean paired difference,
percentage difference relative to method $b$, and percentile 95% interval. No
IID standard errors or naive IID p-values are produced.

## Feature ablations

The runner implements four exact M2 feature sets. M3 runs the last two by
default and can run all four with `--all-m3-ablations`.

| Name | Forecast embedding | Quantile shape | Median | Log spread | Patch position | Location |
|---|---:|---:|---:|---:|---:|---:|
| embedding dynamic only | yes | no | no | no | yes | no |
| quantile dynamic only | no | yes | yes | yes | yes | no |
| combined dynamic | yes | yes | yes | yes | yes | no |
| full | yes | yes | yes | yes | yes | yes |

Patch position remains in every configuration because one Chronos forecast
embedding represents 16 output leads.

## PIT sensitivity

The primary PIT remains the nominal $Q+1$-cell midpoint construction. The
sensitivity estimates, using training origins only,

$$
\hat p_{k,\tau,c},\qquad
\tilde u_{k,\tau,c}=\sum_{r<c}\hat p_{k,\tau,r}
+\tfrac12\hat p_{k,\tau,c}.
$$

This map is frozen for validation and test and saved with the run. It changes
only dependence-training and dependence-diagnostic scores. It does not change
Chronos quantiles, interpolate a CDF, or change scenario marginal projection.

## Static and rank sensitivities

M1a is the primary lead-specific estimate. M1b pools complete training vectors
over all leads and is a sensitivity result. M2/M3 ranks 2, 4, and 8 use the same
training budget and seeds; rank 4 remains primary. Reports include validation
pseudo-NLL, test aggregate pinball, parameter count, and seed variability.

## Execution

Rebuild the cache and run the main family for each group. The output path must
be new because runs are never overwritten.

```bash
uv run python -m simcast.cli.run_powertech_experiments \
  --config configs/powertech2027/transformer.yaml \
  --phase main \
  --output-dir runs/powertech2027/transformer/main \
  --rebuild-cache
```

Replace `main` with `ablation`, `pit_sensitivity`, `rank_sensitivity`, or
`static_sensitivity` and use a distinct output directory. Repeat for
`solar_park`, `wind_park`, `mv_feeder`, and `station_installation`.

After all desired runs exist, build the report without retraining:

```bash
uv run python -m simcast.cli.build_powertech_report \
  --input-root runs/powertech2027 \
  --output-dir reports/powertech2027
```

The report contains tidy Parquet/CSV results, CSV and LaTeX manuscript tables,
vector and PNG figures, diagnostics, the frozen protocol, reproduction commands,
and an evidence-constrained scientific narrative.

## Current execution status

The infrastructure and synthetic report path are validated by the test suite.
No new confirmatory Liander results are claimed in the repository at this point.
The available historical caches predate the weekend-covariate declaration and
are intentionally rejected by the confirmatory runner. Fresh confirmatory
caches and all requested multi-seed runs must be produced on a suitable Chronos
inference/training machine before generating manuscript numbers.
