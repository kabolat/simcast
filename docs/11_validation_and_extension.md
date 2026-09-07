# Validation, limitations, and extension guide

## What the test suite establishes

The repository's 132 tests cover scientific contracts as well as ordinary
input validation. Major protected invariants include:

- deterministic UTC origin generation and exact past/future window endpoints;
- chronological split counts and horizon-overlap purging;
- target measurement-time availability and latest-available weather vintage selection;
- canonical group identity/order and static group membership;
- strict config inheritance, overrides, environment expansion, and validation;
- Chronos freezing, direct-capacity checks, target-row mapping, group-ID
  separation, quantile and forecast-embedding extraction;
- crossing diagnostics, isotonic repair, PIT cell semantics,
  Gaussianization, and complete-vector gating;
- labeled cache dimensions, float16 embeddings, atomic writes, and physical
  test-label separation;
- train-only feature standardization and exact patch-to-lead mapping;
- M0 identity and M1 shrinkage/full-group/save-load behavior;
- M2/M3/M4 positive-definite normalized matrices and shape contracts;
- M3 permutation equivariance while operating on the declared full group;
- pseudo-NLL numerical behavior and trainer checkpoint/history output;
- common discrete marginal projection and aggregate score formulas;
- final evaluation outputs, method comparisons, figures, and replay paths.

Passing tests establish consistency with implemented contracts. They do not
establish model adequacy, marginal calibration, statistical significance, or
generalization to unseen years and grids.

## Current scientific limitations

### Dependence family

A Gaussian copula represents dependence through linear correlation of latent
normal ranks. It is radially symmetric and cannot express asymmetric upper vs
lower tail association. Electricity errors may share weather-driven extremes,
so t-copulas, vine copulas, normalizing flows, or explicit tail models are
scientifically relevant alternatives. Any alternative must preserve the fixed
marginal experiment and be evaluated with the same cases and random-number
protocol.

### Same-lead scope

The modeled object is $R_{i,\tau}$, never a joint matrix across
$(k,\tau)$. The code cannot produce calibrated multistep trajectories for
ramping, storage, or path-dependent decisions. Extending to spatiotemporal
dependence changes matrix dimension from $K$ to $KH$ and changes the research
question; it should be treated as a new method family, not a small patch.

### Marginal discreteness and tails

Nearest-native projection guarantees fairness but truncates support to 21
forecast values. Extreme aggregates cannot exceed sums of native endpoint
quantiles. A future interpolation/tail method should first be validated at the
entity level and applied identically to every dependence model. Randomized PIT
would also change the training target and should be reported as an ablation.

### Missingness and crossing selection

Complete-vector gating provides a fixed group but may preferentially discard
difficult periods, especially without solar repair. Report both absolute and
per-lead retained counts. Potential extensions include likelihoods that handle
subvectors while retaining a declared full-group estimand, but those require
careful missing-at-random assumptions and common evaluation cases.

### Conditional-model identification

Low-rank factors are rotation-invariant; interpret $R$, not individual factor
coordinates. Location is standardized as raw latitude/longitude, not projected
physical distance. The networks may recognize entities indirectly through
stable location/load-scale features even without ID embeddings. Generalization
to truly new assets is not tested by the full-group experiment.

### Hyperparameters and selection

The completed study uses one general architecture and one seed. There is no
nested tuning protocol. Test results should not be used to repeatedly revise
architectures and then presented as untouched confirmatory evidence. A new
study should define validation-only selection rules before opening test labels.

## Safe procedure for adding a new dependence model

1. State the new statistical hypothesis and whether it remains same-lead.
2. Implement a module that maps `[batch,K,F]` to symmetric positive-definite
   unit-diagonal `[batch,K,K]` correlations.
3. Decide and test its permutation properties on the complete group.
4. Add strict config fields with scientifically meaningful defaults.
5. Add construction and checkpoint loading in training/checkpoint modules.
6. Reuse the existing feature builder, pseudo-NLL, valid cases, scenario
   projection, base normals, and evaluator unless the hypothesis explicitly
   requires changing them.
7. Add unit tests for PSD/Cholesky, gradients, save/load, permutations,
   full-group shape enforcement, and malformed inputs.
8. Add a synthetic end-to-end test without touching real test labels.
9. Predeclare whether the method is core, ablation, or smoke-only.
10. Update the mathematical and artifact documentation before running the
    final test evaluation.

Do not “fix” a difficult matrix by silently falling back to identity during
evaluation. A failure should remain observable and motivate a principled
parameterization or stabilization change.

## Adding an entity type or dataset

A new Liander entity type requires adding it to the typed `EntityType` and
group allowlist, providing a named config, and explicitly defining its target
availability rule. Verify that metadata IDs remain unique and that native
target scale/meaning is documented.

A new dataset requires a larger adapter boundary: timestamp normalization,
entity metadata, load/weather readers, availability semantics, and a downloader
or local-data contract. Reuse `ForecastWindow`, the Chronos input contract, and
the cache schema where possible. Never infer publication availability from row
presence alone if vintage or delay information exists.

Before cross-dataset comparison, establish target units and normalization.
Absolute proper scores are scale-dependent.

## Recommended research checks for a new run

### Before Chronos inference

- summarize common coverage and source missingness;
- inspect the exact entity list/order and coordinates;
- verify origin phase and forecast timestamps manually for boundary cases;
- audit several target-timestamp and weather-vintage examples;
- record skipped origins and reasons.

### Before dependence training

- inspect native quantile crossing heatmaps by entity and lead;
- justify any monotonic repair before seeing dependence test results;
- assess tune-only marginal coverage, PIT histogram, tails, and median MAE;
- tabulate complete vector counts per lead and split;
- examine training PIT correlations/eigenvalues;
- verify no test labels are visible through training access.

### Before final evaluation

- fix the model list and checkpoint selection rule;
- confirm best checkpoints arose from validation only;
- record seed, hardware/software versions, Git state, and resolved config;
- decide the primary score and uncertainty procedure;
- verify all methods share fixed quantiles, valid mask, and base normals.

### After final evaluation

- report score differences per origin/lead, not only grand means;
- inspect calibration jointly with interval width and proper scores;
- examine whether gains concentrate at particular leads/seasons;
- report joint scores as well as the aggregate objective;
- preserve unfavorable and failed models, including bounded smoke results;
- distinguish exploratory interpretation from confirmatory conclusions.

## Suggested statistical uncertainty analysis

Define a per-origin score by averaging lead-level loss over valid leads, then a
paired difference $d_i=S_i^{(A)}-S_i^{(B)}$. Because consecutive daily origins
may remain serially dependent, use a moving-block or stationary bootstrap over
origins rather than treating all 6,000+ origin-lead cases as IID. Select block
length from an autocorrelation analysis or a predeclared sensitivity grid.
Report mean difference, relative difference, and a confidence interval.

For M2/M3, separate two sources of uncertainty:

- data/test-period uncertainty, assessed with paired temporal resampling;
- optimization uncertainty, assessed by independent training seeds.

Do not bootstrap Monte Carlo scenario noise as though it were data uncertainty.
Common random numbers already make method differences less sensitive to that
noise; scenario-count sensitivity can be evaluated separately.

## Known implementation semantics to preserve or revise explicitly

- `sampling.common_random_numbers` is currently descriptive; sampling is
  always common across methods.
- either joint-score flag enables both Energy and Variogram calculations.
- `evaluation.report_by_lead` does not currently suppress the lead table.
- `runtime.num_workers` is not passed to the current trainer DataLoaders.
- Full-group configurations reject subset training and variable-$K$ evaluation;
  architecture-level variable-size capability is not an experiment.
- M4 smoke mode uses only the first eligible complete origins within each
  partition; full mode uses the complete train and validation partitions.
- cache reuse is not automatically validated against a full config hash.

These are documented behavior, not necessarily ideal future APIs. If changed,
add regression tests, bump artifact schema where compatibility changes, and
record the scientific impact in the experiment protocol.

## Reproducibility levels

It is useful to distinguish:

1. **Computational replay:** same artifacts, code, config, and hardware/software
   reproduce effectively the same values.
2. **Independent reproduction:** a new environment downloads pinned inputs and
   regenerates caches/models/results.
3. **Scientific replication:** a new time period, dataset, or grid tests the
   same hypothesis.

The repository directly supports levels 1 and 2 through revisions, `uv.lock`,
manifests, and replay commands. Level 3 remains future scientific work.
