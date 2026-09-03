# Simcast implementation record

This file records observed upstream behavior and deliberate adaptations made
while implementing the proof of concept. Core mathematical notation follows
the attached research specification.

| Specification assumption | Observed implementation | Chosen adaptation | Reason |
|---|---|---|---|
| Entity names identify assets | `OS Apeldoorn` and `RS Roodwilligen (Duiven)` occur in more than one Liander category | Use `group_name::name` as the canonical entity ID | Prevent cross-category identity collisions |
| Versioned forecasts extend seven days | Revision `dce7fe9` contains issue lags zero through five days | Select only available vintages and derive eligible origins | Never fabricate unavailable horizons |
| Historical weather has availability metadata | `weather_measurements` has a timestamp index but no `available_at` | Treat measurement time as availability time and require `timestamp <= origin_timestamp` | Explicit, conservative point-in-time rule |
| Parquet schemas are clean columns | Historical weather stores timestamp as an index; versioned weather includes `__index_level_0__` | Normalize timestamp and discard the accidental index column on read | Stable internal schema |
| Bare `main` is reproducible | Chronos and Liander upstream repositories move | Pin exact source, model, and dataset revisions | Reproducible runs |
| Public Chronos embeddings are forecast representations | `Chronos2Pipeline.embed()` returns context-side encoder states | Patch `Chronos2Output` to expose the existing `forecast_embeds` head input | The dependence adapter needs direct output-patch context |
| A fractional split alone prevents leakage | Overlapping horizons can share realizations when stride is shorter than horizon | Purge earlier boundary origins whose future windows overlap the next partition | Test labels cannot become training targets |
| One cache can safely expose every label | Test labels in a general training cache are easy to consume accidentally | Store marginals for all origins but seal test truth/PIT until final evaluation | Enforce the experimental protocol in the API |
| Jitter preserves a correlation matrix | Adding diagonal jitter makes the diagonal exceed one | Renormalize after every jitter operation | Preserve standard-normal marginals |

## Pinned external revisions

- Liander2024: `dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044`
- Chronos source: `8589d1988e9676817548e9626738ff06b6ca6370`
- Chronos-2 model: `29ec3766d36d6f73f0696f85560a422f50e8498c`

## Initial protocol

- Static group: all 15 `transformer` targets, in YAML order.
- Origin timestamp: 23:45 UTC.
- Lookback/horizon/stride: 672/96/96 steps.
- Split: first 80% tune, last 20% test; last 20% of tune validates.
- Default PIT policy: no quantile repair; a crossing invalidates the complete
  spatial vector at that origin and lead.
- Core comparison: M0 through M3. M4 is a bounded smoke experiment only.

## Milestone status

- [x] M1: typed configuration, pinned downloader, entity metadata/grouping,
  availability-safe windows, purged chronological splits, and tests.
- [x] M2: idempotent pinned Chronos setup, minimal forecast-embedding patch,
  frozen direct-prediction wrapper, native quantiles, and regression tests.
- [x] M3: labeled and test-gated Zarr PIT library, discretized PIT construction,
  crossing handling, and tune-only marginal diagnostics.
- [x] M4: independent and Ledoit-Wolf static copulas, common marginal
  projection, spatial scenarios, aggregate metrics, and smoke pipeline.
- [x] M5: deterministic FM feature builder, conditional low-rank copula,
  Cholesky pseudo-likelihood, local trainer, and portable checkpoints.
- [x] M6: set-aware position-free transformer, padding/subset training,
  permutation equivariance, and variable-cardinality tests.
- [x] M7: one-time final evaluation, joint and aggregate comparisons,
  variable-cardinality diagnostics, paper figures, scientific summary,
  high-level runner, and reproduction command.
- [x] Optional M4 method: bounded conditional RBF-kernel smoke implementation.
