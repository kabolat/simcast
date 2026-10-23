# Data Adapter Interface

The adapter layer translates a source into an ordered entity group, UTC
measurements, point-in-time covariates, candidate coverage, and provenance. The
cache builder, forecast model, PIT construction, dependence models, and
evaluation use the shared adapter protocol rather than branching on source
identity.

An adapter is responsible for:

- Ordered canonical entity IDs and per-entity static metadata, including
  optional coordinates.
- UTC target and covariate frames, candidate coverage, and the configured
  target column. `coverage()` returns the candidate-origin span appropriate to
  `protocol.cardinality_mode`: the intersection of entity target lifetimes in
  `fixed` mode and their envelope (earliest start to latest end) in `variable`
  mode. It must not encode dataset-specific eligibility; per-origin entity
  availability belongs in `prepare_window`.
- Past-target masking and historical-covariate availability.
- Future covariates available at each forecast origin, with incomplete
  windows marked ineligible.
- Provenance for source revisions and non-obvious availability assumptions.

Lookback, horizon, origin time and stride, frequency, and chronological split
remain shared configuration and window/split machinery. The canonical entity
group is always retained. Fixed mode rejects an origin when any required input
is unavailable; variable mode marks that entity non-forecastable and can
forecast the remaining members without changing the cache entity axis.

## Implementing an adapter

1. Add a module under `src/simcast/data/` that implements `load_adapter`,
   `allow_patterns`, `validate_data_config`, and `default_cache_name`. Return
   an object satisfying `DatasetAdapter`: ordered `EntityGroup`, provenance,
   `coverage()`, and `prepare_window(window)`.
2. Reuse `FrameDatasetAdapter` when the source can be represented as per-entity
   UTC target, past-weather, and versioned future-weather frames. Otherwise,
   implement the narrow protocol directly and raise `IneligibleWindow` when
   required information is unavailable.
3. Add one registry entry in `simcast/data/registry.py`, a base YAML with an
   explicit adapter ID and ordered entity IDs, and a selective-download test.
4. Test timestamp parsing, ordering, point-in-time availability, both fixed and
   variable forecastability behavior, window sizes, split behavior,
   cache-fingerprint sensitivity, and a mocked cache build. Pin pre-change
   behavior for existing adapters.
5. Record source revisions and scientific assumptions in cache provenance and
   the corresponding [dataset reference](../datasets/README.md).

Dataset-specific scientific settings belong in the base YAML. Put settings
that affect parsed rows or availability in `data.options`; they are part of
the marginal-cache fingerprint. Every accepted key, type, range, fixed value,
and default for `data.options` must be documented in the reference page for
that adapter. Do not add source conditionals to cache, fit, or evaluation
code.
