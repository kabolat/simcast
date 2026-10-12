# Dataset adapters

The data layer converts a dataset into a fixed, ordered entity group and point-in-time windows. The cache builder, Chronos wrapper, PIT construction, dependence models, and evaluation do not branch on dataset identity. `data.adapter` selects the reader; its returned group, UTC coverage, prepared inputs, and provenance are consumed through the shared adapter protocol in `simcast/data/adapter.py`.

An adapter is responsible for:

- Ordered canonical entity IDs and per-entity static metadata, including optional coordinates.
- UTC target and covariate frames, common coverage, and the configured target column.
- Past target masking and historical covariate availability.
- Future covariates that were actually available at each origin, with incomplete windows marked ineligible.
- Provenance for the source revision and any dataset-specific assumptions.

Lookback, horizon, origin time/stride, frequency, and chronological split remain shared configuration and window/split machinery. The full entity group is retained: a missing entity or a required covariate gap never silently changes group cardinality.

## Supported datasets

| Adapter | Dataset | Group | Frequency and protocol | Inputs |
|---|---|---|---|---|
| `liander2024` (default) | OpenSTEF Liander2024 Energy Forecasting Benchmark | One configured physical group: transformer, solar park, wind park, MV feeder, or station installation | Base-specific; supplied configs use 15-minute data, daily origins, 7-day lookback, 1-day horizon | Load targets, measured historical weather, versioned future weather, optional locations |
| `gefcom2014_wind` | `rebase-energy/gefcom2014`, pinned in the base YAML | Ten wind zones, ordered `gefcom_wind::z1` through `gefcom_wind::z10` | Hourly; 00:00 UTC daily origins; 168-hour lookback; 24-hour horizon | Normalized power, four supplied NWP wind components, UTC hour features; no coordinates |

### Liander2024

The existing Liander base files and model inputs are unchanged. Its adapter reads the target metadata YAML and the selected group's load, measured-weather, and versioned-weather Parquet files. Forecast vintages use their supplied `available_at` values. The default adapter and empty options are omitted from the cache fingerprint for compatibility with caches made before adapters were introduced.

### GEFCom2014 Wind

The base is [`configs/bases/gefcom2014/wind.yaml`](../../configs/bases/gefcom2014/wind.yaml). It reads only `data/wind_targets.parquet` and `data/wind_nwp.parquet` from the pinned Hugging Face dataset revision. The targets are 8,784 hourly normalized-power observations for each of ten zones (2012-01-01 01:00 through 2013-01-01 00:00); timestamps are naive in the source and are interpreted as UTC. NWP fields are `u10`, `v10`, `u100`, and `v100`, separately for each zone. `issue_time` and `knowledge_time` in the repack are monthly competition task-release stamps, not physical issue times, and are ignored. The repack provides no dataset card or license metadata.

**NWP availability caveat.** The source repack does not document physical NWP issuance timestamps or actual availability times. The values show a daily stitched pattern consistent with runs covering valid hours `(S, S + 24 h]`. The adapter assigns such a run availability time `S`; the 00:00 UTC daily origin then uses the next 24 valid hours, 01:00 through the following 00:00. The boundary shape supports the 24-hour coverage window, but does not establish when the run was issued or became available. This is an explicit, fingerprinted modeling assumption in `data.options.nwp_run_start_hour_utc` and `data.options.nwp_run_length_hours`, not a claim established by source metadata. Origins at other phases may be ineligible when a needed run is not yet available. The adapter rejects `future_weather_source: oracle` because no realized weather series is supplied.

The split is chronological and uses the shared overlapping-horizon purge. Since daily origins are 24 hours apart and the horizon is 24 hours, these origins do not overlap. Missing weather or targets are never imputed to manufacture an eligible full-group origin. No coordinates are present, so methods that use locations must set `features.use_location: false`; the supplied lab composite does so for M2.

Targets are normalized into `[0, 1]`. The adapter passes them to Chronos without rescaling, and this integration does not clip Chronos quantiles to that interval. Inspect out-of-support forecasts in the first real cache before interpreting them as physically bounded power predictions.

Download, build the frozen marginal cache, then run the small declared lab study:

```bash
uv run download-data --base configs/bases/gefcom2014/wind.yaml
uv run cache --base configs/bases/gefcom2014/wind.yaml
uv run composite --config configs/venues/lab/gefcom_wind_smoke.yaml
```

The base uses the same Chronos-2 source/model revision path as Liander, discretized PITs, isotonic quantile repair, and the `training_frequency` dependence transform. The composite's M2 budget is only a smoke-test budget, not a scientific training recommendation. It runs actual Chronos inference unless a compatible cache has already been built.

## Adding an adapter

1. Add a module under `src/simcast/data/` that implements `load_adapter(config)`, `allow_patterns(data)`, `validate_data_config(data)`, and `default_cache_name(data)`. Return an object satisfying `DatasetAdapter`: ordered `EntityGroup`, provenance, `coverage()`, and `prepare_window(window)`.
2. Reuse `FrameDatasetAdapter` when the dataset can be represented as per-entity UTC target, past-weather, and versioned future-weather frames. Otherwise implement the narrow protocol directly, raising `IneligibleWindow` when required information is unavailable.
3. Add one registry entry in `simcast/data/registry.py`, a base YAML with an explicit adapter ID and ordered entity IDs, and a selective download test.
4. Test timestamp parsing, ordering, point-in-time availability, full-group validity, window sizes, split behavior, cache fingerprint sensitivity, and a mocked cache build. Pin the pre-change behavior for any existing adapter.
5. Record source revisions and non-obvious scientific assumptions in both cache provenance and dataset documentation.

Dataset-specific scientific settings belong in the base YAML. Put settings that affect parsed rows or availability in `data.options`; they are part of the marginal-cache fingerprint. Do not add dataset conditionals to cache, fit, or evaluation code.
