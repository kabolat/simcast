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
| `feederbw` | FeederBW v1.1, Zenodo record 19550073 | Complete configured substation group; supplied base has five feeders | 15-minute mean target; 23:45 UTC daily origins; 7-day lookback; 1-day horizon | Active power, selected historical weather, future-known calendar; no future weather |

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

### FeederBW v1.1

The base is [`configs/bases/feederbw/substation_52.yaml`](../../configs/bases/feederbw/substation_52.yaml),
with a small M0/M1/M2 lab composite at
[`configs/venues/lab/feederbw_smoke.yaml`](../../configs/venues/lab/feederbw_smoke.yaml).
It pins v1.1 of [Zenodo record 19550073](https://zenodo.org/records/19550073)
(DOI `10.5281/zenodo.19550073`, CC BY 4.0). The selected substation is 52, with
five ordered feeders: 060, 106, 109, 165, and 170. The metadata schema uses
`feeder` and `substation`; group membership is validated at load time.

The adapter reads `active_power_kW` and forms a left-labelled 15-minute mean
from one-minute observations. `minimum_samples_per_interval` is fingerprinted
in the base and may be 12 through 15; the supplied experiment requires all 15
samples. Negative readings are retained. Missing values remain missing, and
group eligibility always considers all five feeders together. The selected
target grid has 70,176 intervals with 67,796 complete for all feeders (96.61%)
at the supplied threshold. This corresponds to roughly 706 complete days
before window requirements.

The archive provides hourly historical weather, not forecast vintages. The
base therefore lists eight allowed fields under `covariates.past_weather`,
leaves `covariates.weather` empty, and sets
`future_weather_source: none`. The shared adapter independently assembles
past weather and future-known calendar inputs; no realized future weather is
passed to Chronos. Metadata contributes only the stable grouping and
diagnostics, never model covariates. The diagnostic command
`uv run feederbw-info --base configs/bases/feederbw/substation_52.yaml`
reports data coverage, per-feeder completeness, target range, weather window,
and latest-per-feeder metadata summaries.

By default, the download command obtains the shared metadata/weather assets
and only the ZIP batches needed by the base's feeder IDs. For substation 52,
the three required batches are 041--080, 081--120, and 161--200; only selected
feeder files are extracted. To populate the local data directory with every
feeder so other substation groups can be selected later, pass
`--all-feeders`; it downloads all five ZIPs (about 8.44 GB) and extracts all
200 measurement files. This does not change the configured experiment group:
cache construction still loads only the substation declared by the base.
To select another group, use `feeder_metadata.csv` to find all feeder IDs for
the desired `substation`, then update `data.options.substation_id`,
`protocol.entity_count`, and the ascending `protocol.ordered_entity_ids`
(`feederbw::feeder_NNN`). The adapter checks that the configured list exactly
matches the complete metadata group.
Asset sizes and Zenodo MD5 values are checked, downloads resume through
`curl`, and ZIP traversal/symlink entries are rejected. For example:

```bash
uv run download-data --base configs/bases/feederbw/substation_52.yaml
uv run download-data --base configs/bases/feederbw/substation_52.yaml --all-feeders
uv run feederbw-info --base configs/bases/feederbw/substation_52.yaml
uv run cache --base configs/bases/feederbw/substation_52.yaml
uv run composite --config configs/venues/lab/feederbw_smoke.yaml
```

The default selective download uses three measurement batches (about 5.05 GB)
and extracts only the five configured feeders. Full mode retrieves all five
batches (about 8.44 GB) once for later substation selection; Zenodo serves
each batch as a whole ZIP.

## Adding an adapter

1. Add a module under `src/simcast/data/` that implements `load_adapter(config)`, `allow_patterns(data)`, `validate_data_config(data)`, and `default_cache_name(data)`. Return an object satisfying `DatasetAdapter`: ordered `EntityGroup`, provenance, `coverage()`, and `prepare_window(window)`.
2. Reuse `FrameDatasetAdapter` when the dataset can be represented as per-entity UTC target, past-weather, and versioned future-weather frames. Otherwise implement the narrow protocol directly, raising `IneligibleWindow` when required information is unavailable.
3. Add one registry entry in `simcast/data/registry.py`, a base YAML with an explicit adapter ID and ordered entity IDs, and a selective download test.
4. Test timestamp parsing, ordering, point-in-time availability, full-group validity, window sizes, split behavior, cache fingerprint sensitivity, and a mocked cache build. Pin the pre-change behavior for any existing adapter.
5. Record source revisions and non-obvious scientific assumptions in both cache provenance and dataset documentation.

Dataset-specific scientific settings belong in the base YAML. Put settings that affect parsed rows or availability in `data.options`; they are part of the marginal-cache fingerprint. Do not add dataset conditionals to cache, fit, or evaluation code.
