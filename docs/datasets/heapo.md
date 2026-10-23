# HEAPO v1: control-household weather-station cohorts

> The HEAPO cohort aggregate is not a measured feeder or transformer load.
> Households are grouped by nearest weather station as a coarse geographic cohort.

> `kWh_received_Total` represents active energy received from the grid during
> each 15-minute interval. It is not net grid exchange and does not include PV
> energy self-consumed behind the meter.

## Source, license, and verified release

- Dataset: [Zenodo record 15056919, v1](https://zenodo.org/records/15056919).
- Dataset DOI: [10.5281/zenodo.15056919](https://doi.org/10.5281/zenodo.15056919).
- Published paper: [10.1145/3679240.3734637](https://doi.org/10.1145/3679240.3734637).
  The [open manuscript](https://arxiv.org/abs/2503.16993) describes the measurements.
- Official loader: [tbrumue/heapo](https://github.com/tbrumue/heapo), inspected at
  revision `38768a8092552e9e0f0927a51f30ad31a27bc002`. Simcast does not import it.
- **Dataset license: CC-BY-4.0**, explicitly supplied in Zenodo record metadata
  as `license.id: cc-by-4.0`. This is distinct from the companion code's MIT license.
  Attribute the dataset authors and cite the dataset and published paper.
- Single archive `heapo_data.zip`: **458,317,017 bytes**, published MD5
  `2b92b0d52487e1a733c35ee59860450d`.
- Direct inspection found **5,252,838,790 uncompressed bytes**, 5,281 files
  (5,294 ZIP entries including directories), rooted at `heapo_data/`.

The published population is 1,408 households: 214 treatment and 1,194 control.
The actual quarter-hour files cover 1,407 households: 214 treatment and
**1,193 control**. The missing control quarter-hour file belongs to station `Hg`.
The paper's overall extent (approximately 2018-11-03 through 2024-03-21)
includes other resolutions/weather. Across the actual quarter-hour files, raw
timestamps span **2019-05-20 00:00 UTC through 2024-02-27 23:45 UTC**.
Do not substitute the overall published extent for target coverage.

## Actual archive layout and schemas

All source CSVs are semicolon-delimited. Timestamps in inspected files carry
explicit UTC offsets, for example `2022-05-07 00:00:00+00:00`; the paper also
declares UTC. Simcast normalizes offsets to UTC, never to local Zurich time.
Anonymized household IDs are positive integers; weather IDs are case-sensitive
opaque strings, not coordinates or ordinal measurements.

| Directory | Files |
|---|---:|
| `meta_data/` | 3 |
| `smart_meter_data/15min/` | 1,407 |
| `smart_meter_data/daily/` | 1,298 |
| `smart_meter_data/monthly/` | 1,264 |
| `smart_meter_data/cumulative_counters/` | 1,285 |
| `smart_meter_data/overview/` | 4 |
| `weather_data/hourly/` | 8 |
| `weather_data/daily/` | 8 |
| `weather_data/overview/` | 2 |
| `reports/` | 2 |

### Household and survey files

`meta_data/households.csv` has 1,408 rows and exactly these 11 columns:

```text
Household_ID, Group, Weather_ID, Installation_HasPVSystem,
Protocols_Available, Protocols_HasMultipleVisits, Protocols_ReportIDs,
MetaData_Available, SmartMeterData_Available_15min,
SmartMeterData_Available_Daily, SmartMeterData_Available_Monthly
```

`Group` is literally `control` or `treatment`, not a numeric code.
`Weather_ID` directly identifies the matching hourly weather filename.

`meta_data/meta_data.csv` has 1,358 rows and these 13 columns:

```text
Household_ID
Survey_Building_Type
Survey_Building_LivingArea
Survey_Building_Residents
Survey_HeatPump_Installation_Type
Survey_HeatDistribution_System_FloorHeating
Survey_HeatDistribution_System_Radiator
Survey_DHW_Production_ByHeatPump
Survey_DHW_Production_ByElectricWaterHeater
Survey_DHW_Production_BySolar
Survey_Installation_HasDryer
Survey_Installation_HasFreezer
Survey_Installation_HasElectricVehicle
```

`meta_data/meta_data_variables.csv` has 13 rows and columns
`VariableName;Description`. Survey metadata, missing values, variable
descriptions, and household grouping flags are retained for diagnostics and
provenance only. They are not Chronos or M2/M3 inputs. Inspection protocol
files are not extracted or used. There are no entity-ID embeddings.

### Quarter-hour files and overview

`smart_meter_data/overview/smart_meter_data_15min_overview.csv` has 1,407 rows:

```text
Household_ID
SMD_15min_TimeAvailable_EarliestTimestamp
SMD_15min_TimeAvailable_LatestTimestamp
SMD_15min_TimeAvailable_NumberDays
SMD_15min_TimeAvailable_DaysBeforeVisit
SMD_15min_TimeAvailable_DaysAfterVisit
SMD_15min_MeasurementsAvailable_Total
SMD_15min_MeasurementsAvailable_HeatPump
SMD_15min_MeasurementsAvailable_Other
```

Each `smart_meter_data/15min/<Household_ID>.csv` has these seven columns,
verified across **all 1,407 files**:

```text
Household_ID;AffectsTimePoint;Group;Timestamp;kWh_received_Total;kWh_received_HeatPump;kWh_received_Other
```

Every file contains `kWh_received_Total`. The separate heat-pump channel is
not universally available, so it is not the common target. Source values are
preserved: no kW conversion, clipping, rectification, normalization, or truth
interpolation. Blank CSV cells are missing values. Some quarter-hours/day
blocks are absent altogether. The full scan found **no duplicate target
timestamps**, 2,354,208 explicit missing target values and 5,076,384 absent
quarter-hours inside household index envelopes. These are different counts.

The published average availability of approximately 638.75 days is not a
finite-target guarantee. The scan's mean finite-quarter-hour equivalent was
**621.32 days per household**. The archive inspector writes each household's
raw start/end, row count, finite quarter-hours, equivalent available days,
observed days, and missingness. The adapter keeps each household's actual
index range and never pads it to the cohort envelope.

### Weather files

`weather_data/overview/weather_variables_availability.csv` has eight rows and
19 columns: `Weather_ID` plus one boolean for each of these 18 measurements:

```text
Temperature_avg_hourly, Temperature_max_daily, Temperature_min_daily,
Temperature_avg_daily, HeatingDegree_SIA_daily, HeatingDegree_US_daily,
CoolingDegree_US_daily, DewPoint_hourly, Humidity_avg_hourly,
Humidity_avg_daily, Precipitation_total_hourly, Precipitation_total_daily,
Pressure_BarometricHeight_avg_hourly,
Pressure_SeaLevelStandardAtmosphere_avg_hourly, Pressure_SeaLevel_avg_hourly,
Sunshine_duration_hourly, Sunshine_duration_daily, WindSpeed_hourly
```

`weather_data/overview/weather_variables.csv` has 18 rows with columns
`VariableName;ParameterGroup;Resolution;Unit;Description`.

The hourly files have these 11 columns (unavailable station variables can be blank):

```text
Weather_ID;Timestamp;Temperature_avg_hourly;DewPoint_hourly;Humidity_avg_hourly;Precipitation_total_hourly;Pressure_BarometricHeight_avg_hourly;Pressure_SeaLevelStandardAtmosphere_avg_hourly;Pressure_SeaLevel_avg_hourly;Sunshine_duration_hourly;WindSpeed_hourly
```

Station timestamps were also checked for duplicates during the all-station
protocol audit. Temperature is available at all eight stations.

## Completed-interval availability assumption

The source defines quarter-hour active energy as average power multiplied by
0.25 hours, but does **not** conclusively identify left versus right labels.
The loader reads timestamps unchanged and uses ordinary left-labelled pandas
resampling; that alone does not establish the meter's interval convention.

We compared complete UTC-day quarter-hour sums with cumulative-derived daily
totals for ten households. Left-label grouping consistently had lower mean
error, but neither interpretation reconciled all totals. For example,
household 709911 had mean absolute differences of 0.2861 kWh (left) versus
0.2988 kWh (right), over 475/473 complete days. Household 8790869 had
0.4267 versus 0.4773 kWh over 660/657 days. Daily records are labelled
approximately `23:59:59 UTC`. Reconciliation/rounding and missing measurements
prevent treating these checks as conclusive boundary evidence.

**Conservative Simcast assumption:** raw label `t` denotes `[t,t+15min)`.
Shift the index by **+15 minutes**, without changing the energy value, and
make it available only at that completed boundary. If the original label was
already right-boundary, this delays availability by one quarter-hour rather
than leaking unfinished energy. This is an explicit assumption, not a claim
that source documentation established left labels.

The tracked origin is **23:45 UTC**. The final history interval is complete
at 23:45, the first forecast interval ends at the following 00:00, and 96
future steps end at the next 23:45. Lookback is 672 steps (seven days);
horizon is 96 steps (24 hours); stride is 96 steps. Chronological 64%/16%/20%
train/validation/test partitions arise from the generic 80% tune fraction
and 20% validation-within-tune fraction, with generic overlap purging.

## Control selection and station grouping

Treatment households received deliberate heat-pump optimization interventions,
so the tracked benchmark uses controls. The adapter still supports
`data.options.group: control`, `treatment`, or `all`. Its only other
adapter-specific option is `data.options.weather_id`, a station string.

| Weather_ID | All households | Controls | Controls with 15-minute data |
|---|---:|---:|---:|
| `8jB` | 637 | 572 | 572 |
| `Hg` | 477 | 400 | 399 |
| `z6I` | 68 | 49 | 49 |
| `wDD` | 55 | 46 | 46 |
| `HbsbG` | 47 | 37 | 37 |
| `ceOxS` | 41 | 34 | 34 |
| `sV3mR` | 37 | 33 | 33 |
| `MqO` | 46 | 23 | 23 |

The default is the largest control cohort, **8jB**. From its 572 eligible
controls, rank numerically sorted household IDs by the hexadecimal SHA-256
digest of ASCII `simcast-heapo-v1:42:<Household_ID>`, break any digest tie by
ID, take the first 128, and list the result in numeric order.
There is no coverage/performance ranking and no run-time resampling.
The complete, explicit canonical ID list is in
[`control_station_8jB.yaml`](../../configs/bases/heapo/control_station_8jB.yaml).
Canonical IDs are `household::<Household_ID>`, with **K_max = 128**.

This is proximity grouping by nearest MeteoSwiss station, **not electrical
topology**. Exact household coordinates are intentionally absent. Station
identifiers must never become latitude/longitude; the tracked M2/M3 configs
set `features.use_location: false`, and the adapter rejects location features.

## Oracle temperature alignment

The baseline uses only `Temperature_avg_hourly` (degrees Celsius), shared by
the station cohort. HEAPO supplies observations, **not forecast vintages**.
`future_weather_source: oracle` is required: this is an oracle-weather
experiment, **not operationally deployable forecasting**. Historical-only
weather is not implemented.

On the 15-minute grid, sample `h` supplies only `h`, `h+15min`, `h+30min`,
and `h+45min` by zero-order hold. It cannot fill the next missing hour.
Explicit missing measurements remain missing, as do absent hourly blocks.
No historical value is linearly interpolated from a future observation.
Additional weather features can be selected locally, but the tracked
temperature-only baseline remains fixed.

## Variable-cardinality diagnostics and interpretation

Step 3 is reused unchanged: variable candidate coverage is the global target
envelope; canonical axes remain fixed; at least one finite lookback target
and complete required covariates are needed for forecastability. Internal
target gaps remain missing. A recently ended household can remain forecastable
while its lookback still contains a finite observation, but future missing
labels exclude it from the evaluated active set. Do not confuse forecastability
with label/PIT validity.

`heapo-info` writes per-household lifetimes, entry/exit forecast origins,
active fractions, empty-history and covariate exclusions, label missingness,
and active-K by split/lead. Source diagnostics are explicitly **availability
projections**, not fitted PIT results or model inputs. Optional cache reports
use real PIT validity and keep test labels sealed unless
`--include-test-labels` is explicitly requested. `--m1-run` reports actual
fitted overlap counts and independent pairs after PSD repair.

The real selected-cohort source preflight found:

- Completed target envelope: 2019-12-01 00:15 to 2024-02-28 00:00 UTC.
- Lifetime days: minimum 109, median 607.5, mean 639.91, maximum 1,550.
- Finite-quarter-hour equivalent days: minimum 102, median 599, mean 627,
  maximum 1,546.
- 1,542 daily candidate/retained origins; forecastable K ranges 2--128,
  median 22, mean 52.36.
- 116,635 entity-origin empty-history exclusions, no covariate-driven
  exclusions, and 62,368 missing forecastable future entity/lead labels.
- **56 households start after training**, giving **5,376 potential
  entity/lead fallbacks** (56 x 96) from source availability alone.
- Zero training-overlap unordered pairs: 5,573--5,613 per lead, among 8,128
  possible pairs. Fewer than two overlaps: 5,650--5,687 pairs per lead.

| Split | Source-label active K min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| Train | 2 | 11 | 16.39 | 71 |
| Validation | 0 | 111 | 103.68 | 125 |
| Test | 0 | 125 | 124.29 | 127 |

These distributions include below-threshold cases; evaluation admits only
cases with at least two valid active entities. They do not constitute
Chronos/PIT benchmark results. Recheck after real inference using
`pit_training_frequency_fallback` in dependence/evaluation metadata,
`fallback_mask` in `pit_training_frequency_map.npz`, and
`pairwise_overlap_counts` in M1 diagnostics. Training-frequency mapping is
train-only; validation/test observations cannot repair missing training
support. M1 initializes insufficient-overlap pairs to independence, also
handles constant-score pairs, then applies its generic PSD repair.

The evaluation statistic is the **sum of grid-received interval energy over
the active household cohort**. It is neither fixed-population consumption
nor transformer loading. Varying membership changes the estimand; entry
patterns and informative missingness must accompany scientific conclusions.

## Download, audit, cache, and smoke run

```bash
uv run download-data --base configs/bases/heapo/control_station_8jB.yaml

# No Chronos inference: selected cohort availability and all station controls.
uv run heapo-info --base configs/bases/heapo/control_station_8jB.yaml \
  --all-stations-archive data/heapo/archives/heapo_data.zip \
  --output artifacts/diagnostics/heapo_source.json

# Optional exhaustive raw-schema/lifetime audit, streaming the verified ZIP.
uv run python -c "from simcast.data.heapo import inspect_archive; inspect_archive('data/heapo/archives/heapo_data.zip', 'data/heapo/audit')"

uv run cache --base configs/bases/heapo/control_station_8jB.yaml
uv run composite --config configs/venues/lab/heapo_smoke.yaml --run-id heapo_smoke_01
```

The composite reuses the compatible frozen cache, fits M0--M3, and runs its
declared evaluation. Neural smoke fits use two epochs; smoke evaluation uses
256 scenarios, 64 joint-score samples, and retains ES, VS, and pseudo-NLL.
These settings test execution, not convergence or Monte Carlo precision.

After inference/fitting, supply the cache path printed by `cache`:

```bash
uv run heapo-info --base configs/bases/heapo/control_station_8jB.yaml \
  --cache /path/to/printed/cache \
  --m1-run runs/lab/heapo_smoke/heapo_smoke_01/models/heapo_control_station_8jb/m1/deterministic \
  --output artifacts/diagnostics/heapo_fitted_train.json

# Explicitly open test-label-derived diagnostics only for final evaluation.
uv run heapo-info --base configs/bases/heapo/control_station_8jB.yaml \
  --cache /path/to/printed/cache --include-test-labels \
  --output artifacts/diagnostics/heapo_final.json
```

The downloader always verifies official size and MD5, checks every ZIP path
before extraction, and rejects traversal, duplicate members, symlinks, and
unexpectedly large members. It extracts all nine small metadata/overview
CSVs, 128 selected quarter-hour files, and one hourly station file: **138
files, 449,447,436 extracted bytes** for the tracked cohort. Daily/monthly,
cumulative, protocol, unselected household, and unrelated station files stay
inside the archive. Keep the archive for reproducible inspection.

Full extraction is unnecessary, but full upstream download is unavoidable.
The marginal cache still has canonical `[N,128,96,...]` axes, including
missing slots. This preflight has 18,948,096 entity/lead slots: float32 native
quantiles alone require about 75.8 MB **per quantile level** before compression,
plus PIT/truth arrays and `[N,K,P,D]` embeddings. Cache construction currently
holds arrays in memory. Provision RAM/disk and GPU time accordingly; reducing
the first experiment locally is preferable to silently changing the tracked
households or selecting only long-lived ones.

## Regression coverage and lessons for later adapters

[`test_heapo.py`](../../tests/test_heapo.py) tests the real schema contract,
selective checksum-verified download, UTC/completed boundaries, station and
group filters, hash selection, lifetime gaps, oracle weather limits, no
coordinates/metadata leakage, and mock-inference M0--M3 cache/fit/evaluation.
Late joiners trigger the existing nominal fallback and M1 independence;
test labels stay sealed and all methods evaluate the same active cohort.
Existing fixed-cardinality Liander, GEFCom, and FeederBW regressions remain
part of the full suite.

For WPuQ and SDWPF, repeat the raw-file interval/availability audit, distinguish
published dataset extent from usable target support, preserve lifetimes,
document what grouping means physically, and report late-entity training
support before interpreting copula scores. Do not invent coordinates, weather
vintages, topology, or coverage-based entity rankings.
