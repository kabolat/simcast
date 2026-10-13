# Data, forecast origins, and information sets

**Operational realization.** The corresponding data, forecast, covariate, and
split arguments are listed in the
[configuration reference](../technical/configuration_reference.md). The
[usage guide](../technical/usage_guide.md) explains data download and local
data locations.

## Dataset and homogeneous groups

The pipeline consumes a dataset adapter rather than assuming a single data
source. The adapter supplies the ordered entity group, UTC target/covariate
frames, common coverage, point-in-time input assembly, and provenance. Forecast
frequency, origin phase and stride, lookback, horizon, and chronological split
use the same shared window and split logic for every adapter. See the
[dataset adapter guide](../technical/datasets.md) for the interface and setup.

### Liander2024

The Liander2024 bases use the Energy Forecasting Benchmark snapshot
`dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044`. Data are downloaded from
`OpenSTEF/liander2024-energy-forecasting-benchmark` with only the paths needed
for the selected entity type:

```text
liander2024_targets.yaml
load_measurements/<entity_type>/*.parquet
weather_measurements/<entity_type>/*.parquet
weather_forecasts_versioned/<entity_type>/*.parquet
```

The code supports five homogeneous groups:

| Entity type / static group $g$ | Full-group cardinality $K_g$ | Target availability rule |
|---|---:|---|
| `transformer` | 15 | available at measurement timestamp |
| `solar_park` | 5 | available at measurement timestamp |
| `wind_park` | 5 | available at measurement timestamp |
| `mv_feeder` | 15 | available at measurement timestamp |
| `station_installation` | 15 | available at measurement timestamp |

Full-group cardinality is not hardcoded in model code. The group builder selects every
metadata row whose `group_name` equals the requested type, preserving YAML
order. The canonical ID is `group_name::name`; names alone are insufficient
because some physical names occur in multiple categories.

Every experiment uses one static group. Missing data never cause the group to
shrink for a particular case.

### GEFCom2014 Wind

The GEFCom base uses the pinned `rebase-energy/gefcom2014` dataset revision and
reads `data/wind_targets.parquet` and `data/wind_nwp.parquet`. It forms one
ordered group of ten zones, `gefcom_wind::z1` through `gefcom_wind::z10`.
Targets are hourly normalized wind power for 2012-01-01 01:00 through
2013-01-01 00:00; naive source timestamps are interpreted as UTC. Each zone
has four supplied NWP fields (`u10`, `v10`, `u100`, `v100`) and cyclic UTC hour
features. The configured protocol is one origin per day at 00:00 UTC, 168
lookback hours, and 24 future hours. The chronological 80/20 tune/test split
uses the same horizon-purge implementation as every other dataset.

The repack's `issue_time` and `knowledge_time` values describe monthly
competition task releases rather than physical NWP issuance, so they are not
used to establish forecast availability. The observed NWP runs appear stitched
daily, consistent with coverage `(S, S + 24 h]`; the adapter assigns that run
`available_at = S`. The source does not establish when the run was issued or
actually became available. Timestamps are treated as UTC and the run-window
rule is explicitly marked as inferred in cache provenance. Its start hour and
length are fingerprinted base options. With the configured 00:00 origin, the
next 24 valid hours (01:00 through the following 00:00) are selected under
this assumption.
The supplied NWP is the only future weather source; oracle weather is rejected.
The dataset has no locations, so location-based conditional features are
disabled in its supplied lab composite. Missing required input at any zone
makes the full-group origin ineligible.

### FeederBW v1.1

FeederBW v1.1 is Zenodo record [19550073](https://zenodo.org/records/19550073),
DOI `10.5281/zenodo.19550073`, licensed CC BY 4.0. The base pins version
`v1.1`; it does not use the v1.0 record 17831177. The source metadata has 1,608
dated rows for 200 feeders at 121 substations. Its actual identifier columns
are `feeder` and `substation`.

The supplied base selects substation 52 and the complete ascending group
`feederbw::feeder_060`, `feederbw::feeder_106`, `feederbw::feeder_109`,
`feederbw::feeder_165`, and `feederbw::feeder_170`. It is one of two largest
groups (`K=5`), and has substantially more dated metadata than the other
five-feeder candidate. The group is fixed by the base; incomplete targets do
not shrink it. The loader verifies that metadata's group membership exactly
matches the configured ordered IDs.

Measurements are one-minute `active_power_kW` values in UTC. The adapter
computes a mean for each left-labelled 15-minute interval and accepts it only
when the configured threshold of 12--15 source samples is met; the supplied
base uses the strict 15-of-15 threshold. Missing intervals remain missing and
negative values are preserved. At that threshold the selected two-year target
grid contains 70,176 intervals, of which 67,796 (96.61%) are complete for all
five feeders, equivalent to about 706 complete days. Individual feeder
coverage ranges from 98.16% to 99.90%; no source active-power values are null.
The adapter reports exact coverage, per-feeder missingness, and target ranges
in `feederbw-info` output and cache provenance.

Only these eight archived hourly weather fields are configured as historical
inputs: direct and diffuse radiation, air temperature, precipitation,
humidity, maximum wind gust, and meridional and zonal wind. The last hourly
observation no later than each 15-minute history timestamp is used. The archive
contains no forecast vintages, so future inputs are calendar-only
(`future_weather_source: none`); measured future weather is never passed to
Chronos. Metadata is used only to establish the substation group and produce
diagnostics. Latest dated capacities are asynchronous summaries, not static
model predictors.

By default, the downloader fetches shared metadata/weather plus only the
measurement ZIP batches containing configured feeders. For substation 52 that
is batches 041--080, 081--120, and 161--200; it extracts only the five selected
feeder Parquets. Use the dataset-wide download profile
`configs/downloads/feederbw_all.yaml` to fetch all five batches (about 8.44 GB)
and extract all 200 feeder files for later substation selection. It declares
`entity_count: 200` and has no experiment-group selection. This changes local
data availability, not the configured experiment group. Each asset is
verified against its pinned Zenodo size and MD5, and ZIP paths and symlink
members are rejected before extraction.

## Timestamp normalization

All internal timestamps are timezone-aware UTC. Readers normalize either a
`timestamp` column or a `DatetimeIndex`, remove the accidental Parquet column
`__index_level_0__`, normalize `available_at` when present for versioned
weather, and stably sort the index. Naive forecast-origin timestamps are
rejected by the window layer.

This repository uses UTC calendar features. Consequently daylight-saving
changes do not alter the 96-step daily grid, although upstream local-time data
may still contain missing observations around transitions.

## Forecast windows

For the supplied Liander bases, let $\Delta=15$ minutes, lookback $L=672$ steps (seven days), and horizon
$H=96$ steps (one day). For forecast instance $i$ with origin $t^{(i)}$, the exact windows are

$$
\mathcal T_{\mathrm{past}}^{(i)}
=\{t^{(i)}-(L-1)\Delta,\ldots,t^{(i)}\},
$$

$$
\mathcal T_{\mathrm{future}}^{(i)}
=\{t^{(i)}+\Delta,\ldots,t^{(i)}+H\Delta\}.
$$

Thus lead $\tau$ corresponds to $t^{(i)}+\tau\Delta$ and is one-based in public
interfaces. Origins are generated daily (`origin_stride_steps: 96`) at 23:45
UTC on a stable phase anchored to 1970-01-01 23:45 UTC. A candidate is created
only when the common target coverage contains both complete timestamp grids.

The real all-type study retained 347 daily origins, from 2024-01-19 23:45 UTC
through 2024-12-30 23:45 UTC.

## The point-in-time information set

For every origin, inputs must be measurable with respect to the information
set $\mathcal I^{(i)}$ available at $t^{(i)}$.

### Historical target values

For a target row measured at time $s$, availability is its measurement time:

$$
a_k(s)=s.
$$

The target value is supplied to Chronos if $s\le t^{(i)}$; otherwise it is
replaced with `NaN`. A target-file `available_at` column is intentionally
ignored. This makes the target-input rule the same for every entity type.

Future target values are retained solely as labels. They are not passed to the
Chronos test-mode dataset.

### Historical weather

`weather_measurements` has no separate publication timestamp. The conservative
rule treats measurement time as availability time, selects only $s\le t^{(i)}$,
and reindexes to the exact lookback grid.

### Future weather

The default `covariates.future_weather_source: vintage` uses versioned weather
forecasts. For each requested future timestamp $s>t^{(i)}$, the pipeline
filters to vintages with `available_at <= t^{(i)}`, stably sorts by
availability, and takes the newest eligible vintage for $s$. It then reindexes
to the exact horizon grid.

`covariates.future_weather_source: oracle` instead takes the realized future
rows from `weather_measurements` on the horizon grid. This deliberately uses
information unavailable at the forecast origin and is therefore an oracle
diagnostic, not a leakage-safe forecasting configuration.

If any configured past or future covariate is non-finite for any entity, the
entire origin is skipped before Chronos inference. This is stricter than
imputing unknown exogenous information.

## Covariates

The default weather vector, in fixed configured order, is:

1. `temperature_2m`
2. `relative_humidity_2m`
3. `cloud_cover`
4. `wind_speed_10m`
5. `shortwave_radiation`

Five deterministic UTC calendar features are appended. For a periodic variable
$x$ with period $P$, the encoding is $(\sin(2\pi x/P),\cos(2\pi x/P))$.
The implemented cyclic variables are fractional hour with $P=24$ and zero-based
weekday with $P=7$. The additional binary channel is

$$
\mathrm{is\_weekend}=\mathbb 1\{\mathrm{dayofweek}\ge5\}.
$$

There is no day-of-year covariate. This gives 10 past and 10 future covariate
channels by default.

Past and future weather lists are configured independently. For example,
FeederBW has historical weather but no archived future forecast, so its future
weather list is empty and calendar features remain available.

## Chronological partitioning and purging

Given $N$ ordered eligible origins, the unpurged boundary counts are

$$
N_{\mathrm{tune}}=\lfloor0.8N\rfloor,
\quad
N_{\mathrm{val}}=\lfloor0.2N_{\mathrm{tune}}\rfloor,
\quad
N_{\mathrm{train}}=N_{\mathrm{tune}}-N_{\mathrm{val}}.
$$

The last $N-N_{\mathrm{tune}}$ origins are test. Before returning the split,
the algorithm removes origins from the end of each earlier partition while
their last target time is greater than or equal to the first target time in
the next partition:

$$
t^{(i)}+H\Delta\ge t^{(i_{\mathrm{next}})}+\Delta.
$$

This prevents a realized target timestamp from appearing on both sides of a
partition boundary when the origin stride is shorter than the horizon. With
the current daily stride equal to the daily horizon, no boundary origin needs
to be removed; the retained counts are 222 train, 55 validation, and 70 test.
The purge remains important when either setting changes.

## Missingness and complete-vector policy

There are two different eligibility gates:

- an **origin gate** before Chronos: all requested covariates must exist for
  all group members;
- an **origin-lead gate** after forecasts: all $K$ truths and forecasts must be
  finite and every marginal quantile row must be nondecreasing.

If entity $k$ is invalid at $(i,\tau)$, all $K$ PIT entries at that same case
are set to `NaN`. This complete-case policy keeps the estimand tied to the
physical group and gives every dependence model the same cases. It can,
however, induce selection bias if missingness is informative; that limitation
must be considered in scientific interpretation.

## Leakage audit checklist

Before treating a new run as valid, verify that:

- the data and model revisions in `metadata.json` match the intended protocol;
- all target history has measurement timestamp at or before the origin;
- every vintage-weather row has `available_at <= origin_timestamp`, unless an
  explicitly declared oracle diagnostic is being run;
- scalar feature statistics were fitted on training origins only;
- model selection used validation pseudo-NLL, not test results;
- `test_labels.zarr` was opened only through evaluation access;
- any changed stride/horizon still uses boundary purging;
- skipped origins and invalid vectors are reported, not silently imputed.

## Worked origin example

Let $t^{(i)}=$ 2024-06-01 23:45 UTC. With $L=672$ and $H=96$, history begins
at 2024-05-26 00:00 UTC and ends at the origin. The first future target is at
2024-06-02 00:00 UTC and the final one at 23:45 UTC. A target measured at the
origin is immediately admissible; the 00:00 target is a future label.

For a 12:00 future-weather value, `vintage` admits only forecasts whose
`available_at` is no later than the origin. If vintages were issued at 18:00
and 00:00, it selects 18:00. `oracle` instead selects the realized 12:00
weather measurement and must be labelled an oracle diagnostic.

## Configuration and implementation guidance

| Scientific choice | Configuration | Controlling implementation |
|---|---|---|
| static group | `data.entity_type`, `protocol.*` | `data/grouping.py` |
| temporal grid | `forecast.*` | `data/windows.py` |
| immediate target availability | fixed project rule | `data/availability.py` |
| vintage or oracle weather | `covariates.future_weather_source` | `data/availability.py` |
| covariate definition | `covariates.weather`, `covariates.calendar.*` | `data/covariates.py` |
| chronological partitions | `split.*` | `data/windows.py` |

Changing the weather source changes $\mathcal I^{(i)}$ and therefore defines a
different scientific experiment and a different marginal fingerprint. The
cache locator compares that fingerprint with stored metadata; a human-chosen
directory name has no scientific role.
