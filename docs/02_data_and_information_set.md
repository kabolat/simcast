# Data, forecast origins, and information sets

## Dataset and homogeneous groups

The experiments use the Liander2024 Energy Forecasting Benchmark snapshot
`dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044`. Data are downloaded from
`OpenSTEF/liander2024-energy-forecasting-benchmark` with only the paths needed
for the selected entity type:

```text
liander2024_targets.yaml
load_measurements/<entity_type>/*.parquet
weather_measurements/<entity_type>/*.parquet
weather_forecasts_versioned/<entity_type>/*.parquet
```

The optional EPEX and profile files are disabled in the present experiments.
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

Let $\Delta=15$ minutes, lookback $L=672$ steps (seven days), and horizon
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
\texttt{is\_weekend}=\mathbb 1\{\operatorname{dayofweek}\ge5\}.
$$

There is no day-of-year covariate. This gives 10 past and 10 future covariate
channels by default.

The past and future column lists must match exactly. EPEX and profile flags are
part of the data-download schema but are not incorporated into the implemented
Chronos covariate construction.

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
