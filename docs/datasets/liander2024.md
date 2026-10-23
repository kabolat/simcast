# Liander2024

## Source and population

The configured snapshot is [OpenSTEF's Liander2024 benchmark](https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark)
at revision `dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044`. Its card describes 55
targets in the Netherlands over 2024, with 15-minute load measurements and
hourly weather later interpolated to the load grid. The reader downloads only
the files needed for the configured entity type:

```text
liander2024_targets.yaml
load_measurements/<entity_type>/*.parquet
weather_measurements/<entity_type>/*.parquet
weather_forecasts_versioned/<entity_type>/*.parquet
```

The upstream archive also contains a latest-forecast weather component,
day-ahead prices, and standardized consumption profiles. This adapter does
not read those components: it uses measured weather and the versioned weather
forecasts so the forecast vintage can be selected at each origin. The dataset
card lists CC BY 4.0 for the benchmark and notes that some underlying sources
have separate terms; consult the card before redistribution.

Each experiment uses one complete, static, metadata-defined group. The group
builder selects every metadata row whose `group_name` matches the configured
type and preserves the source YAML order. Entity IDs use
`<group_name>::<name>`; names alone are not unique across categories.

| `data.entity_type` | Entities | Physical meaning | Target scale |
|---|---:|---|---|
| `transformer` | 15 | Power transformers at primary substations | `load`, active power in W |
| `mv_feeder` | 15 | Outgoing medium-voltage cables from primary substations | `load`, active power in W |
| `station_installation` | 15 | Installations within primary substations | `load`, active power in W |
| `solar_park` | 5 | Anonymized individual solar parks | `load`, source-normalized |
| `wind_park` | 5 | Anonymized individual wind parks | `load`, source-normalized |

Group membership does not change by forecast origin. The supplied bases use
the default fixed-cardinality mode: a missing observation excludes that
origin-lead case as a whole. The generic opt-in variable mode can retain other
valid entities without changing this group's canonical membership.

Metadata coordinates are approximate locations inferred from asset names, not
surveyed facility coordinates. Location-conditioned results therefore use
coarse geographic context, not precise siting information.

## Physical interpretation

For entity $k$ in group $g$, write the measured target as $Y_{k,t}$. The
default evaluation statistic sums the full group at each time:

$$
A_{g,t}=\sum_{k\in\mathcal E_g}Y_{k,t}.
$$

When every target is an additive power measurement in watts and the meters do
not overlap, this sum has units of watts and can represent the combined power
of the measured assets. That physical interpretation is not automatic here:
the source combines different grid hierarchy levels across separate group
types, and the park series are anonymized and normalized. For those park
groups, the sum is an unweighted benchmark statistic, not a recoverable total
in watts. The data do not provide an inverse normalization or prove that
measurements within every group are electrically non-overlapping.

Dependence matters because the uncertainty of a sum is not determined by the
individual uncertainty alone:

$$
\mathrm{Var}(A_{g,t})=
\sum_{k\in\mathcal E_g}\mathrm{Var}(Y_{k,t})+
2\sum_{k<\ell}\mathrm{Cov}(Y_{k,t},Y_{\ell,t}).
$$

Weather can help explain common movements: temperature is related to heating
and cooling demand, while cloud cover and shortwave radiation describe solar
conditions relevant to photovoltaic output and net feeder load. These are
predictive relationships, not causal effects identified by this benchmark.

## Measurement and information set

The shared base uses a 15-minute grid, daily origins at 23:45 UTC, a 672-step
(seven-day) lookback, a 96-step (one-day) horizon, and a 96-step origin stride.
For target history, the adapter currently treats measurement time as
availability. The archive includes target `available_at` fields, and the
upstream card states that solar- and wind-park data have a two-day release
delay. The shared selector masks by measurement timestamp and ignores
`available_at` ([implementation](../../src/simcast/data/availability.py)). A
park value can therefore enter model history before its documented release
time. The supplied `solar_park` and `wind_park` configurations are not
point-in-time safe under that source policy; no `data.options` setting changes
this. Correcting it requires the adapter to enforce target availability.

The source's versioned weather records carry `available_at`. Let
$a_{k,s,v}$ be the recorded availability time of vintage $v$ for entity $k$
and valid time $s$. For origin $t_0$, the newest eligible vintage is selected:

$$
v^*(k,s\mid t_0)=\arg\max_{v:\,a_{k,s,v}\le t_0}a_{k,s,v},
\qquad
\widehat W_{k,s\mid t_0}=W_{k,s}^{(v^*)}.
$$

Historical weather comes from the benchmark's Open-Meteo weather-measurements
component. The adapter treats its timestamp as availability; this should not
be mistaken for a sensor installed at every asset. The configured weather
fields, with upstream units and physical interpretation, are:

| Field | Unit | Physical reading |
|---|---|---|
| `temperature_2m` | deg C | Near-surface air temperature |
| `relative_humidity_2m` | percent | Relative humidity near the surface |
| `cloud_cover` | percent | Fraction of the sky covered by cloud |
| `wind_speed_10m` | km/h | Wind speed at 10 m |
| `shortwave_radiation` | W/m^2 | Incoming shortwave solar energy flux |

The upstream card says weather is interpolated from hourly values to the
15-minute load grid. Adjacent sub-hourly rows therefore do not constitute
independent weather measurements; interpolation smooths fast changes. The
versioned forecast archive extends up to seven days ahead, so the configured
one-day horizon fits within its stated lead range.

The base uses `future_weather_source: vintage`, which is the leakage-safe
choice for this archive. The archive also contains realized measurements;
using realized future weather would instead be an oracle diagnostic and must
not be interpreted as an operational forecast.

For the all-type study window used by the project, the common protocol retained
347 daily origins from 2024-01-19 23:45 UTC through 2024-12-30 23:45 UTC. This
is a property of that study window and its eligibility rules, not a guaranteed
count for another base or data snapshot.

## Configuration reference

The supplied bases are under `configs/bases/liander2024/`. The common values
are defined in `common.yaml`; group-specific files declare the entity type.

| Field | Accepted value or constraint | Supplied value | Meaning |
|---|---|---|---|
| `data.adapter` | `liander2024` | `liander2024` | Selects this reader and its validation rules |
| `data.dataset_id` | Pinned source repository | `OpenSTEF/liander2024-energy-forecasting-benchmark` | Identifies the source population |
| `data.revision` | Pinned source revision | `dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044` | Makes the input snapshot reproducible |
| `data.local_dir` | Local path | `${SIMCAST_DATA_DIR:-data/liander2024}` | Download/cache source location; not part of the scientific fingerprint |
| `data.entity_type` | One of the five group names above | Base-specific | Chooses one full homogeneous group |
| `data.target_column` | `load` | `load` | Measured target column |
| `data.options` | Must be empty; no keys are accepted | Omitted, equivalent to `{}` | This adapter has no extra option keys |
| `covariates.weather` | The five fields listed above | All five, in listed order | Ordered weather input columns |
| `covariates.future_weather_source` | `vintage` or `oracle` where available | `vintage` | The selected vintage must have `available_at <= origin` |

All shared fields such as `forecast.*`, `split.*`, `chronos.*`, and `pit.*` are
defined in the [configuration reference](../technical/configuration_reference.md).

## Usage

Download just the files needed by a configured group, then build its marginal
cache:

```bash
uv run download-data --base configs/bases/liander2024/transformer.yaml
uv run cache --base configs/bases/liander2024/transformer.yaml
```

Set `SIMCAST_DATA_DIR` to change the local data root. The configured
`data.local_dir` remains the authoritative location when the variable is not
set.
