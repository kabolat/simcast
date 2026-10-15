# FeederBW v1.1

## Source and population

FeederBW v1.1 is [Zenodo record 19550073](https://zenodo.org/records/19550073),
DOI `10.5281/zenodo.19550073`, licensed CC BY 4.0. The base pins v1.1, not
v1.0 (record 17831177). The metadata contains 1,608 dated rows for 200 feeders
at 121 substations, with identifier columns `feeder` and `substation`.

The record describes measurements from low-voltage grids operated by Netze BW
in southwest Germany, collected with the Institute for Automation and Applied
Informatics at KIT. The two-year archive spans 2023--2025. The dataset paper
describes high photovoltaic feed-in and minute-resolution feeder data as
important features of the collection
([Scientific Data](https://www.nature.com/articles/s41597-026-08152-9)).

The supplied substation-52 experiment selects its complete ascending group:
`feederbw::feeder_060`, `feederbw::feeder_106`,
`feederbw::feeder_109`, `feederbw::feeder_165`, and
`feederbw::feeder_170`. Group membership is checked against metadata when the
data are loaded. The group remains fixed; missing observations do not shrink
it for an origin.

The separate `configs/bases/feederbw/all_feeders.yaml` base selects all 200
feeders in ascending ID order as one fixed group. These feeders span all 121
metadata substations; this is a cross-substation dataset group, not a single
electrical substation. The default statistic sums all 200 values, so interpret
it as an experiment-level cross-substation aggregate, not one station's power.
A physical total additionally requires consistent sign conventions and
non-overlapping feeder boundaries.

## Measurements, weather, and information set

The target is one-minute `active_power_kW` in UTC. Let $t$ be the right
boundary of a 15-minute interval and let $M_{f,t}$ be the finite minute
readings for feeder $f$ whose timestamps $s$ lie in $[t-15\text{ min},t)$.
The adapter computes

$$
\bar P_{f,t}=\frac{1}{n_{f,t}}\sum_{s\in M_{f,t}}p_{f,s},
\qquad n_{f,t}=|M_{f,t}|.
$$

The interval is retained only when $n_{f,t}$ is at least the configured
`minimum_samples_per_interval`; otherwise the target is missing. With all 15
samples, $\bar P_{f,t}$ estimates average active power over the quarter-hour.
The corresponding interval energy is approximately
$0.25\text{ h}\times\bar P_{f,t}$ in kWh. For a threshold of 12--14, the
adapter still averages only the available samples: it does not interpolate,
time-weight, or scale them to compensate for missing minutes. Interpreting
that mean as the interval average therefore assumes the observed minutes are
representative of the missing ones.

Timestamps label completed intervals: the target at `23:45` is the mean of
measurements from `23:30` through `23:44`, and the target at `00:00` is the
mean from `23:45` through `23:59`. A measurement timestamped exactly at $t$
belongs to the next interval, never the target at $t$.

At the supplied strict 15-of-15 threshold, the two-year target grid has
70,176 intervals, of which 67,796 (96.61%) are complete for all five feeders.
Individual feeder coverage ranges from 98.16% to 99.90%; source active-power
values are non-null. The complete group has about 706 days before forecast
window requirements are applied. Negative values are retained without
rectification or clipping; verify the source sign convention before reading
their direction as import or export.

For the default evaluation statistic, group power is the sum of feeder means:

$$
P_{g,j}=\sum_{f\in\mathcal E_g}\bar P_{f,j}.
$$

If the selected feeders are non-overlapping circuits measured with a
consistent sign convention, this sum has units of kW and approximates their
combined net active power. The metadata proves that the feeders belong to one
substation, but does not establish circuit topology or rule out overlapping
measurement boundaries. The sum is therefore the configured group statistic;
its interpretation as total substation power requires that additional
electrical assumption. As with any sum, cross-feeder covariance contributes
to uncertainty:

$$
\mathrm{Var}(P_{g,j})=
\sum_f\mathrm{Var}(\bar P_{f,j})+
2\sum_{f<q}\mathrm{Cov}(\bar P_{f,j},\bar P_{q,j}).
$$

The weather file contains hourly model-derived weather from DWD's ICON-D2
system, not a station measurement at each feeder. It contains no archived
forecast vintages. The accepted fields and their physical readings are:

| Field | Unit | Physical reading |
|---|---|---|
| `direct_radiation_W/m2` | W/m^2 | Direct shortwave irradiance |
| `diffuse_radiation_W/m2` | W/m^2 | Diffuse shortwave irradiance |
| `air_temperature_C` | deg C | Near-surface air temperature |
| `precipitation_kg/m2` | kg/m^2 | Precipitation mass per area; interpret the accumulation interval from the source cadence |
| `humidity_kg/kg` | kg/kg | Humidity as a mass ratio, not relative humidity in percent |
| `max_wind_gust_m/s` | m/s | Maximum gust speed, distinct from mean wind speed |
| `meridional_wind_m/s` | m/s | North-south horizontal wind component |
| `zonal_wind_m/s` | m/s | East-west horizontal wind component |

For a 15-minute target timestamp $t$, the adapter uses the most recent
hourly weather timestamp $h$ satisfying $h\le t$, but only while the value
is no more than one hour old:

$$
h^*(t)=\max\{h:h\le t,\ t-h\le 1\text{ h}\},\qquad
W_{f,t}=W_{f,h^*(t)}.
$$

This is a bounded zero-order hold, not interpolation: adjacent quarter-hours
reuse the same hourly atmospheric state, up to an age of 60 minutes inclusive.
If an hourly source row is missing, later quarter-hours remain unavailable once
the preceding row is older than that limit. The same alignment is used for
historical measured weather and future oracle weather; the adapter's normal
window-validity check rejects windows containing unavailable required weather.
Temperature and irradiance are plausible drivers of heating/cooling demand
and photovoltaic feed-in, while wind and precipitation describe other shared
weather conditions. Their association with net feeder power is predictive,
not a causal effect estimated by this experiment.

Future inputs use the same realized valid-time series under
`covariates.future_weather_source: oracle`. Because those values occur after
the forecast origin, they deliberately leak future information; results are
an oracle diagnostic, not an operational forecast. `vintage` is rejected
because no archived forecasts are available.

Each entity is a feeder-level meter, not a household or an individual device.
The dated metadata tracks housing units, industry and commerce, and installed
PV, heat-pump, EV-charger, and battery capacities. Those quantities can help
explain long-term changes in net load, but records are asynchronous and may
change during the two-year measurement period. The adapter uses the metadata
to validate substation membership and report latest-record diagnostics only;
it does not give the latest values to the forecaster, where they could be
stale or unavailable at a historical origin.

Metadata `date` values are local German calendar dates. The adapter interprets
each as midnight in `Europe/Berlin`, applies the timezone's DST rules, then
converts to UTC for internal storage. Thus `2024-01-01` becomes
`2023-12-31T23:00:00Z`, while `2024-07-01` becomes `2024-06-30T22:00:00Z`.
UTC diagnostic timestamps can therefore fall on the preceding calendar day.

The supplied forecast protocol uses 15-minute frequency, 23:45 UTC daily
origins, a 672-step lookback, a 96-step horizon, and a 96-step origin stride.
At a `23:45` origin, the last historical target is timestamped `23:45` and
represents `[23:30, 23:45)`; the first future target is timestamped `00:00` and
represents `[23:45, 00:00)`. The 96 future targets run from `00:00` through
`23:45` the next day, covering exactly the 24 hours after the origin. The
base's calendar covariates include UTC hour, weekday, and weekend features.

## `data.options` reference

Exactly the five keys below are required. The adapter rejects missing or
additional keys. Options describing resampling, group selection, or metadata
use are included in the cache fingerprint.

| Key | Type and allowed values | Supplied value | Meaning |
|---|---|---:|---|
| `substation_id` | Positive integer or literal `all`; booleans are rejected | `52` or `all` | Selects one metadata-defined substation, or every feeder ID 1--200 across all substations; protocol count and ascending IDs must match |
| `source_resolution_minutes` | Must equal integer `1` | `1` | Declares the one-minute source cadence |
| `aggregation` | Must equal `mean` | `mean` | Aggregates source readings into 15-minute target intervals |
| `minimum_samples_per_interval` | Integer from 12 through 15; booleans are rejected | `15` | Minimum number of one-minute readings required for an interval; 15 is strict complete-case aggregation |
| `metadata_usage` | Must equal `grouping_and_diagnostics_only` | `grouping_and_diagnostics_only` | Prevents metadata from being treated as model covariates |

The sample threshold controls the coverage/completeness tradeoff. Requiring
all 15 readings excludes any incomplete quarter-hour; accepting 12--14 can
retain more intervals, but does not reconstruct missing power. If missing
minutes are associated with rapid ramps or high-load conditions, the mean of
the remaining samples can be biased for the full interval.

The selected group and sample threshold are meaningfully adjustable under the
current adapter. Source resolution, aggregation, and metadata usage are fixed
validation guards. For a substation selection, update `protocol.entity_count`
and `protocol.ordered_entity_ids` to list that entire substation in ascending
order. For `substation_id: all`, use all 200 IDs in ascending order and set
`protocol.entity_count: 200` as in the all-feeder base.

Other adapter-specific base values are:

| Field | Supplied value | Meaning |
|---|---|---|
| `data.adapter` | `feederbw` | Selects this reader |
| `data.dataset_id` | `https://zenodo.org/records/19550073` | Pinned source record |
| `data.revision` | `v1.1` | Dataset version |
| `data.entity_type` | `feederbw` | Required entity-group prefix |
| `data.target_column` | `active_power_kW` | Target measurement column |
| `covariates.weather` | The eight fields listed above | Ordered weather inputs |
| `covariates.future_weather_source` | `oracle` | Realized future weather; intentionally leaky |

Shared fields are documented in the
[configuration reference](../technical/configuration_reference.md).

## Download and diagnostics

The substation experiment base is `configs/bases/feederbw/substation_52.yaml`;
the K=200 base is `configs/bases/feederbw/all_feeders.yaml`. Selective download
fetches shared metadata/weather and only the measurement ZIP batches containing
the configured feeders. Substation 52 needs batches 041--080, 081--120, and
161--200 (about 5.05 GB); only the five selected feeder files are extracted.
The all-feeder base selects all five batches and extracts all 200 measurement
files.

To download every measurement file for later group selection, use the
separate profile. It declares `scope: all` and `entity_count: 200`, fetches all
five ZIP batches (about 8.44 GB), and extracts all 200 feeder files. This
changes local availability, not an experiment's configured group. Each asset
is checked against its pinned size and MD5; archive traversal and symlink
entries are rejected.

```bash
uv run download-data --base configs/bases/feederbw/substation_52.yaml
uv run download-data --base configs/bases/feederbw/all_feeders.yaml
uv run download-data --config configs/downloads/feederbw_all.yaml
uv run feederbw-info --base configs/bases/feederbw/substation_52.yaml
uv run feederbw-info --base configs/bases/feederbw/all_feeders.yaml
uv run cache --base configs/bases/feederbw/substation_52.yaml
uv run cache --base configs/bases/feederbw/all_feeders.yaml
uv run composite --config configs/venues/lab/feederbw_smoke.yaml
```

`feederbw-info` reports coverage, per-feeder completeness, target range,
weather window, and latest-per-feeder metadata summaries. The supplied smoke
composite is deliberately small and is not a training-budget recommendation.
