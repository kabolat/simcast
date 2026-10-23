# Data, forecast origins, and information sets

**Operational realization.** The corresponding data, forecast, covariate, and
split arguments are listed in the
[configuration reference](../technical/configuration_reference.md). The
[usage guide](../technical/usage_guide.md) explains data download and local
data locations. Source revisions, population definitions, source-specific
availability assumptions, and adapter options are documented in the separate
[dataset references](../datasets/README.md).

## Adapters and homogeneous groups

The pipeline consumes a configured adapter rather than assuming one source
schema. An adapter supplies an ordered, static entity group, UTC target and
covariate frames, common coverage, point-in-time input assembly, and
provenance. Forecast frequency, origin phase and stride, lookback, horizon,
and chronological split use shared window and split logic. See the generic
[adapter interface](../technical/data_adapter_interface.md) for the contract.

Each experiment retains its complete ordered canonical group. The default
`protocol.cardinality_mode: fixed` invalidates a case if any member is missing
or invalid. In opt-in `variable` mode, an entity with unavailable forecast-time
inputs is marked non-forecastable; it remains on the canonical cache axis but
is excluded from compact inference. Label validity is recorded separately.
Adapter-specific group definitions and exceptions are documented in the
corresponding dataset reference.

## Timestamp normalization

All internal timestamps are timezone-aware UTC. Readers normalize either a
`timestamp` column or a `DatetimeIndex`, remove the accidental Parquet column
`__index_level_0__`, normalize `available_at` when present for versioned
weather, and stably sort the index. Naive forecast-origin timestamps are
rejected by the window layer.

Calendar features use normalized UTC timestamps. Local daylight-saving shifts
do not alter the configured UTC grid, although an upstream source may still
contain gaps around local-time transitions.

## Forecast windows

Let $\Delta$ be the configured sampling interval, $L$ the lookback length, and
$H$ the horizon length. For forecast instance $i$ with origin $t^{(i)}$, the
exact windows are

$$
\mathcal T_{\mathrm{past}}^{(i)}
=\{t^{(i)}-(L-1)\Delta,\ldots,t^{(i)}\},
$$

$$
\mathcal T_{\mathrm{future}}^{(i)}
=\{t^{(i)}+\Delta,\ldots,t^{(i)}+H\Delta\}.
$$

Thus lead $\tau$ corresponds to $t^{(i)}+\tau\Delta$ and is one-based in public
interfaces. Candidate origins follow the configured time, phase, and stride;
a candidate is created only when common target coverage contains both
complete timestamp grids.

## The point-in-time information set

For every origin, inputs must be measurable with respect to the information
set $\mathcal I^{(i)}$ available at $t^{(i)}$.

### Historical target values

For a target row measured at time $s$, availability is its measurement time:

$$
a_k(s)=s.
$$

The shared selector supplies the target if $s\le t^{(i)}$ and otherwise
replaces it with `NaN`. It intentionally ignores a source target
`available_at` column, so $a_k(s)=s$ is an implementation assumption, not a
universal property of the raw data. Any source-reported release delay must be
handled by its adapter and documented in that source's reference.

Future target values are retained solely as labels. They are not passed to the
Chronos test-mode dataset.

### Historical weather

Measured weather without a separate publication timestamp is treated as
available at its observation time. Only observations at or before
$t^{(i)}$ are selected for history and reindexed to the exact lookback grid.

### Future weather

The default `covariates.future_weather_source: vintage` uses versioned weather
forecasts. For each requested future timestamp $s>t^{(i)}$, the pipeline
filters to vintages with `available_at <= t^{(i)}`, stably sorts by
availability, and takes the newest eligible vintage for $s$. It then reindexes
to the exact horizon grid.

`covariates.future_weather_source: oracle` instead takes realized future
measurement rows on the horizon grid. This deliberately uses information
unavailable at the forecast origin and is therefore an oracle diagnostic,
not a leakage-safe forecasting configuration.

If a configured past or future covariate is missing or non-finite, fixed mode
skips the entire origin before Chronos inference. Variable mode marks only the
affected entity non-forecastable and forecasts the remaining available
entities. It never imputes unknown exogenous information.

## Covariates

The ordered weather fields are selected by `covariates.weather`; their names
and allowed values are adapter-specific and listed in the dataset references.
When enabled, deterministic UTC calendar features are appended. For a periodic variable
$x$ with period $P$, the encoding is $(\sin(2\pi x/P),\cos(2\pi x/P))$.
The implemented cyclic variables are fractional hour with $P=24$ and zero-based
weekday with $P=7$. The additional binary channel is

$$
\mathrm{is\_weekend}=\mathbb 1\{\mathrm{dayofweek}\ge5\}.
$$

There is no day-of-year covariate. With all three calendar blocks enabled,
these rules contribute five channels to past and future inputs.

One `covariates.weather` list selects the feature columns used for both past
and future weather. `future_weather_source` selects whether future values come
from available forecast vintages or realized measurements (`oracle`). The
oracle mode is intentionally leaky and is valid only for adapters with
realized weather measurements.

## Chronological partitioning and purging

Given $N$ ordered eligible origins, let $f_{\mathrm{tune}}$ be
`split.tune_fraction` and $f_{\mathrm{val}}$ be
`split.validation_fraction_within_tune`. The unpurged boundary counts are

$$
N_{\mathrm{tune}}=\lfloor f_{\mathrm{tune}}N\rfloor,
\quad
N_{\mathrm{val}}=\lfloor f_{\mathrm{val}}N_{\mathrm{tune}}\rfloor,
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
partition boundary when the origin stride is shorter than the horizon. The
purge can remove boundary origins; the configured split and retained counts
are saved with each run.

## Missingness and cardinality policy

There are two distinct masks:

- `forecastable_entity[i,k]` is forecast-time information: the required inputs
  and entity forecast are finite and available;
- `entity_valid[i,k,tau]` is label-derived: truth, marginal forecast, PIT, and
  monotonicity validity all hold for that entity and lead.

In fixed mode, one invalid entity invalidates the origin-lead case and all
group PIT scores for it are missing. In variable mode, each entity PIT is kept
independently. The active set is the intersection of forecastable and
label-valid entities, and a case is admitted only when its active count reaches
`protocol.min_entities_per_case`. This same active set is shared by all
methods. `forecastable_entity` is visible in training access because it is
known at forecast time; `entity_valid` and case-level `pit_valid` are sealed
with the test labels.

The variable-mode estimand is conditional on this admission rule. Informative
missingness can still induce selection bias and must be considered in
scientific interpretation.

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

At origin $t^{(i)}$, the history starts at
$t^{(i)}-(L-1)\Delta$ and ends at the origin. The first future target is at
$t^{(i)}+\Delta$ and the final one at $t^{(i)}+H\Delta$. A target measured at
the origin is admissible; every future target remains a label only.

For a requested future covariate timestamp, `vintage` admits only records
whose `available_at` is no later than the origin and selects the newest
eligible record. `oracle` selects realized measurements and must be labelled
an oracle diagnostic.

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
