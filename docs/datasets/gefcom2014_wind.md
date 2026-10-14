# GEFCom2014 Wind

## Source and population

The configured source is the Hugging Face dataset
[`rebase-energy/gefcom2014`](https://huggingface.co/datasets/rebase-energy/gefcom2014), pinned to revision
`713e65e837912e531e5293ca35543c465f1acb4f`. The repack provides
`data/wind_targets.parquet` and `data/wind_nwp.parquet`. It has no dataset card
or license metadata.

The target file contains 8,784 hourly normalized-power observations for each
of ten zones, from 2012-01-01 01:00 through 2013-01-01 00:00. Source timestamps
are naive and are interpreted as UTC. The complete ordered group is
`gefcom_wind::z1` through `gefcom_wind::z10`; missing target or weather input
for a member makes the full-group origin ineligible. There are no entity
coordinates.

## Measurements and NWP availability

The target column is `wind_power`, a dimensionless normalized power value in
`[0, 1]`, not megawatts. The repack does not provide installed capacities or
the normalization denominator. Chronos forecasts are not clipped to that
interval, so inspect out-of-support quantiles before interpreting them as
physically bounded power predictions.

The default evaluation statistic is the unweighted sum of the ten zone values,
$S_t=\sum_{z=1}^{10}q_{z,t}$. It is a dimensionless portfolio index, not the
total electrical output of the wind fleet. Recovering a fleet total in MW
would require zone capacities and the exact normalization convention, neither
of which is available in this repack.

Wind power is not a linear transform of wind speed. Under the usual actuator
disk approximation, kinetic power crossing rotor area $A$ scales as

$$
P_{\mathrm{kin}}=\frac{1}{2}\rho A V^3,
$$

where $\rho$ is air density and $V$ is wind-speed magnitude. A turbine's
electrical output then follows a nonlinear power curve: it rises above
cut-in, is limited near rated power, and can shut down above cut-out. The
repack does not provide turbine curves, hub heights, or capacities, so this
equation is physical intuition, not a transformation applied by the adapter.

The NWP file supplies four fields per zone:

| Field | Conventional meteorological reading |
|---|---|
| `u10` | Zonal horizontal wind component at 10 m |
| `v10` | Meridional horizontal wind component at 10 m |
| `u100` | Zonal horizontal wind component at 100 m |
| `v100` | Meridional horizontal wind component at 100 m |

These four raw components are the only accepted `covariates.weather` values.
Under the conventional component definition, a speed magnitude at height $h$
would be $V_h=\sqrt{u_h^2+v_h^2}$; the adapter does not derive speed or
direction and passes the configured components through. The repack has no
dataset card documenting its component units or sign convention, so verify
those conventions before interpreting derived directions or absolute speeds.
The 100 m components can be a useful proxy for rotor-level flow, but are not a
measured hub-height wind for a known turbine.

Weather describes atmospheric forcing shared by zones, but the conversion
from flow to power also depends on turbine design, wake effects, availability,
and curtailment. Those mechanisms are not identified by these four inputs
alone. Future weather must use `covariates.future_weather_source: vintage`;
no realized weather series is supplied, so `oracle` is rejected.

**Availability assumption.** The source's `issue_time` and `knowledge_time`
columns describe monthly competition task releases, not physical NWP issue or
availability times, and are ignored. The NWP values appear stitched into
daily runs covering valid hours `(S, S + L]`, while realized power has no
matching daily discontinuity. The archive does not establish when a run was
issued or became available. The adapter therefore infers availability at
`S`, fingerprints that assumption with the two options below, and records
`nwp_availability.inferred: true` in provenance. With the supplied 00:00 UTC
origin and 24-hour run, the next 24 valid hours (01:00 through the following
00:00) are used. Other origin phases can be ineligible when a required run is
not yet available.

In symbols, a reconstructed run with start time $S$ and length $R$ supplies
valid times in $(S,S+R]$ and is assumed available at $S$:

$$
a_r(s)=S,\qquad S<s\le S+R.
$$

This is a forecast-timing convention, not evidence of the actual publication
latency. Results are conditional on this assumption; a later real release time
could make some selected covariates unavailable at the stated origin.

The shared protocol is hourly, with 00:00 UTC daily origins, 168 lookback
steps (one week), 24 horizon steps (one day), and a 24-step origin stride. At
origin $t_0$, history contains the previous 168 hourly target timestamps
through $t_0$; future labels run from $t_0+1$ hour through $t_0+24$ hours.
The adapter applies the shared chronological split and overlapping-horizon
purge. Missing inputs are not imputed to create an eligible origin.

## `data.options` reference

Exactly these two keys are required. Both affect the NWP availability mapping
and are included in the marginal-cache fingerprint.

| Key | Type and allowed values | Supplied value | Meaning |
|---|---|---:|---|
| `nwp_run_start_hour_utc` | Integer in `[0, 24)`; booleans are rejected | `0` | UTC hour assigned to the inferred run start/availability time `S` |
| `nwp_run_length_hours` | Positive integer divisor of 24; booleans are rejected | `24` | Length `L` of the inferred valid-time interval `(S, S + L]` |

Other adapter-specific base fields are:

| Field | Supplied value | Meaning |
|---|---|---|
| `data.adapter` | `gefcom2014_wind` | Selects this reader |
| `data.dataset_id` | `rebase-energy/gefcom2014` | Source repository identifier |
| `data.revision` | `713e65e837912e531e5293ca35543c465f1acb4f` | Pinned source snapshot |
| `data.entity_type` | `gefcom_wind` | Required entity-group prefix |
| `data.target_column` | `wind_power` | Target column supplied to the marginal model |
| `covariates.weather` | `[u10, v10, u100, v100]` | Ordered NWP inputs |
| `covariates.future_weather_source` | `vintage` | Uses the inferred availability rule above |

Shared fields are documented in the
[configuration reference](../technical/configuration_reference.md).

## Usage

The supplied base is `configs/bases/gefcom2014/wind.yaml`. Download, build the
frozen marginal cache, then run the small lab composite:

```bash
uv run download-data --base configs/bases/gefcom2014/wind.yaml
uv run cache --base configs/bases/gefcom2014/wind.yaml
uv run composite --config configs/venues/lab/gefcom_wind_smoke.yaml
```

The lab composite uses a small M2 budget for a smoke test, not as a scientific
training recommendation. Cache construction still runs actual Chronos
inference unless a compatible cache already exists.
