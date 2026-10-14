# Dataset References

These pages are the reference for source-specific scientific assumptions and
configuration. Each page records the source and revision, population and
grouping, measurements and covariates, point-in-time availability, supported
protocol, usage commands, and the complete `data.options` contract for that
adapter. Shared configuration fields are documented in the
[configuration reference](../technical/configuration_reference.md).

| Adapter | Dataset reference | Main scientific distinction |
|---|---|---|
| `liander2024` | [Liander2024](liander2024.md) | Load groups with measured historical weather and versioned forecasts |
| `gefcom2014_wind` | [GEFCom2014 Wind](gefcom2014_wind.md) | Hourly normalized wind power with an explicitly inferred NWP availability rule |
| `feederbw` | [FeederBW v1.1](feederbw.md) | One-minute active power aggregated to 15 minutes; the supplied future-weather mode is an oracle diagnostic |
