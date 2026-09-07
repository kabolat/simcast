# Legacy entity-selection audit

This audit records the resolved configuration actually saved with the earlier
runs. It prevents their neural results from being mistaken for full-group
full-group results.

| Legacy run | M2 subset training | M3 subset training | M4 subset training | M4 scope |
|---|---|---|---|---|
| `real_liander2024_transformer_m0_m4` | enabled | enabled | disabled | bounded smoke |
| `real_liander2024_solar_park_isotonic_m0_m4` | enabled | enabled | disabled | bounded smoke |
| `real_liander2024_wind_park_m0_m4` | enabled | enabled | disabled | bounded smoke |
| `real_liander2024_mv_feeder_m0_m4` | enabled | enabled | disabled | bounded smoke |
| `real_liander2024_station_installation_m0_m4` | enabled | enabled | disabled | bounded smoke |
| `real_liander2024_transformer_m4_full` | not part of this run | not part of this run | enabled | full-duration exploratory training |

In the five original multi-method runs, M0 and M1 did not execute the generic
subset collator: their implementations fit and evaluated complete group
vectors. Their resolved configurations nevertheless inherited the old global
subset flag, so their old metadata is not protocol-clean. New M0 and M1
artifacts were therefore created alongside the corrected M2 and M3 artifacts.

The M2 and M3 entries above sampled random entity subsets during training and
are legacy/exploratory only. The bounded M4 checkpoints did not subset entities
but remain diagnostics because their training budget was intentionally smaller.
The later transformer M4 checkpoint used the normal training-origin budget but
did use subset training; it is also legacy/exploratory. No old M2, M3, or M4
number is a headline full-group result.
