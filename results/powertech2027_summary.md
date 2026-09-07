# PowerTech 2027 full-group result summary

This file is generated only after validating every experiment and method manifest as
`full_group_only: true`, `subset_training: false`, with entity-selection augmentation
disabled. M2 and M3 are newly trained full-group models; no legacy neural checkpoint is used.
Each table describes the saved resolved configuration of its source run; regenerate after
any input-protocol change before treating it as a current experimental result.

## Headline ranking by group

| Group | $K_g$ | Valid cases | Best mean-pinball method | Mean pinball | Change vs M0 | 90% coverage |
|---|---:|---:|---|---:|---:|---:|
| Transformer | 15 | 6,439 | M2 conditional low-rank | 3,886,698 | -1.511% | 0.9219 |
| Solar park | 5 | 6,717 | M1 static | 0.019617192 | -0.633% | 0.8733 |
| Wind park | 5 | 6,664 | M1 static | 0.20608784 | -8.456% | 0.8950 |
| MV feeder | 15 | 6,347 | M1 static | 392,177.125 | -14.715% | 0.8927 |
| Station installation | 15 | 6,427 | M2 conditional low-rank | 5,340,423 | -8.873% | 0.9115 |

## Complete fixed entity groups

### Transformer ($K_g=15$)

- `transformer::OS Apeldoorn`
- `transformer::OS Middenmeer`
- `transformer::RS Grouw (Grou)`
- `transformer::RS Wieringerwerf`
- `transformer::OS Lelystad`
- `transformer::RS Poederoijen`
- `transformer::OS Tiel`
- `transformer::OS Doetinchem`
- `transformer::OS Oterleek`
- `transformer::OS Nijmegen`
- `transformer::OS Amsterdam Hemweg`
- `transformer::RS Surhuisterveen`
- `transformer::OS Ede`
- `transformer::RS Roodwilligen (Duiven)`
- `transformer::RS Minnertsga`

### Solar park ($K_g=5$)

- `solar_park::Within 5 kilometers of Leeuwarden_normalized`
- `solar_park::Within 15 kilometers of Oosterwolde_normalized`
- `solar_park::Within 15 kilometers of Zutphen_normalized`
- `solar_park::Within 20 kilometers of Rotterdam_normalized`
- `solar_park::Within 10 kilometers of Westwoud_normalized`

### Wind park ($K_g=5$)

- `wind_park::Within Stadsregio Arnhem Nijmegen_normalized`
- `wind_park::Within 15 kilometers of Opmeer_normalized`
- `wind_park::Within 20 kilometers of Leeuwarden_normalized`
- `wind_park::Within 15 kilometers of Alphen aan den Rijn_normalized`
- `wind_park::Within 15 kilometers of Dronten_normalized`

### MV feeder ($K_g=15$)

- `mv_feeder::OS Naarden`
- `mv_feeder::OS Leiden Noord`
- `mv_feeder::SS Harlingen`
- `mv_feeder::OS Eibergen`
- `mv_feeder::OS Sneek`
- `mv_feeder::OS Weesp`
- `mv_feeder::SS Ureterp`
- `mv_feeder::OS Gorredijk`
- `mv_feeder::OS Sassenheim`
- `mv_feeder::OS Waarderpolder`
- `mv_feeder::OS Edam`
- `mv_feeder::OS Watergraafsmeer`
- `mv_feeder::RS Angerlo`
- `mv_feeder::SS Heerenveen Pim Mulier`
- `mv_feeder::RS Roodwilligen (Duiven)`

### Station installation ($K_g=15$)

- `station_installation::OS Apeldoorn`
- `station_installation::RS Hallum`
- `station_installation::OS Almere`
- `station_installation::SS Stiens`
- `station_installation::OS Herbayum`
- `station_installation::OS Westhaven`
- `station_installation::OS Zevenhuizen`
- `station_installation::RS Anklaar`
- `station_installation::SS Noordwolde`
- `station_installation::OS Zevenaar`
- `station_installation::OS Bergum`
- `station_installation::OS Texel`
- `station_installation::RS Waskemeer`
- `station_installation::RS Wamel`
- `station_installation::OS Drachten`

## Complete metric table

Energy Score is the empirical all-pairs estimator on the selected 512-member joint ensemble.

| Group | Method | Mean pinball | WIS | 90% coverage | 90% width | Energy Score |
|---|---|---:|---:|---:|---:|---:|
| Transformer | M0 independent | 3,946,312.5 | 30,693,540 | 0.9258 | 62,003,436 | 12,806,960 |
| Transformer | M1 static | 3,944,830.25 | 30,682,010 | 0.9144 | 59,698,028 | 12,739,834 |
| Transformer | M2 conditional low-rank | 3,886,698 | 30,229,870 | 0.9219 | 59,714,272 | 12,725,114 |
| Transformer | M3 set-aware | 3,904,673 | 30,369,680 | 0.9230 | 59,424,064 | 12,730,929 |
| Solar park | M0 independent | 0.019742087 | 0.15354955 | 0.8446 | 0.25837108 | 0.037339915 |
| Solar park | M1 static | 0.019617192 | 0.15257819 | 0.8733 | 0.30911481 | 0.037203334 |
| Solar park | M2 conditional low-rank | 0.020104649 | 0.15636951 | 0.8921 | 0.35972953 | 0.037245993 |
| Solar park | M3 set-aware | 0.020085147 | 0.15621781 | 0.8860 | 0.36205295 | 0.037256483 |
| Wind park | M0 independent | 0.22512449 | 1.7509683 | 0.6955 | 1.6298919 | 0.32283098 |
| Wind park | M1 static | 0.20608784 | 1.6029054 | 0.8950 | 2.5734615 | 0.32034478 |
| Wind park | M2 conditional low-rank | 0.20611842 | 1.6031435 | 0.8969 | 2.6567435 | 0.32047984 |
| Wind park | M3 set-aware | 0.20619807 | 1.6037629 | 0.9034 | 2.7380993 | 0.32067937 |
| MV feeder | M0 independent | 459,843.594 | 3,576,561.5 | 0.6342 | 2,713,431.25 | 527,885.562 |
| MV feeder | M1 static | 392,177.125 | 3,050,266.75 | 0.8927 | 5,173,175.5 | 522,559.031 |
| MV feeder | M2 conditional low-rank | 396,026.625 | 3,080,207.25 | 0.9450 | 6,425,532 | 521,631.375 |
| MV feeder | M3 set-aware | 397,109.031 | 3,088,625.75 | 0.9464 | 6,508,360 | 521,691 |
| Station installation | M0 independent | 5,860,416.5 | 45,581,016 | 0.7192 | 47,225,272 | 9,127,623 |
| Station installation | M1 static | 5,348,543 | 41,599,780 | 0.8827 | 71,646,592 | 9,061,769 |
| Station installation | M2 conditional low-rank | 5,340,423 | 41,536,624 | 0.9115 | 80,262,528 | 9,050,469 |
| Station installation | M3 set-aware | 5,370,834.5 | 41,773,160 | 0.8859 | 73,881,952 | 9,052,517 |
