# Local experiment configurations

Everything in this directory except this README is ignored by git, as is any
file named `*.local.yaml` anywhere below `configs/`.

## What is tracked

Tracked configurations are the shared, reproducible vocabulary of the project:

- canonical dataset bases (`configs/bases/`);
- canonical method definitions (`configs/methods/`);
- reusable evaluation and report definitions (`configs/evaluations/`,
  `configs/reports/`);
- a small number of representative or demonstration composites
  (`configs/venues/<venue>/`) that are useful for reproduction or
  documentation.

## What stays local

Exploratory experiments (one-off ablations, hyperparameter sweeps, debugging
runs) should not be committed. Create them locally by **extending** tracked
configurations rather than editing the tracked files.

Two equivalent conventions are supported:

1. a file under `configs/local/`, for example
   `configs/local/lab/wind_rank_sweep.yaml`;
2. a file named `*.local.yaml` next to the tracked file, for example
   `configs/venues/lab/quick_shot.local.yaml`.

Bases, methods, and evaluations can live anywhere under `configs/local/`.
Composites must keep the same directory depth as tracked venues, i.e.
`configs/local/<venue>/<name>.yaml`, because relative `config:`/`method:`
paths in a composite (including inherited ones) resolve against the composite
file itself.

```yaml
# configs/local/lab/all_groups_variogram_1.yaml
extends: ../../venues/lab/all_groups_all_methods.yaml
name: all_groups_variogram_1   # distinct name => distinct runs/lab/<name>/ directory

# local exploratory overrides
evaluations:
  - id: variogram_power_1
    config: ../../evaluations/variogram_power_1.yaml
reports: []
```

`extends` deep-merges mappings but **replaces lists**. Overriding `methods`,
`bases`, or `evaluations` therefore replaces the whole list, so restate every
entry that later sections (for example `evaluations[].method_ids` or
`reports[].evaluation_ids`) still reference.

```yaml
# configs/local/m2_wide.yaml (a local method variant)
extends: ../methods/m2_conditional_low_rank.yaml
id: m2_wide
model:
  hidden_dims: [512, 256]
```

The default composite name is the file stem; a `study.local.yaml` composite
defaults to `study_local`. When a local composite extends a tracked composite
that sets `name` explicitly, set a new `name` so that its runs do not mix with
the tracked composite's `runs/<venue>/<name>/` directory.

If a local experiment becomes part of a reported result, promote it to a
tracked configuration in the same change that reports it.
