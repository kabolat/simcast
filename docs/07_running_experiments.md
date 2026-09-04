# Running and reproducing experiments

All dependency management in this repository uses `uv`. Run commands from the
repository root.

## Hardware expectations

Building a real cache loads Chronos-2 and is intended for a CUDA-capable GPU.
Conditional adapters can fall back to CPU if CUDA is unavailable, but Chronos
loading follows `chronos.device`; set it explicitly to `cpu` if needed and
choose a compatible dtype. Full cache construction and five all-type runs are
substantial computations. Reusing a verified cache avoids repeating foundation
model inference.

## 1. Create the locked environment

```bash
uv sync --group dev
```

`uv.lock` is the dependency lock. Do not substitute an ad hoc `pip install` if
the goal is reproduction.

## 2. Install the pinned patched Chronos source

```bash
bash scripts/setup_chronos.sh
```

This checks out the configured source revision, exposes `forecast_embeds`, and
installs the source editable through `uv`. Rerunning the script is safe.

## 3. Select the data directory

```bash
export SIMCAST_DATA_DIR=/absolute/path/to/liander2024
```

If unset, configs default to `data/liander2024` relative to the repository.
The optional device variable is:

```bash
export SIMCAST_DEVICE=cuda
```

## 4. Download the pinned files for one entity type

```bash
uv run python -m simcast.cli.download_data \
  --config configs/liander2024_transformer.yaml
```

Replace the config with `solar_park`, `wind_park`, `mv_feeder`, or
`station_installation` as required. Each invocation downloads that group's
minimum required file selection into the same snapshot directory.

## 5A. Run the complete core experiment

```bash
uv run python -m simcast.cli.run_experiment \
  --config configs/liander2024_transformer.yaml \
  --output-dir runs/my_transformer_experiment
```

This builds the cache if absent, trains M0--M3, and evaluates. The explicit
output directory must not already exist. Omit it to get a UTC timestamped name.

To include the bounded M4 feasibility run:

```bash
uv run python -m simcast.cli.run_experiment \
  --config configs/liander2024_transformer.yaml \
  --output-dir runs/my_transformer_m0_m4 \
  --include-kernel-smoke
```

Use `--rebuild-cache` only when intentionally replacing the configured cache.
The overwrite is scoped to that exact resolved cache directory.

To train only a full M4 model for transformers, reusing the existing cache:

```bash
uv run python -m simcast.cli.train_dependence \
  --config configs/method_conditional_kernel.yaml \
  --output-dir runs/transformer_m4_full
```

The full config uses every chronological training and validation origin, the
standard 100-epoch maximum/patience 12, and the complete entity group. It does not retrain
M0--M3 or rerun Chronos. Evaluation should receive this directory through
`--method-run conditional_kernel=runs/transformer_m4_full`.

## 5B. Run stages separately

Separate stages are preferable when inspecting marginal quality before fitting
dependence or when comparing several training configurations against one cache.

Build the cache:

```bash
uv run python -m simcast.cli.build_cache \
  --config configs/liander2024_transformer.yaml
```

Train the four core methods:

```bash
uv run python -m simcast.cli.train_dependence --config configs/method_independent.yaml
uv run python -m simcast.cli.train_dependence --config configs/method_static_gaussian.yaml
uv run python -m simcast.cli.train_dependence --config configs/method_conditional_low_rank.yaml
uv run python -m simcast.cli.train_dependence --config configs/method_set_aware_low_rank.yaml
```

Then evaluate, preferably giving explicit method-run paths so selection does
not depend on lexicographically latest directory names:

```bash
uv run python -m simcast.cli.evaluate \
  --config configs/liander2024_transformer.yaml \
  --methods independent \
  --methods static_gaussian \
  --methods conditional_low_rank \
  --methods set_aware_low_rank \
  --method-run static_gaussian=/path/to/static_run \
  --method-run conditional_low_rank=/path/to/m2_run \
  --method-run set_aware_low_rank=/path/to/m3_run \
  --output-dir runs/my_evaluation
```

M0 needs no supplied run path during evaluation. If other paths are omitted,
the evaluator selects the lexicographically latest directory under
`output.root_dir` matching `*_<method>`.

## Run every entity type

There is intentionally no hidden batch script; the entity type is a scientific
factor and each run should have an explicit artifact. A reproducible shell loop
for core M0--M3 is:

```bash
for entity_type in transformer solar_park wind_park mv_feeder station_installation; do
  uv run python -m simcast.cli.run_experiment \
    --config "configs/liander2024_${entity_type}.yaml" \
    --output-dir "runs/replication_${entity_type}"
done
```

Solar automatically inherits `pit.monotone_repair: isotonic`; the other four
configs use no repair. Every named config inherits the machine-verified
PowerTech full-group protocol. Add `--include-kernel-smoke` only if the bounded
M4 diagnostic is desired for every group.

The tracked corrected baseline was produced from existing compatible caches
with the following exact commands. The five output directories must not exist
before execution.

```bash
uv run python -m simcast.cli.run_experiment \
  --config configs/liander2024_transformer.yaml \
  --cache-dir artifacts/cache/liander2024_transformer \
  --output-dir runs/powertech2027_transformer_m0_m3

uv run python -m simcast.cli.run_experiment \
  --config configs/liander2024_solar_park.yaml \
  --cache-dir artifacts/cache/liander2024_solar_park_isotonic \
  --output-dir runs/powertech2027_solar_park_m0_m3

uv run python -m simcast.cli.run_experiment \
  --config configs/liander2024_wind_park.yaml \
  --cache-dir artifacts/cache/liander2024_wind_park \
  --output-dir runs/powertech2027_wind_park_m0_m3

uv run python -m simcast.cli.run_experiment \
  --config configs/liander2024_mv_feeder.yaml \
  --cache-dir artifacts/cache/liander2024_mv_feeder \
  --output-dir runs/powertech2027_mv_feeder_m0_m3

uv run python -m simcast.cli.run_experiment \
  --config configs/liander2024_station_installation.yaml \
  --cache-dir artifacts/cache/liander2024_station_installation \
  --output-dir runs/powertech2027_station_installation_m0_m3

uv run python scripts/summarize_powertech_results.py
```

If the compatible caches do not exist, omit `--cache-dir` and allow the runner
to build the configured cache after setting the data directory and device.

## Configuration overrides

Any stage accepts repeatable dotted YAML assignments:

```bash
uv run python -m simcast.cli.run_experiment \
  --config configs/liander2024_transformer.yaml \
  --set seed=7 \
  --set training.epochs=20 \
  --set sampling.num_samples=2048
```

Values are parsed as YAML, so quote shell-sensitive lists and strings. Configs
are strict: unknown keys are errors, lists replace parent lists rather than
merging, and relative `extends` paths resolve relative to the child file.

## Reproduce a saved run

Reproduce conditional/static training from a model run directory:

```bash
uv run python -m simcast.cli.reproduce \
  runs/my_transformer_experiment/conditional_low_rank \
  --output-dir runs/replayed_m2
```

Reproduce evaluation from an evaluation directory:

```bash
uv run python -m simcast.cli.reproduce \
  runs/my_transformer_experiment/evaluation \
  --output-dir runs/replayed_evaluation
```

The command distinguishes the two by `evaluation_manifest.json`. Evaluation
replay uses the cache and model-run paths recorded in that manifest. Training
replay uses the saved `run_metadata.json`. Reproduction therefore requires
those referenced local artifacts to remain available; it does not bundle raw
data, the cache, or all checkpoints into the replay directory.

An entire experiment root is not directly accepted by `reproduce`; pass one of
its method subdirectories or its `evaluation/` subdirectory.

## Inspect before interpreting

After cache construction, inspect:

```bash
python -m json.tool artifacts/cache/liander2024_transformer/metadata.json
python -m json.tool artifacts/cache/liander2024_transformer/marginal_diagnostics.json
```

Check crossing rates, dropped vectors, origin counts, reporting-delay masks,
complete ordered entity set, revisions, and marginal calibration. After
fitting, inspect each `training_summary.json` and training curve. After
evaluation, use `metrics.json`, `metrics_by_lead.csv`, and full-group figures
together; do not select a method from one aggregate number alone.

## Development and scientific regression checks

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
uv lock --check
```

The latest implementation check completed with 130 passing tests. Reported
warnings were xarray/NumPy deprecations, not test failures.

## Common failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| `forecast_embeds` missing | stock Chronos installed | run `scripts/setup_chronos.sh` |
| data file not found | wrong `SIMCAST_DATA_DIR` or group files absent | set the absolute path and run download for that config |
| CUDA or dtype load error | device unavailable or unsupported dtype | override `chronos.device=cpu` and usually `chronos.dtype=float32` |
| cache already exists | overwrite protection | choose another cache or intentionally pass `--overwrite`/`--rebuild-cache` |
| run/evaluation directory exists | immutable run protection | select a new output directory |
| no complete vectors for M1 | missing truth or excessive crossings | inspect metadata; repair only with an explicit scientific rationale |
| horizon exceeds capacity | direct Chronos output patches insufficient | reduce horizon; unrolled embedding semantics are unsupported |
| unknown checkpoint entity | evaluation group differs from fitted group | use a matching cache/checkpoint or retrain |

## Resource-conscious dry runs

Unit tests use injectable mock forecasters and synthetic caches; they do not
validate real Chronos forecast quality. For a quick adapter test on a verified
real cache, reduce epochs through `--set`. Do not present such a run as the full
protocol, and never reduce test sampling only after seeing unfavorable model
results without documenting the change.
