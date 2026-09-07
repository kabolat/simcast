# Interactive documentation

These notebooks are an executable companion to `docs/`. They deliberately call
the same public functions used by the CLI: no notebook reimplements Chronos
inference, PIT construction, model fitting, sampling, or scoring.

Start from the repository root:

```bash
uv sync --group dev
bash scripts/setup_chronos.sh
uv run jupyter lab
```

The Chronos setup step installs the pinned patched source that exposes forecast
embeddings; it is the same prerequisite as the CLI cache builder.

Open and run the notebooks in numerical order. Each starts with a `CONFIG_FILE`
and an `OVERRIDES` list. `CONFIG_FILE` is a path below `configs/`; `OVERRIDES`
uses the identical dotted `key=value` syntax as CLI `--set`. For example:

```python
CONFIG_FILE = "configs/liander2024_transformer.yaml"
OVERRIDES = ("chronos.device=cpu", "training.epochs=20")
```

The stages map directly to the CLI:

| Notebook | CLI-equivalent operation |
|---|---|
| `00_configuration_and_reproducibility.ipynb` | `load_config` and configuration validation |
| `01_eda_and_preprocessing.ipynb` | input inspection before `build_cache` |
| `02_chronos_cache_and_features.ipynb` | `simcast.cli.build_cache.build_cache_from_config` |
| `03_dependence_models.ipynb` | `simcast.cli.train_dependence.train_from_config` |
| `04_evaluation_reporting_and_visualisation.ipynb` | `run_experiment_from_config` or `evaluate_from_config` |

The cache grants only training/validation labels unless explicitly opened with
`access="evaluation"`; the evaluation notebook makes that boundary visible.
All notebooks retain the full static group declared in the YAML configuration.
The Boolean run controls default to `False`: change them deliberately before
executing an operation that downloads data, performs Chronos inference, trains
a neural model, or writes a new artifact.
