# Interactive documentation

These notebooks are an executable companion to `docs/`. They deliberately call
the same public functions used by the CLI: no notebook reimplements Chronos
inference, PIT construction, model fitting, sampling, or scoring.

Start from the repository root:

```bash
uv sync --group dev
bash scripts/setup_chronos.sh
uv run python -m ipykernel install --user --name simcast --display-name "Python (simcast)"
uv run jupyter lab
```

The Chronos setup step installs the pinned patched source that exposes forecast
embeddings; it is the same prerequisite as the CLI cache builder.

Select the `Python (simcast)` kernel created above, both in JupyterLab and in
VS Code. Do not select another project's `.venv`: `xarray` and all other
runtime dependencies are installed by `uv sync` into this project's virtual
environment. The notebook helper adds the repository's `src/` directory to
`sys.path`, so the local package is found even when the notebook server has a
different working directory. It also changes the notebook process to the
repository root when loading a configuration, so relative config paths such as
`data/liander2024` have the same meaning as in the CLI.

The notebooks have two independent reading paths. The first is the scientific
workflow:

1. `00_configuration_and_reproducibility.ipynb` defines the experiment.
2. `01_eda_and_preprocessing.ipynb` studies data and information.
3. `02_chronos_cache_and_features.ipynb` derives frozen quantiles and PITs.
4. `03_dependence_models.ipynb` compares the model hypotheses.
5. `04_evaluation_reporting_and_visualisation.ipynb` derives the estimands.

The second path consists of independent mathematical monographs in
`dependency_methods/`. Each restarts the notation and can be read without
running notebooks 00--04.

The workflow notebooks accept a base file; each method monograph accepts a base
and exactly one method file. Their separate override lists use the same dotted
`key=value` syntax understood by the corresponding loaders. For example:

```python
BASE_FILE = "configs/bases/liander2024/transformer.yaml"
METHOD_FILE = "configs/methods/m4_conditional_kernel.yaml"
BASE_OVERRIDES = ("chronos.device=cpu",)
METHOD_OVERRIDES = ("optimization.epochs=20",)
```

The stages map directly to the CLI:

| Notebook | CLI-equivalent operation |
|---|---|
| `00_configuration_and_reproducibility.ipynb` | `load_base_config`, `load_method_config`, and role validation |
| `01_eda_and_preprocessing.ipynb` | input inspection before `build_cache` |
| `02_chronos_cache_and_features.ipynb` | `simcast.cli.build_cache.build_cache_from_config` |
| `03_dependence_models.ipynb` | `simcast.cli.train_dependence.train_from_config` |
| `04_evaluation_reporting_and_visualisation.ipynb` | `load_composite_config` and read-only result interpretation |

The cache grants only training/validation labels unless explicitly opened with
`access="evaluation"`; the evaluation notebook makes that boundary visible.
All notebooks retain the full static group declared in the YAML configuration.
The Boolean run controls default to `False`: change them deliberately before
executing an operation that downloads data, performs Chronos inference, trains
a neural model, or writes a new artifact.

The method notebooks `dependency_methods/00_m0_independent_copula.ipynb`
through `dependency_methods/04_m4_conditional_kernel.ipynb` provide a separate
mathematical walk-through for every copula family. They take separate base and
method YAML files and locate the compatible cache from its metadata fingerprint,
exactly as the CLI does. In their default
read-only state they inspect cached arrays, existing checkpoints, and existing
evaluation files only. Their scientific reading guide is the individual
notebook introduction together with
[Dependence models](../docs/scientific/04_dependence_models.md) and
[Training, sampling, and scoring](../docs/scientific/05_training_sampling_scoring.md).
