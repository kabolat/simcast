# Simcast

Simcast is a research proof of concept for learning same-lead, cross-entity
forecast-error dependence on top of frozen Chronos-2 marginal forecasts. It
keeps every entity-wise quantile forecast fixed and changes only the copula
used to form probabilistic spatial aggregates.

The default experiment uses one static group containing all Liander2024
transformers, 15-minute data, seven days of context, and the next complete UTC
day as the forecast horizon.

## Setup

```bash
uv sync --group dev
bash scripts/setup_chronos.sh
```

All commands run through the project environment, for example:

```bash
uv run python -m simcast.cli.download_data --config configs/liander2024_transformer.yaml
uv run python -m simcast.cli.run_experiment --config configs/liander2024_transformer.yaml
```

Raw data, model weights, generated caches, and run outputs are intentionally
excluded from Git.

## Attribution

The Liander2024 Energy Forecasting Benchmark is published by OpenSTEF under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The default data
source is `OpenSTEF/liander2024-energy-forecasting-benchmark` on Hugging Face.

Chronos is Copyright Amazon.com, Inc. or its affiliates and licensed under
Apache-2.0. Simcast applies a minimal local patch that exposes an already
computed forecast-patch representation without changing forecast numerics.

Simcast's own source code is licensed under the MIT License.

