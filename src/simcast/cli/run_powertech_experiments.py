"""Execute predeclared full-group confirmatory experiment families."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

import typer

from simcast.cli.build_cache import build_cache_from_config
from simcast.cli.evaluate import evaluate_from_config
from simcast.cli.run_experiment import _with_method
from simcast.cli.train_dependence import _cache_path, train_from_config
from simcast.config import SimcastConfig, deep_merge, load_config

Phase = Literal["main", "ablation", "pit_sensitivity", "rank_sensitivity", "static_sensitivity"]

FEATURE_SETS: dict[str, dict[str, bool]] = {
    "embedding_dynamic_only": {
        "use_forecast_embedding": True,
        "use_quantile_shape": False,
        "use_median": False,
        "use_log_spread": False,
        "use_within_patch_position": True,
        "use_location": False,
    },
    "quantile_dynamic_only": {
        "use_forecast_embedding": False,
        "use_quantile_shape": True,
        "use_median": True,
        "use_log_spread": True,
        "use_within_patch_position": True,
        "use_location": False,
    },
    "combined_dynamic": {
        "use_forecast_embedding": True,
        "use_quantile_shape": True,
        "use_median": True,
        "use_log_spread": True,
        "use_within_patch_position": True,
        "use_location": False,
    },
    "full": {
        "use_forecast_embedding": True,
        "use_quantile_shape": True,
        "use_median": True,
        "use_log_spread": True,
        "use_within_patch_position": True,
        "use_location": True,
    },
}


def _updated(config: SimcastConfig, values: dict[str, object]) -> SimcastConfig:
    return SimcastConfig.model_validate(deep_merge(config.model_dump(mode="python"), values))


def _variants(config: SimcastConfig, phase: Phase) -> Iterator[tuple[str, SimcastConfig]]:
    if phase == "main":
        yield "main", _updated(config, {"confirmatory": {"experiment_family": "main"}})
    elif phase == "ablation":
        for name, features in FEATURE_SETS.items():
            yield name, _updated(
                config,
                {"features": features, "confirmatory": {"experiment_family": "ablation", "feature_set": name}},
            )
    elif phase == "pit_sensitivity":
        for mode in ("nominal_cells", "training_frequency"):
            yield mode, _updated(
                config,
                {"pit": {"dependence_transform": mode}, "confirmatory": {"experiment_family": phase}},
            )
    elif phase == "rank_sensitivity":
        for rank in (2, 4, 8):
            yield f"rank_{rank}", _updated(
                config,
                {
                    "dependence": {
                        "conditional_low_rank": {"latent_rank": rank},
                        "set_aware_low_rank": {"latent_rank": rank},
                    },
                    "confirmatory": {"experiment_family": phase},
                },
            )
    else:
        for pooled in (False, True):
            yield "pooled" if pooled else "lead_specific", _updated(
                config,
                {
                    "dependence": {"static_gaussian": {"share_across_leads": pooled}},
                    "confirmatory": {"experiment_family": phase},
                },
            )


def run_confirmatory_group(
    config: SimcastConfig,
    *,
    phase: Phase = "main",
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    rebuild_cache: bool = False,
    all_m3_ablations: bool = False,
) -> Path:
    """Run one predeclared family for one complete static entity group."""

    if config.protocol.name != "powertech2027" or not config.protocol.full_group_only:
        raise ValueError("confirmatory runner requires a powertech2027 full-group configuration")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = _cache_path(config, cache_dir)
    if rebuild_cache or not cache.is_dir():
        build_cache_from_config(config, output_dir=cache, overwrite=rebuild_cache)

    for variant_name, variant in _variants(config, phase):
        variant_dir = output / variant_name
        variant_dir.mkdir()
        independent = train_from_config(
            _with_method(variant, "independent"), cache_dir=cache, output_dir=variant_dir / "independent"
        )
        static = train_from_config(
            _with_method(variant, "static_gaussian"), cache_dir=cache, output_dir=variant_dir / "static_gaussian"
        )
        if phase == "static_sensitivity":
            evaluate_from_config(
                variant,
                methods=("independent", "static_gaussian"),
                method_runs={"independent": independent, "static_gaussian": static},
                cache_dir=cache,
                output_dir=variant_dir / "evaluation",
            )
            continue

        for seed in variant.confirmatory.neural_seeds:
            seeded = _updated(variant, {"seed": seed})
            seed_dir = variant_dir / f"seed_{seed}"
            seed_dir.mkdir()
            methods = ["independent", "static_gaussian", "conditional_low_rank"]
            runs = {
                "independent": independent,
                "static_gaussian": static,
                "conditional_low_rank": train_from_config(
                    _with_method(seeded, "conditional_low_rank"),
                    cache_dir=cache,
                    output_dir=seed_dir / "conditional_low_rank",
                ),
            }
            run_m3 = phase != "ablation" or all_m3_ablations or variant.confirmatory.feature_set in {
                "combined_dynamic",
                "full",
            }
            if run_m3:
                methods.append("set_aware_low_rank")
                runs["set_aware_low_rank"] = train_from_config(
                    _with_method(seeded, "set_aware_low_rank"),
                    cache_dir=cache,
                    output_dir=seed_dir / "set_aware_low_rank",
                )
            evaluate_from_config(
                seeded,
                methods=methods,
                method_runs=runs,
                cache_dir=cache,
                output_dir=seed_dir / "evaluation",
            )
    return output


def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    phase: Annotated[Phase, typer.Option("--phase")] = "main",
    override: Annotated[list[str] | None, typer.Option("--set")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir", file_okay=False)] = None,
    rebuild_cache: Annotated[bool, typer.Option("--rebuild-cache")] = False,
    all_m3_ablations: Annotated[bool, typer.Option("--all-m3-ablations")] = False,
) -> None:
    resolved = load_config(config, overrides=override or ())
    logging.basicConfig(level=getattr(logging, resolved.runtime.log_level))
    typer.echo(
        run_confirmatory_group(
            resolved,
            phase=phase,
            output_dir=output_dir,
            cache_dir=cache_dir,
            rebuild_cache=rebuild_cache,
            all_m3_ablations=all_m3_ablations,
        )
    )


if __name__ == "__main__":
    typer.run(main)
