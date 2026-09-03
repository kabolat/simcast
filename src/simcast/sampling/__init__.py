"""Copula scenario generation with fixed discrete FM marginals."""

from simcast.sampling.gaussian_copula import ScenarioBatch, generate_scenarios, sample_gaussian_uniforms
from simcast.sampling.quantile_projection import (
    project_uniforms_to_quantiles,
    quantile_cell_boundaries,
)

__all__ = [
    "ScenarioBatch",
    "generate_scenarios",
    "project_uniforms_to_quantiles",
    "quantile_cell_boundaries",
    "sample_gaussian_uniforms",
]
