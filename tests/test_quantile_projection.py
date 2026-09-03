import pytest
import torch

from simcast.sampling.quantile_projection import (
    project_uniforms_to_quantiles,
    quantile_cell_boundaries,
)


def test_cell_boundaries_and_exact_boundary_assignment() -> None:
    levels = torch.tensor([0.1, 0.5, 0.9])
    torch.testing.assert_close(quantile_cell_boundaries(levels), torch.tensor([0.0, 0.3, 0.7, 1.0]))
    predictions = torch.tensor([[10.0, 20.0, 30.0]])
    uniforms = torch.tensor([[0.0], [0.2999], [0.3], [0.6999], [0.7], [1.0]])
    samples, indices = project_uniforms_to_quantiles(uniforms, predictions, levels, return_indices=True)
    torch.testing.assert_close(samples[:, 0], torch.tensor([10.0, 10.0, 20.0, 20.0, 30.0, 30.0]))
    torch.testing.assert_close(indices[:, 0], torch.tensor([0, 0, 1, 1, 2, 2]))


def test_batched_projection_preserves_entity_specific_values() -> None:
    values = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
            [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
        ]
    )
    uniforms = torch.tensor([[[0.1, 0.8]], [[0.5, 0.1]]])
    projected = project_uniforms_to_quantiles(uniforms, values, torch.tensor([0.1, 0.5, 0.9]))
    torch.testing.assert_close(projected, torch.tensor([[[1.0, 30.0]], [[5.0, 40.0]]]))


def test_projection_validates_inputs() -> None:
    with pytest.raises(ValueError):
        quantile_cell_boundaries(torch.tensor([0.5, 0.1]))
    with pytest.raises(ValueError):
        project_uniforms_to_quantiles(torch.tensor([[1.1]]), torch.tensor([[1.0, 2.0]]), torch.tensor([0.25, 0.75]))
