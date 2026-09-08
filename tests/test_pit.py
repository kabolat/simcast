import numpy as np
import pytest
import torch

from simcast.fm.pit import (
    apply_training_frequency_midpoints,
    build_group_pit,
    crossing_diagnostics,
    dependence_pit_scores,
    discretized_pit,
    fit_training_frequency_midpoints,
    gaussianize_pit,
    quantile_crossings,
)


def test_all_discretized_pit_bins_and_threshold_equalities() -> None:
    levels = torch.tensor([0.1, 0.5, 0.9])
    predictions = torch.tensor([[1.0, 2.0, 3.0]]).expand(8, -1)
    truth = torch.tensor([0.0, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 100.0])
    u, valid = discretized_pit(truth, predictions, levels)
    assert valid.all()
    assert torch.allclose(u, torch.tensor([0.05, 0.05, 0.3, 0.3, 0.7, 0.7, 0.95, 0.95]))


def test_nonuniform_quantile_spacing() -> None:
    levels = torch.tensor([0.05, 0.2, 0.8])
    predictions = torch.tensor([[0.0, 1.0, 2.0]]).expand(4, -1)
    u, _ = discretized_pit(torch.tensor([-1.0, 0.5, 1.5, 3.0]), predictions, levels)
    assert torch.allclose(u, torch.tensor([0.025, 0.125, 0.5, 0.9]))


def test_crossing_is_flagged_or_repaired_explicitly() -> None:
    predictions = torch.tensor([[1.0, 3.0, 2.0]])
    assert quantile_crossings(predictions).item()
    u, valid = discretized_pit(torch.tensor([2.5]), predictions, torch.tensor([0.1, 0.5, 0.9]))
    assert not valid.item()
    assert torch.isnan(u).item()
    repaired_u, repaired_valid = discretized_pit(
        torch.tensor([2.5]), predictions, torch.tensor([0.1, 0.5, 0.9]), monotone_repair="isotonic"
    )
    assert repaired_valid.item()
    assert torch.isfinite(repaired_u).item()


def test_invalid_entity_drops_complete_spatial_vector() -> None:
    truth = torch.ones(2, 3, 2)
    predictions = torch.stack((truth - 1, truth, truth + 1), dim=-1)
    truth[0, 1, 1] = torch.nan
    result = build_group_pit(truth, predictions, torch.tensor([0.1, 0.5, 0.9]))
    assert not result.valid_origin_lead[0, 1]
    assert torch.isnan(result.u[0, :, 1]).all()
    assert result.valid_origin_lead[1].all()


def test_gaussianized_scores_are_inverse_normal() -> None:
    u = torch.tensor([0.1, 0.5, 0.9])
    z = gaussianize_pit(u)
    assert z[1].item() == pytest.approx(0.0)
    assert torch.allclose(torch.special.ndtr(z), u, atol=1e-6)


def test_crossing_diagnostics_axes() -> None:
    predictions = torch.zeros(2, 3, 4, 2)
    predictions[..., 1] = 1.0
    predictions[0, 1, 2] = torch.tensor([2.0, 1.0])
    diagnostics = crossing_diagnostics(predictions)
    assert diagnostics.overall == pytest.approx(1 / 24)
    np.testing.assert_allclose(diagnostics.by_entity, [0.0, 1 / 8, 0.0])
    np.testing.assert_allclose(diagnostics.by_lead, [0.0, 0.0, 1 / 6, 0.0])


def test_rejects_invalid_levels_and_shapes() -> None:
    with pytest.raises(ValueError):
        discretized_pit(torch.ones(2), torch.ones(2, 3), torch.tensor([0.5, 0.1, 0.9]))
    with pytest.raises(ValueError):
        discretized_pit(torch.ones(3), torch.ones(2, 3), torch.tensor([0.1, 0.5, 0.9]))


def test_training_frequency_normalization_uses_training_origins_only() -> None:
    levels = torch.tensor([0.5])
    training_u = torch.tensor([[[0.25]], [[0.25]], [[0.75]]])
    mapping = fit_training_frequency_midpoints(training_u, levels)
    torch.testing.assert_close(mapping, torch.tensor([[[1 / 3, 5 / 6]]]))
    mapped = apply_training_frequency_midpoints(torch.tensor([[[0.25]], [[0.75]]]), levels, mapping)
    torch.testing.assert_close(mapped, torch.tensor([[[1 / 3]], [[5 / 6]]]))

    all_u = torch.cat((training_u, torch.full((10, 1, 1), 0.75)))
    nominal_z = gaussianize_pit(all_u)
    _, fitted = dependence_pit_scores(
        all_u,
        nominal_z,
        torch.tensor([True, True, True] + [False] * 10),
        levels,
        mode="training_frequency",
    )
    assert fitted is not None
    torch.testing.assert_close(fitted, mapping)
