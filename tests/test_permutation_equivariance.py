import pytest
import torch

from simcast.dependence.conditional_low_rank import ConditionalLowRankGaussianCopula
from simcast.dependence.set_aware_low_rank import SetAwareLowRankGaussianCopula


@pytest.mark.parametrize("kind", ["m2", "m3"])
def test_conditional_models_are_permutation_equivariant(kind: str) -> None:
    torch.manual_seed(41)
    if kind == "m2":
        model = ConditionalLowRankGaussianCopula(7, hidden_dims=(12,), dropout=0.0).eval()
    else:
        model = SetAwareLowRankGaussianCopula(
            7,
            model_dim=12,
            num_layers=1,
            num_heads=3,
            dropout=0.0,
        ).eval()
    features = torch.randn(3, 8, 7)
    permutation = torch.tensor([5, 0, 7, 2, 1, 6, 4, 3])
    original = model(features)
    permuted = model(features[:, permutation])
    expected = original[:, permutation][:, :, permutation]
    torch.testing.assert_close(permuted, expected, atol=2e-6, rtol=1e-6)


def test_padding_does_not_change_valid_set_aware_principal_matrix() -> None:
    torch.manual_seed(8)
    model = SetAwareLowRankGaussianCopula(
        5,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    ).eval()
    valid = torch.randn(1, 3, 5)
    padded = torch.cat((valid, torch.randn(1, 2, 5)), dim=1)
    mask = torch.tensor([[False, False, False, True, True]])
    direct = model(valid)
    batched = model(padded, padding_mask=mask)
    torch.testing.assert_close(batched[:, :3, :3], direct, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(batched[:, 3:, :3], torch.zeros(1, 2, 3), atol=1e-7, rtol=0)
