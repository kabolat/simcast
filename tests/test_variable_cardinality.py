import pytest
import torch

from simcast.dependence.conditional_kernel import ConditionalKernelGaussianCopula
from simcast.dependence.conditional_low_rank import ConditionalLowRankGaussianCopula
from simcast.dependence.set_aware_low_rank import SetAwareLowRankGaussianCopula


@pytest.mark.parametrize("num_entities", [3, 5, 11, 25])
def test_m3_parameters_support_arbitrary_cardinality(num_entities: int) -> None:
    model = SetAwareLowRankGaussianCopula(
        6,
        model_dim=12,
        num_layers=1,
        num_heads=3,
        latent_rank=3,
        dropout=0.0,
    ).eval()
    features = torch.randn(2, num_entities, 6)
    matrix = model(features)
    assert matrix.shape == (2, num_entities, num_entities)
    torch.testing.assert_close(matrix, matrix.transpose(-1, -2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        matrix.diagonal(dim1=-2, dim2=-1),
        torch.ones(2, num_entities),
        atol=1e-6,
        rtol=0,
    )
    assert torch.linalg.eigvalsh(matrix).min() > 0


@pytest.mark.parametrize(
    "model",
    [
        ConditionalLowRankGaussianCopula(6, latent_rank=3, hidden_dims=(12,), dropout=0.0),
        ConditionalKernelGaussianCopula(6, hidden_dims=(12,), embedding_dim=4, dropout=0.0),
    ],
)
def test_m2_and_m4_active_subsets_ignore_padded_nan_rows(model) -> None:
    model.eval()
    active = torch.randn(1, 3, 6)
    unpadded = model(active)
    padded = torch.cat((active, torch.full((1, 2, 6), torch.nan)), dim=1)
    padding_mask = torch.tensor([[False, False, False, True, True]])

    masked = model(padded, padding_mask=padding_mask)

    torch.testing.assert_close(masked[:, :3, :3], unpadded, atol=1e-6, rtol=0)
    assert torch.isfinite(masked).all()
    torch.testing.assert_close(masked[:, 3:, 3:], torch.eye(2).unsqueeze(0))


def test_m3_padding_does_not_change_active_outputs_and_is_permutation_equivariant() -> None:
    torch.manual_seed(11)
    model = SetAwareLowRankGaussianCopula(
        6,
        model_dim=12,
        num_layers=1,
        num_heads=3,
        latent_rank=3,
        dropout=0.0,
    ).eval()
    active = torch.randn(1, 4, 6)
    base = model(active)
    padded = torch.cat((active, torch.full((1, 2, 6), torch.nan)), dim=1)
    mask = torch.tensor([[False, False, False, False, True, True]])
    masked = model(padded, padding_mask=mask)
    torch.testing.assert_close(masked[:, :4, :4], base, atol=1e-6, rtol=0)

    permutation = torch.tensor([2, 5, 0, 4, 3, 1])
    permuted = model(
        padded.index_select(1, permutation),
        padding_mask=mask.index_select(1, permutation),
    )
    expected = masked.index_select(1, permutation).index_select(2, permutation)
    torch.testing.assert_close(permuted, expected, atol=1e-6, rtol=0)
