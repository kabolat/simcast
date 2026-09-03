import pytest
import torch

from simcast.dependence.conditional_low_rank import ConditionalLowRankGaussianCopula
from simcast.dependence.set_aware_low_rank import SetAwareLowRankGaussianCopula


@pytest.mark.parametrize("num_entities", [3, 5, 11, 25])
def test_same_m2_and_m3_parameters_support_arbitrary_cardinality(num_entities: int) -> None:
    m2 = ConditionalLowRankGaussianCopula(6, latent_rank=3, hidden_dims=(10,), dropout=0.0).eval()
    m3 = SetAwareLowRankGaussianCopula(
        6,
        model_dim=12,
        num_layers=1,
        num_heads=3,
        latent_rank=3,
        dropout=0.0,
    ).eval()
    features = torch.randn(2, num_entities, 6)
    for matrix in (m2(features), m3(features)):
        assert matrix.shape == (2, num_entities, num_entities)
        torch.testing.assert_close(matrix, matrix.transpose(-1, -2), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            matrix.diagonal(dim1=-2, dim2=-1),
            torch.ones(2, num_entities),
            atol=1e-6,
            rtol=0,
        )
        assert torch.linalg.eigvalsh(matrix).min() > 0
