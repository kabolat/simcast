import pytest
import torch

from simcast.dependence.conditional_kernel import ConditionalKernelGaussianCopula
from simcast.training.losses import gaussian_copula_pseudo_nll


@pytest.mark.parametrize("num_entities", [3, 7, 15, 25])
def test_kernel_copula_is_valid_for_variable_cardinality(num_entities: int) -> None:
    model = ConditionalKernelGaussianCopula(6, hidden_dims=(8,), embedding_dim=3, dropout=0.0)
    correlation = model(torch.randn(2, num_entities, 6))
    torch.testing.assert_close(correlation, correlation.mT, atol=1e-6, rtol=0)
    torch.testing.assert_close(correlation.diagonal(dim1=-2, dim2=-1), torch.ones(2, num_entities), atol=1e-6, rtol=0)
    assert torch.linalg.eigvalsh(correlation).min() > 0


def test_kernel_copula_is_permutation_equivariant_and_trainable() -> None:
    torch.manual_seed(4)
    model = ConditionalKernelGaussianCopula(5, hidden_dims=(7,), embedding_dim=3, dropout=0.0)
    features = torch.randn(4, 6, 5)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    original = model(features)
    permuted = model(features[:, permutation])
    torch.testing.assert_close(permuted, original[:, permutation][:, :, permutation], atol=2e-6, rtol=1e-6)
    loss = gaussian_copula_pseudo_nll(torch.randn(4, 6), original)
    loss.backward()
    assert model.raw_length_scale.grad is not None
