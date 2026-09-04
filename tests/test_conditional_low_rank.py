import pytest
import torch

from simcast.dependence.conditional_low_rank import (
    ConditionalLowRankGaussianCopula,
    covariance_to_correlation,
)
from simcast.training.losses import gaussian_copula_pseudo_nll, stabilize_correlation


@pytest.mark.parametrize("num_entities", [3, 5, 11, 25])
def test_valid_correlation_for_variable_cardinality(num_entities: int) -> None:
    torch.manual_seed(4)
    model = ConditionalLowRankGaussianCopula(
        input_dim=7,
        latent_rank=3,
        hidden_dims=(12, 8),
        dropout=0.0,
    )

    correlation = model(torch.randn(2, num_entities, 7))

    assert correlation.shape == (2, num_entities, num_entities)
    torch.testing.assert_close(correlation, correlation.transpose(-1, -2), atol=1e-6, rtol=0)
    torch.testing.assert_close(correlation.diagonal(dim1=-2, dim2=-1), torch.ones(2, num_entities), atol=1e-6, rtol=0)
    assert torch.linalg.eigvalsh(correlation).min() > 0


def test_factor_noise_is_strictly_above_floor() -> None:
    model = ConditionalLowRankGaussianCopula(
        input_dim=4,
        latent_rank=2,
        hidden_dims=(6,),
        dropout=0.0,
        sigma_floor=0.25,
    )

    loadings, sigma = model.factor_parameters(torch.randn(3, 5, 4))

    assert loadings.shape == (3, 5, 2)
    assert sigma.shape == (3, 5)
    assert bool(torch.all(sigma > 0.25))


def test_shared_mlp_is_permutation_equivariant() -> None:
    torch.manual_seed(8)
    model = ConditionalLowRankGaussianCopula(
        input_dim=6,
        latent_rank=3,
        hidden_dims=(10,),
        dropout=0.0,
    ).eval()
    features = torch.randn(4, 7, 6)
    permutation = torch.tensor([5, 0, 6, 2, 1, 4, 3])

    original = model(features)
    permuted = model(features[:, permutation])
    expected = original[:, permutation][:, :, permutation]

    torch.testing.assert_close(permuted, expected, atol=1e-6, rtol=1e-6)


def test_gaussian_copula_pseudo_nll_is_finite_and_differentiable() -> None:
    torch.manual_seed(12)
    model = ConditionalLowRankGaussianCopula(
        input_dim=5,
        latent_rank=2,
        hidden_dims=(9,),
        dropout=0.0,
    )
    features = torch.randn(6, 8, 5)
    z = torch.randn(6, 8)

    loss = gaussian_copula_pseudo_nll(z, model(features))
    loss.backward()

    assert loss.ndim == 0 and torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None)


def test_identity_correlation_has_zero_pseudo_nll() -> None:
    z = torch.randn(3, 5)
    identity = torch.eye(5)

    loss = gaussian_copula_pseudo_nll(z, identity, jitter=0.0, reduction="none")

    torch.testing.assert_close(loss, torch.zeros(3), atol=1e-6, rtol=0)


def test_correlation_normalization_and_loss_stabilization() -> None:
    covariance = torch.tensor([[4.0, 1.0], [1.0, 9.0]])
    correlation = covariance_to_correlation(covariance)
    noisy = correlation.clone()
    noisy[0, 1] += 1.0e-4
    noisy[0, 0] = 0.99

    stabilized = stabilize_correlation(noisy)

    torch.testing.assert_close(stabilized, stabilized.T, atol=1e-7, rtol=0)
    torch.testing.assert_close(stabilized.diagonal(), torch.ones(2), atol=1e-7, rtol=0)
    assert torch.linalg.cholesky_ex(stabilized).info.item() == 0
    stabilized_twice = stabilize_correlation(stabilized)
    torch.testing.assert_close(stabilized_twice, stabilized_twice.T, atol=1e-7, rtol=0)
    torch.testing.assert_close(stabilized_twice.diagonal(), torch.ones(2), atol=1e-7, rtol=0)
    assert torch.linalg.eigvalsh(stabilized_twice).min() > 0
