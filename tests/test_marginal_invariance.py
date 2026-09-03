import torch

from simcast.sampling.gaussian_copula import generate_scenarios


def test_copula_dependence_does_not_change_discrete_marginal_masses() -> None:
    levels = torch.tensor([0.1, 0.5, 0.9])
    values = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
    identity = torch.eye(3)
    correlated = torch.full((3, 3), 0.75)
    correlated.fill_diagonal_(1.0)
    sample_count = 100_000
    generator = torch.Generator().manual_seed(123)
    base = torch.randn(sample_count, 3, generator=generator)

    independent = generate_scenarios(identity, values, levels, num_samples=sample_count, base_normals=base)
    dependent = generate_scenarios(correlated, values, levels, num_samples=sample_count, base_normals=base)
    expected = torch.tensor([0.3, 0.4, 0.3])
    for result in (independent, dependent):
        for entity in range(3):
            frequencies = torch.bincount(result.quantile_indices[:, entity], minlength=3) / sample_count
            torch.testing.assert_close(frequencies, expected, atol=0.006, rtol=0)

    # The marginals agree, while positive dependence changes joint co-occurrence.
    independent_joint = (independent.quantile_indices[:, 0] == independent.quantile_indices[:, 1]).float().mean()
    dependent_joint = (dependent.quantile_indices[:, 0] == dependent.quantile_indices[:, 1]).float().mean()
    assert dependent_joint > independent_joint + 0.15


def test_common_base_normals_make_sampling_reproducible() -> None:
    levels = torch.tensor([0.25, 0.75])
    values = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    base = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
    left = generate_scenarios(torch.eye(2), values, levels, num_samples=2, base_normals=base)
    right = generate_scenarios(torch.eye(2), values, levels, num_samples=2, base_normals=base)
    torch.testing.assert_close(left.entity_samples, right.entity_samples)
