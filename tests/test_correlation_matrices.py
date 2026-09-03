from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from simcast.dependence import IndependentCopula, StaticGaussianCopula


def _training_scores(n_origins: int = 500) -> np.ndarray:
    rng = np.random.default_rng(13)
    lead_zero = rng.multivariate_normal(
        np.zeros(3),
        np.asarray([[1.0, 0.85, 0.10], [0.85, 1.0, 0.05], [0.10, 0.05, 1.0]]),
        size=n_origins,
    )
    lead_one = rng.multivariate_normal(
        np.zeros(3),
        np.asarray([[1.0, 0.05, -0.70], [0.05, 1.0, 0.20], [-0.70, 0.20, 1.0]]),
        size=n_origins,
    )
    return np.stack((lead_zero, lead_one), axis=-1)


def _assert_valid_correlation(correlation: torch.Tensor) -> None:
    assert torch.allclose(correlation, correlation.mT, atol=1e-12)
    assert torch.allclose(torch.diagonal(correlation), torch.ones(correlation.shape[0], dtype=correlation.dtype))
    assert torch.linalg.eigvalsh(correlation).min() >= -1e-12
    torch.linalg.cholesky(correlation)


def test_independent_identity_sampling_and_roundtrip(tmp_path: Path) -> None:
    model = IndependentCopula(["a", "b", "c"], n_leads=2)
    correlation = model.correlation_matrix(1, entity_ids=["c", "a"])
    assert torch.equal(correlation, torch.eye(2, dtype=torch.float64))
    _assert_valid_correlation(correlation)

    first = model.sample_uniforms(32, 1, generator=torch.Generator().manual_seed(7))
    second = model.sample_uniforms(32, 1, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)
    assert bool(((first > 0) & (first < 1)).all())

    path = model.save(tmp_path / "m0.npz")
    restored = IndependentCopula.load(path)
    assert restored.entity_ids == model.entity_ids
    assert torch.equal(restored.correlation_matrix(1), torch.eye(3, dtype=torch.float64))


def test_static_per_lead_is_valid_and_respects_entity_order() -> None:
    model = StaticGaussianCopula(jitter=1e-6).fit(_training_scores(), ["a", "b", "c"])
    first = model.correlation_matrix(1)
    second = model.correlation_matrix(2)
    _assert_valid_correlation(first)
    _assert_valid_correlation(second)
    assert first[0, 1] > 0.6
    assert second[0, 2] < -0.5
    assert not torch.allclose(first, second)

    order = torch.tensor([2, 0, 1])
    reordered = model.correlation_matrix(1, entity_ids=["c", "a", "b"])
    assert torch.equal(reordered, first.index_select(0, order).index_select(1, order))
    _assert_valid_correlation(model.correlation_matrix(1, entity_ids=["b", "a"]))


def test_static_pooled_shares_matrix_across_leads_and_roundtrips(tmp_path: Path) -> None:
    model = StaticGaussianCopula(share_across_leads=True, jitter=1e-5).fit(
        _training_scores(),
        ["a", "b", "c"],
    )
    assert torch.equal(model.correlation_matrix(1), model.correlation_matrix(2))

    restored = StaticGaussianCopula.load(model.save(tmp_path / "m1.npz"))
    assert restored.entity_ids == ("a", "b", "c")
    assert restored.share_across_leads
    assert restored.jitter == pytest.approx(1e-5)
    assert torch.equal(restored.correlation_matrix(1), model.correlation_matrix(1))


def test_static_drops_an_incomplete_vector_as_a_whole() -> None:
    complete = _training_scores(50)
    incomplete = complete.copy()
    incomplete[0, 1, 0] = np.nan
    expected_mask = np.ones((50, 2), dtype=np.bool_)
    expected_mask[0, 0] = False

    from_nan = StaticGaussianCopula().fit(incomplete, ["a", "b", "c"])
    from_mask = StaticGaussianCopula().fit(complete, ["a", "b", "c"], expected_mask)
    assert torch.equal(from_nan.correlation_matrix(1), from_mask.correlation_matrix(1))
    assert torch.equal(from_nan.correlation_matrix(2), from_mask.correlation_matrix(2))


def test_static_rejects_unknown_entities_and_too_few_vectors() -> None:
    model = StaticGaussianCopula().fit(_training_scores(20), ["a", "b", "c"])
    with pytest.raises(ValueError, match="unknown"):
        model.correlation_matrix(1, entity_ids=["missing"])
    with pytest.raises(ValueError, match="at least two"):
        StaticGaussianCopula().fit(_training_scores(1), ["a", "b", "c"])
