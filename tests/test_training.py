from pathlib import Path

import pytest
import torch

from simcast.cli.train_dependence import _seed_everything
from simcast.dependence.conditional_low_rank import ConditionalLowRankGaussianCopula
from simcast.training.dataset import DependenceCollator, DependenceDataset
from simcast.training.losses import gaussian_copula_pseudo_nll
from simcast.training.trainer import ConditionalTrainer


def _dataset(origins: int = 8, entities: int = 5) -> DependenceDataset:
    torch.manual_seed(origins + entities)
    return DependenceDataset(torch.randn(origins, entities, 2, 6), torch.randn(origins, entities, 2))


def test_deterministic_runtime_is_strict() -> None:
    _seed_everything(7, deterministic=True)
    try:
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        assert torch.backends.cuda.math_sdp_enabled()
    finally:
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def test_dependence_dataset_flattens_only_complete_origin_leads() -> None:
    features = torch.randn(3, 4, 2, 5)
    z = torch.randn(3, 4, 2)
    z[1, 2, 0] = torch.nan
    dataset = DependenceDataset(features, z)
    assert len(dataset) == 5
    assert all(dataset[index].lead in {1, 2} for index in range(len(dataset)))


def test_collator_always_preserves_the_complete_group() -> None:
    dataset = _dataset(3, 7)
    collator = DependenceCollator()
    batch = collator([dataset[0], dataset[1], dataset[2]])
    assert batch.features.shape[1] == 7
    assert batch.z.shape[:2] == (3, 7)


def test_full_group_trainer_rejects_a_shrunken_batch(tmp_path: Path) -> None:
    dataset = _dataset(3, 7)
    model = ConditionalLowRankGaussianCopula(input_dim=6, latent_rank=2, hidden_dims=(8,), dropout=0.0)
    trainer = ConditionalTrainer(
        batch_size=3,
        epochs=1,
        patience=1,
        seed=5,
        expected_num_entities=7,
    )
    def shrink(samples):
        complete = DependenceCollator()(samples)
        return type(complete)(
            features=complete.features[:, :-1],
            z=complete.z[:, :-1],
            active_mask=complete.active_mask[:, :-1],
            canonical_entity_indices=complete.canonical_entity_indices[:, :-1],
            origin_indices=complete.origin_indices,
            leads=complete.leads,
        )

    with pytest.raises(ValueError, match="expected 7 entities"):
        trainer.fit(model, dataset, dataset, output_dir=tmp_path, training_collator=shrink)


def test_variable_dataset_compacts_cases_and_collator_pads_to_batch_maximum() -> None:
    features = torch.arange(2 * 5 * 1 * 2, dtype=torch.float32).reshape(2, 5, 1, 2)
    scores = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5, 1)
    entity_valid = torch.tensor(
        [
            [[True], [False], [True], [False], [True]],
            [[False], [True], [False], [False], [True]],
        ]
    )
    dataset = DependenceDataset(features, scores, entity_valid)

    first, second = dataset[0], dataset[1]
    batch = DependenceCollator()([first, second])

    assert first.active_entity_indices.tolist() == [0, 2, 4]
    assert second.active_entity_indices.tolist() == [1, 4]
    assert first.active_count == 3 and second.active_count == 2
    assert batch.features.shape == (2, 3, 2)
    assert batch.active_mask.tolist() == [[True, True, True], [True, True, False]]
    assert batch.canonical_entity_indices.tolist() == [[0, 2, 4], [1, 4, -1]]
    assert torch.equal(batch.features[1, 2], torch.zeros(2))
    assert torch.isfinite(batch.features).all()


def test_masked_gaussian_nll_matches_manual_principal_submatrices() -> None:
    correlations = torch.tensor(
        [
            [[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]],
            [[1.0, -0.2, 0.4], [-0.2, 1.0, 0.1], [0.4, 0.1, 1.0]],
        ],
        dtype=torch.float64,
    )
    scores = torch.tensor([[0.4, -0.7, 2.0], [0.1, 0.5, -0.3]], dtype=torch.float64)
    active_mask = torch.tensor([[True, False, True], [False, True, True]])
    masked = gaussian_copula_pseudo_nll(
        scores,
        correlations,
        active_mask=active_mask,
        reduction="none",
    )
    expected = []
    for case in range(scores.shape[0]):
        indices = active_mask[case].nonzero(as_tuple=False).flatten()
        expected.append(
            gaussian_copula_pseudo_nll(
                scores[case, indices],
                correlations[case][indices][:, indices],
                reduction="none",
            )
        )

    torch.testing.assert_close(masked, torch.stack(expected))
    all_active = torch.ones_like(active_mask)
    torch.testing.assert_close(
        gaussian_copula_pseudo_nll(scores, correlations, active_mask=all_active),
        gaussian_copula_pseudo_nll(scores, correlations),
    )


def test_masked_gaussian_nll_rejects_cases_below_minimum() -> None:
    with pytest.raises(ValueError, match="min_entities_per_case"):
        gaussian_copula_pseudo_nll(
            torch.ones(1, 3),
            torch.eye(3).unsqueeze(0),
            active_mask=torch.tensor([[True, False, False]]),
        )


def test_conditional_trainer_handles_varying_active_sets(tmp_path: Path) -> None:
    torch.manual_seed(19)
    features = torch.randn(4, 5, 1, 6)
    scores = torch.randn(4, 5, 1)
    entity_valid = torch.tensor(
        [
            [[True], [True], [True], [False], [False]],
            [[False], [True], [False], [True], [False]],
            [[True], [False], [True], [False], [True]],
            [[True], [True], [False], [True], [False]],
        ]
    )
    dataset = DependenceDataset(features, scores, entity_valid)
    model = ConditionalLowRankGaussianCopula(input_dim=6, latent_rank=2, hidden_dims=(8,), dropout=0.0)
    trainer = ConditionalTrainer(
        batch_size=2,
        epochs=1,
        patience=1,
        seed=3,
        min_entities_per_case=2,
    )

    result = trainer.fit(model, dataset, dataset, output_dir=tmp_path)

    assert result.epochs_completed == 1
    assert torch.isfinite(torch.tensor(result.best_validation_nll))


def test_small_trainer_saves_best_final_and_curves(tmp_path: Path) -> None:
    model = ConditionalLowRankGaussianCopula(
        input_dim=6,
        latent_rank=2,
        hidden_dims=(8,),
        dropout=0.0,
    )
    trainer = ConditionalTrainer(batch_size=4, epochs=3, patience=2, seed=5)
    result = trainer.fit(model, _dataset(), _dataset(4), output_dir=tmp_path)
    assert 1 <= result.best_epoch <= result.epochs_completed <= 3
    for filename in ("best.pt", "final.pt", "training_metrics.csv", "training_metrics.json", "training_summary.json"):
        assert (tmp_path / filename).is_file()
