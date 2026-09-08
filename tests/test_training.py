from pathlib import Path

import pytest
import torch

from simcast.cli.train_dependence import _seed_everything
from simcast.dependence.conditional_low_rank import ConditionalLowRankGaussianCopula
from simcast.training.dataset import DependenceCollator, DependenceDataset
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


def test_subset_collator_uses_variable_but_batch_consistent_cardinality() -> None:
    dataset = _dataset(3, 7)
    collator = DependenceCollator(
        subset_enabled=True,
        min_entities=3,
        full_group_probability=0.0,
        generator=torch.Generator().manual_seed(3),
    )
    batch = collator([dataset[0], dataset[1], dataset[2]])
    assert 3 <= batch.features.shape[1] < 7
    assert batch.z.shape[:2] == batch.features.shape[:2]


def test_full_group_trainer_rejects_legacy_subset_batches(tmp_path: Path) -> None:
    dataset = _dataset(3, 7)
    model = ConditionalLowRankGaussianCopula(input_dim=6, latent_rank=2, hidden_dims=(8,), dropout=0.0)
    trainer = ConditionalTrainer(
        batch_size=3,
        epochs=1,
        patience=1,
        seed=5,
        expected_num_entities=7,
    )
    collator = DependenceCollator(
        subset_enabled=True,
        min_entities=3,
        full_group_probability=0.0,
        generator=torch.Generator().manual_seed(3),
    )

    with pytest.raises(ValueError, match="expected 7 entities"):
        trainer.fit(model, dataset, dataset, output_dir=tmp_path, training_collator=collator)


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
