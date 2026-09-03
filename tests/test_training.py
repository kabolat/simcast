from pathlib import Path

import torch

from simcast.dependence.conditional_low_rank import ConditionalLowRankGaussianCopula
from simcast.training.dataset import DependenceCollator, DependenceDataset
from simcast.training.trainer import ConditionalTrainer


def _dataset(origins: int = 8, entities: int = 5) -> DependenceDataset:
    torch.manual_seed(origins + entities)
    return DependenceDataset(torch.randn(origins, entities, 2, 6), torch.randn(origins, entities, 2))


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
