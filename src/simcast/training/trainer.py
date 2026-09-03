"""Small local AdamW trainer for conditional Gaussian copula adapters."""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from simcast.training.dataset import DependenceBatch, DependenceCollator, DependenceDataset
from simcast.training.losses import gaussian_copula_pseudo_nll


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    training_nll: float
    validation_nll: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    best_epoch: int
    best_validation_nll: float
    epochs_completed: int
    history: tuple[EpochMetrics, ...]


def _forward(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    correlation = model(features)
    if not isinstance(correlation, torch.Tensor):
        raise TypeError("dependence model must return a torch.Tensor correlation matrix")
    return correlation


class ConditionalTrainer:
    """Train a parameter-count-independent conditional dependence network."""

    def __init__(
        self,
        *,
        batch_size: int = 64,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip_norm: float = 1.0,
        patience: int = 12,
        jitter: float = 1e-6,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ) -> None:
        if min(batch_size, epochs, patience) <= 0:
            raise ValueError("batch_size, epochs, and patience must be positive")
        if learning_rate <= 0 or weight_decay < 0 or gradient_clip_norm <= 0 or jitter < 0:
            raise ValueError("invalid optimizer or numerical configuration")
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.patience = patience
        self.jitter = jitter
        self.seed = seed
        self.device = torch.device(device)

    @staticmethod
    def _mean_loss(
        model: nn.Module,
        loader: DataLoader[DependenceBatch],
        device: torch.device,
        jitter: float,
    ) -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                features = batch.features.to(device)
                scores = batch.z.to(device)
                loss = gaussian_copula_pseudo_nll(scores, _forward(model, features), jitter=jitter, reduction="sum")
                total += float(loss)
                count += scores.shape[0]
        if count == 0:
            raise ValueError("validation loader is empty")
        return total / count

    def fit(
        self,
        model: nn.Module,
        training_data: DependenceDataset,
        validation_data: DependenceDataset,
        *,
        output_dir: str | Path,
        training_collator: DependenceCollator | None = None,
        checkpoint_payload: Callable[[], dict[str, object]] | None = None,
    ) -> TrainingResult:
        """Fit and save best/final local checkpoints plus CSV/JSON curves."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(self.seed)
        collator = training_collator or DependenceCollator(generator=generator)
        training_loader = DataLoader(
            training_data,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
        )
        validation_loader = cast(
            DataLoader[DependenceBatch],
            DataLoader(
                validation_data,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=DependenceCollator(),
            ),
        )
        torch.manual_seed(self.seed)
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        history: list[EpochMetrics] = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            total = 0.0
            count = 0
            for batch in training_loader:
                features = batch.features.to(self.device)
                scores = batch.z.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = gaussian_copula_pseudo_nll(
                    scores,
                    _forward(model, features),
                    jitter=self.jitter,
                    reduction="mean",
                )
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                optimizer.step()
                total += float(loss.detach()) * scores.shape[0]
                count += scores.shape[0]
            validation_loss = self._mean_loss(model, validation_loader, self.device, self.jitter)
            metrics = EpochMetrics(epoch, total / count, validation_loss)
            history.append(metrics)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
                self._save_checkpoint(output / "best.pt", model, epoch, checkpoint_payload)
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                break
        self._save_checkpoint(output / "final.pt", model, len(history), checkpoint_payload)
        if best_state is None:
            raise RuntimeError("training did not produce a best checkpoint")
        model.load_state_dict(best_state)
        self._write_history(output, history)
        result = TrainingResult(best_epoch, best_loss, len(history), tuple(history))
        summary = {"best_epoch": best_epoch, "best_validation_nll": best_loss, "epochs_completed": len(history)}
        (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return result

    @staticmethod
    def _save_checkpoint(
        path: Path,
        model: nn.Module,
        epoch: int,
        payload_factory: Callable[[], dict[str, object]] | None,
    ) -> None:
        payload = {} if payload_factory is None else payload_factory()
        payload.update({"epoch": epoch, "model_state_dict": model.state_dict()})
        torch.save(payload, path)

    @staticmethod
    def _write_history(output: Path, history: list[EpochMetrics]) -> None:
        with (output / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "training_nll", "validation_nll"])
            writer.writeheader()
            writer.writerows(asdict(item) for item in history)
        (output / "training_metrics.json").write_text(
            json.dumps([asdict(item) for item in history], indent=2) + "\n",
            encoding="utf-8",
        )
