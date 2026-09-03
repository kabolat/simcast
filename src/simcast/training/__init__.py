"""Training objectives and orchestration for dependence adapters."""

from simcast.training.checkpoint import LoadedConditionalModel, load_conditional_checkpoint
from simcast.training.dataset import DependenceCollator, DependenceDataset
from simcast.training.losses import gaussian_copula_nll, gaussian_copula_pseudo_nll
from simcast.training.trainer import ConditionalTrainer, TrainingResult

__all__ = [
    "ConditionalTrainer",
    "DependenceCollator",
    "DependenceDataset",
    "LoadedConditionalModel",
    "TrainingResult",
    "gaussian_copula_nll",
    "gaussian_copula_pseudo_nll",
    "load_conditional_checkpoint",
]
