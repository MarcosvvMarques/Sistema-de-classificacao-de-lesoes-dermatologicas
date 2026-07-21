"""Configuração compartilhada pelo pipeline."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}

MODEL_INPUT_SIZES = {
    "resnet18": 224,
    "efficientnet_b3": 300,
}


@dataclass(slots=True)
class TrainConfig:
    model_name: str
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5
    num_workers: int = 0
    seed: int = 42
    pretrained: bool = True
    freeze_backbone_epochs: int = 1
    imbalance_strategy: str = "class_weights"
    loss_name: str = "cross_entropy"
    focal_gamma: float = 2.0
    selection_metric: str = "val_loss"
    gradual_unfreeze: bool = False
    drop_rate: float = 0.0

    @property
    def image_size(self) -> int:
        try:
            return MODEL_INPUT_SIZES[self.model_name]
        except KeyError as exc:
            supported = ", ".join(MODEL_INPUT_SIZES)
            raise ValueError(f"Modelo inválido. Use: {supported}") from exc


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas não está disponível.")
    return torch.device(requested)


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_dirs() -> None:
    for directory in (ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
