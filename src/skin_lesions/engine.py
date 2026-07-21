"""Loop de treinamento e persistência de checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .config import CLASS_NAMES, CLASS_TO_IDX, TrainConfig, seed_everything
from .models import create_model, set_backbone_trainable, set_gradual_unfreeze_stage


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        probability = torch.exp(-cross_entropy)
        return (((1.0 - probability) ** self.gamma) * cross_entropy).mean()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    samples = 0
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += batch_size
        all_labels.append(labels.detach().cpu().numpy())
        all_probabilities.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())

    if samples == 0:
        raise ValueError("O DataLoader está vazio.")
    return (
        total_loss / samples,
        correct / samples,
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
    )


def train_model(
    config: TrainConfig,
    loaders: dict[str, DataLoader],
    device: torch.device,
    output_dir: Path,
    loss_weights: torch.Tensor | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    seed_everything(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(
        config.model_name,
        pretrained=config.pretrained,
        drop_rate=config.drop_rate,
    ).to(device)
    device_weights = loss_weights.to(device) if loss_weights is not None else None
    if config.loss_name == "focal":
        criterion = FocalLoss(gamma=config.focal_gamma, weight=device_weights)
    elif config.loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=device_weights)
    else:
        raise ValueError("loss_name deve ser 'cross_entropy' ou 'focal'.")
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    maximize = config.selection_metric != "val_loss"
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_value = -float("inf") if maximize else float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "best.pt"

    for epoch in range(config.epochs):
        if config.gradual_unfreeze:
            set_gradual_unfreeze_stage(model, config.model_name, min(epoch, 2))
        else:
            set_backbone_trainable(model, epoch >= config.freeze_backbone_epochs)
        train_loss, train_accuracy, _train_labels, _train_probabilities = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        with torch.inference_mode():
            val_loss, val_accuracy, val_labels, val_probabilities = _run_epoch(
                model,
                loaders["val"],
                criterion,
                device,
            )
        melanoma_target = (val_labels == CLASS_TO_IDX["mel"]).astype(np.int64)
        val_auc_melanoma = float(
            roc_auc_score(
                melanoma_target,
                val_probabilities[:, CLASS_TO_IDX["mel"]],
            )
        )
        selection_value = (
            val_loss
            if config.selection_metric == "val_loss"
            else val_auc_melanoma
        )
        scheduler.step(selection_value)
        epoch_result = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_auc_roc_melanoma": val_auc_melanoma,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_result)
        print(
            f"Época {epoch + 1:03d}/{config.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f} val_auc_mel={val_auc_melanoma:.4f}"
        )

        improved = (
            selection_value > best_value if maximize else selection_value < best_value
        )
        if improved:
            best_value = selection_value
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_name": config.model_name,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_metric": config.selection_metric,
                    "best_value": best_value,
                    "best_val_loss": val_loss,
                    "best_val_auc_roc_melanoma": val_auc_melanoma,
                    "config": asdict(config),
                    "classes": list(CLASS_NAMES),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping na época {epoch + 1}.")
                break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model, history


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = str(checkpoint["model_name"])
    model = create_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
