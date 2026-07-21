"""Métricas e gráficos de avaliação."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from .config import CLASS_NAMES, CLASS_TO_IDX

SUCCESS_TARGETS = {
    "auc_roc_melanoma": 0.85,
    "auc_roc_macro": 0.80,
    "f1_macro": 0.55,
    "balanced_accuracy": 0.65,
}


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    if probabilities.shape != (len(y_true), len(CLASS_NAMES)):
        raise ValueError(
            f"Probabilidades devem ter formato (N, {len(CLASS_NAMES)})."
        )
    y_pred = probabilities.argmax(axis=1)
    melanoma_target = (y_true == CLASS_TO_IDX["mel"]).astype(np.int64)
    melanoma_probability = probabilities[:, CLASS_TO_IDX["mel"]]
    metrics: dict[str, object] = {
        "auc_roc_melanoma": float(
            roc_auc_score(melanoma_target, melanoma_probability)
        ),
        "auc_pr_melanoma": float(
            average_precision_score(melanoma_target, melanoma_probability)
        ),
        "auc_roc_macro": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=np.arange(len(CLASS_NAMES)),
                multi_class="ovr",
                average="macro",
            )
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    metrics["targets_met"] = {
        name: bool(metrics[name] >= target)
        for name, target in SUCCESS_TARGETS.items()
    }
    return metrics


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    image_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            labels.append(batch["label"].cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            image_ids.extend(batch["image_id"])
    if not labels:
        raise ValueError("O DataLoader de avaliação está vazio.")
    return (
        np.concatenate(labels),
        np.concatenate(probabilities),
        image_ids,
    )


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        normalize="true",
    )
    fig, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set(xlabel="Predito", ylabel="Real", title="Matriz de confusão normalizada")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_roc_pr(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
) -> None:
    binary = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))
    fig_roc, axis_roc = plt.subplots(figsize=(8, 7))
    fig_pr, axis_pr = plt.subplots(figsize=(8, 7))
    for index, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(binary[:, index], probabilities[:, index])
        precision, recall, _ = precision_recall_curve(
            binary[:, index], probabilities[:, index]
        )
        axis_roc.plot(fpr, tpr, label=class_name)
        axis_pr.plot(recall, precision, label=class_name)
    axis_roc.plot([0, 1], [0, 1], "--", color="gray")
    axis_roc.set(xlabel="Taxa de falso positivo", ylabel="Sensibilidade", title="Curvas ROC")
    axis_pr.set(xlabel="Recall", ylabel="Precisão", title="Curvas Precisão-Recall")
    axis_roc.legend()
    axis_pr.legend()
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    fig_roc.savefig(output_dir / "roc_curves.png", dpi=160)
    fig_pr.savefig(output_dir / "pr_curves.png", dpi=160)
    plt.close(fig_roc)
    plt.close(fig_pr)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, probabilities, image_ids = predict(model, loader, device)
    metrics = calculate_metrics(y_true, probabilities)
    predictions = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_label": [CLASS_NAMES[index] for index in y_true],
            "predicted_label": [
                CLASS_NAMES[index] for index in probabilities.argmax(axis=1)
            ],
        }
    )
    for index, class_name in enumerate(CLASS_NAMES):
        predictions[f"prob_{class_name}"] = probabilities[:, index]
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    _plot_confusion_matrix(
        y_true,
        probabilities.argmax(axis=1),
        output_dir / "confusion_matrix.png",
    )
    _plot_roc_pr(y_true, probabilities, output_dir)
    return metrics
