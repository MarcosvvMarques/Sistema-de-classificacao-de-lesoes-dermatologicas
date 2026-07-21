"""Validação estatística, calibração e ajuste de limiar."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .config import CLASS_NAMES, CLASS_TO_IDX
from .evaluation import calculate_metrics


def predict_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    image_ids: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits.append(model(images).cpu().numpy())
            labels.append(batch["label"].numpy())
            image_ids.extend(batch["image_id"])
    return np.concatenate(labels), np.concatenate(logits), image_ids


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    tensor_logits = torch.tensor(logits, dtype=torch.float64)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.cross_entropy(tensor_logits / temperature, tensor_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correctness = (predictions == labels).astype(float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean())
    return float(ece)


def multiclass_brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(len(CLASS_NAMES))[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def select_melanoma_threshold(
    labels: np.ndarray,
    melanoma_probabilities: np.ndarray,
    target_sensitivity: float = 0.90,
) -> dict[str, float]:
    binary = (labels == CLASS_TO_IDX["mel"]).astype(int)
    fpr, sensitivity, thresholds = roc_curve(binary, melanoma_probabilities)
    valid = np.flatnonzero((sensitivity >= target_sensitivity) & np.isfinite(thresholds))
    if not len(valid):
        raise ValueError("Nenhum limiar atingiu a sensibilidade solicitada.")
    best = valid[np.argmin(fpr[valid])]
    return {
        "threshold": float(thresholds[best]),
        "validation_sensitivity": float(sensitivity[best]),
        "validation_specificity": float(1.0 - fpr[best]),
    }


def threshold_metrics(
    labels: np.ndarray,
    melanoma_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    binary = (labels == CLASS_TO_IDX["mel"]).astype(int)
    predictions = (melanoma_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(binary, predictions, labels=[0, 1]).ravel()
    return {
        "sensitivity": float(recall_score(binary, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(binary, predictions, zero_division=0)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def grouped_bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    iterations: int = 2_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    samples: dict[str, list[float]] = {
        "auc_roc_melanoma": [],
        "auc_roc_macro": [],
        "f1_macro": [],
        "balanced_accuracy": [],
    }
    for _ in range(iterations):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in selected])
        if len(np.unique(labels[indices])) != len(CLASS_NAMES):
            continue
        metrics = calculate_metrics(labels[indices], probabilities[indices])
        for name in samples:
            samples[name].append(float(metrics[name]))
    return {
        name: {
            "estimate": float(calculate_metrics(labels, probabilities)[name]),
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
            "valid_resamples": len(values),
        }
        for name, values in samples.items()
    }


def _plot_calibration(
    labels: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    output_path: Path,
) -> None:
    binary = (labels == CLASS_TO_IDX["mel"]).astype(int)
    fig, axis = plt.subplots(figsize=(7, 6))
    for name, probabilities in (("Antes", raw), ("Após temperature scaling", calibrated)):
        observed, predicted = calibration_curve(
            binary,
            probabilities[:, CLASS_TO_IDX["mel"]],
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", label=name)
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Calibração perfeita")
    axis.set(
        xlabel="Probabilidade prevista de melanoma",
        ylabel="Frequência observada",
        title="Reliability diagram — melanoma",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_threshold_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    selected_threshold: float,
    output_path: Path,
) -> None:
    binary = (labels == CLASS_TO_IDX["mel"]).astype(int)
    fpr, sensitivity, thresholds = roc_curve(binary, probabilities)
    finite = np.isfinite(thresholds)
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(thresholds[finite], sensitivity[finite], label="Sensibilidade")
    axis.plot(thresholds[finite], 1.0 - fpr[finite], label="Especificidade")
    axis.axvline(selected_threshold, linestyle="--", color="black", label="Limiar selecionado")
    axis.axhline(0.90, linestyle=":", color="gray", label="Meta de sensibilidade")
    axis.set(
        xlabel="Limiar de probabilidade",
        ylabel="Pontuação",
        title="Seleção de limiar para melanoma na validação",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_advanced_evaluation(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    split_frame: pd.DataFrame,
    device: torch.device,
    output_dir: Path,
    bootstrap_iterations: int = 2_000,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    val_labels, val_logits, _ = predict_logits(model, val_loader, device)
    test_labels, test_logits, test_ids = predict_logits(model, test_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    raw_test = softmax(test_logits)
    calibrated_test = softmax(test_logits, temperature)
    calibrated_val = softmax(val_logits, temperature)

    threshold = select_melanoma_threshold(
        val_labels,
        calibrated_val[:, CLASS_TO_IDX["mel"]],
    )
    threshold_test = threshold_metrics(
        test_labels,
        calibrated_test[:, CLASS_TO_IDX["mel"]],
        threshold["threshold"],
    )
    lesion_by_image = split_frame.set_index("image_id")["lesion_id"]
    groups = np.asarray([lesion_by_image.loc[image_id] for image_id in test_ids])
    intervals = grouped_bootstrap_intervals(
        test_labels,
        calibrated_test,
        groups,
        iterations=bootstrap_iterations,
    )
    results: dict[str, object] = {
        "temperature": temperature,
        "calibration": {
            "ece_before": expected_calibration_error(test_labels, raw_test),
            "ece_after": expected_calibration_error(test_labels, calibrated_test),
            "brier_before": multiclass_brier(test_labels, raw_test),
            "brier_after": multiclass_brier(test_labels, calibrated_test),
        },
        "threshold_validation": threshold,
        "threshold_test": threshold_test,
        "calibrated_metrics": calculate_metrics(test_labels, calibrated_test),
        "bootstrap_95_ci": intervals,
    }
    (output_dir / "advanced_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    _plot_calibration(test_labels, raw_test, calibrated_test, output_dir / "calibration.png")
    _plot_threshold_curve(
        val_labels,
        calibrated_val[:, CLASS_TO_IDX["mel"]],
        threshold["threshold"],
        output_dir / "threshold_selection.png",
    )
    return results
