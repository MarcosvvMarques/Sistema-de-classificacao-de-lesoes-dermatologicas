"""Gera gráficos comparativos a partir dos artefatos de treinamento."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ("resnet18", "efficientnet_b3")
LABELS = {"resnet18": "ResNet18", "efficientnet_b3": "EfficientNet-B3"}
OUTPUT_DIR = ROOT / "reports" / "comparison"
TARGETS = {
    "auc_roc_melanoma": 0.85,
    "auc_roc_macro": 0.80,
    "f1_macro": 0.55,
    "balanced_accuracy": 0.65,
}
METRIC_LABELS = {
    "auc_roc_melanoma": "AUC-ROC\nmelanoma",
    "auc_roc_macro": "AUC-ROC\nmacro",
    "f1_macro": "F1 macro",
    "balanced_accuracy": "Acurácia\nbalanceada",
}
CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_training_curves(histories: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")
    for column, model_name in enumerate(MODELS):
        history = histories[model_name]
        epochs = [int(row["epoch"]) for row in history]
        axes[0, column].plot(
            epochs, [row["train_loss"] for row in history], label="Treino"
        )
        axes[0, column].plot(
            epochs, [row["val_loss"] for row in history], label="Validação"
        )
        axes[0, column].set_title(f"{LABELS[model_name]} — Loss")
        axes[0, column].set_ylabel("Cross-entropy")
        axes[0, column].legend()
        axes[0, column].grid(alpha=0.25)

        axes[1, column].plot(
            epochs,
            [100 * row["train_accuracy"] for row in history],
            label="Treino",
        )
        axes[1, column].plot(
            epochs,
            [100 * row["val_accuracy"] for row in history],
            label="Validação",
        )
        axes[1, column].set_title(f"{LABELS[model_name]} — Acurácia")
        axes[1, column].set_xlabel("Época")
        axes[1, column].set_ylabel("Acurácia (%)")
        axes[1, column].set_ylim(0, 100)
        axes[1, column].legend()
        axes[1, column].grid(alpha=0.25)
    fig.suptitle("Evolução do treinamento e generalização", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_metric_comparison(metrics: dict[str, dict[str, object]]) -> None:
    names = list(TARGETS)
    positions = np.arange(len(names))
    width = 0.34
    fig, axis = plt.subplots(figsize=(11, 6))
    for offset, model_name in zip((-width / 2, width / 2), MODELS):
        values = [float(metrics[model_name][name]) for name in names]
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=LABELS[model_name],
        )
        axis.bar_label(bars, fmt="%.3f", padding=3)
    for index, name in enumerate(names):
        axis.hlines(
            TARGETS[name],
            index - 0.45,
            index + 0.45,
            colors="black",
            linestyles="dashed",
            linewidth=1.2,
        )
    axis.set_xticks(positions, [METRIC_LABELS[name] for name in names])
    axis.set_ylabel("Pontuação (0–1)")
    axis.set_ylim(0, 1.08)
    axis.set_title("Comparação no conjunto de teste — tracejado indica a meta")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def plot_per_class_f1(metrics: dict[str, dict[str, object]]) -> None:
    positions = np.arange(len(CLASS_NAMES))
    width = 0.38
    fig, axis = plt.subplots(figsize=(12, 6))
    for offset, model_name in zip((-width / 2, width / 2), MODELS):
        report = metrics[model_name]["classification_report"]
        values = [100 * float(report[name]["f1-score"]) for name in CLASS_NAMES]
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=LABELS[model_name],
        )
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.set_xticks(positions, CLASS_NAMES)
    axis.set_xlabel("Classe HAM10000")
    axis.set_ylabel("F1-score (%)")
    axis.set_ylim(0, 100)
    axis.set_title("F1-score por classe no conjunto de teste")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_class_f1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = {
        name: _load_json(ROOT / "models" / name / "history.json")
        for name in MODELS
    }
    metrics = {
        name: _load_json(ROOT / "reports" / name / "metrics.json")
        for name in MODELS
    }
    plot_training_curves(histories)
    plot_metric_comparison(metrics)
    plot_per_class_f1(metrics)
    print(f"Gráficos comparativos salvos em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
