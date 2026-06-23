"""
Full evaluation suite: confusion matrix, ROC curves, classification report.

Usage:
    python src/evaluate.py --model-name efficientnet_b4 \
        --checkpoint checkpoints/efficientnet_b4_best.pth

Outputs saved to outputs/evaluation/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from dataset import CLASSES, get_dataloaders

OUTPUT_DIR = Path("outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            probs = model(imgs.to(device)).softmax(1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(probs.argmax(1))
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs)


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, fmt in [
        (axes[0], cm,      "Counts",      "d"),
        (axes[1], cm_norm, "Normalized",  ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title(f"{model_name} — Confusion Matrix ({title})")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    out = OUTPUT_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def plot_roc_curves(y_true, y_probs, model_name: str):
    y_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curves (one-vs-rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()

    out = OUTPUT_DIR / f"{model_name}_roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()

    macro_auc = roc_auc_score(y_bin, y_probs, average="macro")
    print(f"Macro AUC-ROC: {macro_auc:.4f}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = get_dataloaders(args.data_dir, batch_size=64)

    model = timm.create_model(args.model_name, pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    epoch = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_acc", 0)
    print(f"Model: {args.model_name} | Epoch: {epoch} | Val acc: {val_acc:.4f}\n")

    y_true, y_pred, y_probs = predict(model, test_loader, device)

    print(f"Test Accuracy: {(y_true == y_pred).mean():.4f}\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    plot_confusion_matrix(y_true, y_pred, args.model_name)
    plot_roc_curves(y_true, y_probs, args.model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="efficientnet_b4")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/organized")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
