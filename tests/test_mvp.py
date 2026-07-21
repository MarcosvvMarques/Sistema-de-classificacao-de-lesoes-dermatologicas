from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skin_lesions.config import CLASS_NAMES
from skin_lesions.advanced_evaluation import (
    expected_calibration_error,
    grouped_bootstrap_intervals,
    select_melanoma_threshold,
)
from skin_lesions.data import HAM10000Dataset, create_grouped_splits
from skin_lesions.engine import FocalLoss
from skin_lesions.evaluation import calculate_metrics
from skin_lesions.explain import GradCAM
from skin_lesions.models import create_model, get_gradcam_target_layer


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        for lesion_index in range(20):
            lesion_id = f"{class_name}_{lesion_index}"
            for image_index in range(1 + lesion_index % 2):
                rows.append(
                    {
                        "lesion_id": lesion_id,
                        "image_id": f"{lesion_id}_{image_index}",
                        "dx": class_name,
                        "label": class_index,
                        "image_path": f"/tmp/{lesion_id}_{image_index}.jpg",
                    }
                )
    return pd.DataFrame(rows)


def test_grouped_split_has_no_lesion_leakage() -> None:
    result = create_grouped_splits(_synthetic_frame(), seed=42)
    assert set(result["split"]) == {"train", "val", "test"}
    assert result.groupby("lesion_id")["split"].nunique().max() == 1
    assert set(result.groupby("split")["dx"].nunique()) == {len(CLASS_NAMES)}


def test_dataset_reads_and_transforms_rgb_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image = np.full((40, 50, 3), 127, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    encoded.tofile(image_path)
    frame = pd.DataFrame(
        [
            {
                "image_id": "sample",
                "image_path": str(image_path),
                "label": 4,
            }
        ]
    )
    item = HAM10000Dataset(frame, image_size=32, training=False)[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["label"].item() == 4


def test_metrics_are_perfect_for_perfect_predictions() -> None:
    y_true = np.repeat(np.arange(len(CLASS_NAMES)), 2)
    probabilities = np.eye(len(CLASS_NAMES))[y_true]
    metrics = calculate_metrics(y_true, probabilities)
    assert metrics["auc_roc_melanoma"] == pytest.approx(1.0)
    assert metrics["auc_roc_macro"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert all(metrics["targets_met"].values())


@pytest.mark.parametrize("model_name", ["resnet18", "efficientnet_b3"])
def test_model_forward_has_seven_logits(model_name: str) -> None:
    model = create_model(model_name, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, len(CLASS_NAMES))


@pytest.mark.parametrize("method", ["gradcam", "gradcam++"])
def test_gradcam_generates_normalized_map(method: str) -> None:
    model = create_model("resnet18", pretrained=False).eval()
    target_layer = get_gradcam_target_layer(model, "resnet18")
    with GradCAM(model, target_layer, method=method) as explainer:
        cam, predicted, confidence = explainer(torch.randn(1, 3, 64, 64))
    assert cam.shape == (64, 64)
    assert 0 <= cam.min() <= cam.max() <= 1
    assert 0 <= predicted < len(CLASS_NAMES)
    assert 0 <= confidence <= 1


def test_threshold_reaches_target_sensitivity() -> None:
    labels = np.array([4, 4, 4, 0, 0, 0, 0])
    melanoma_probabilities = np.array([0.9, 0.7, 0.4, 0.5, 0.3, 0.2, 0.1])
    result = select_melanoma_threshold(
        labels,
        melanoma_probabilities,
        target_sensitivity=2 / 3,
    )
    assert result["validation_sensitivity"] >= 2 / 3
    assert 0 <= result["threshold"] <= 1


def test_calibration_error_is_zero_for_perfect_confidence() -> None:
    labels = np.arange(len(CLASS_NAMES))
    probabilities = np.eye(len(CLASS_NAMES))
    assert expected_calibration_error(labels, probabilities) == pytest.approx(0.0)


def test_grouped_bootstrap_returns_intervals() -> None:
    labels = np.repeat(np.arange(len(CLASS_NAMES)), 6)
    probabilities = np.eye(len(CLASS_NAMES))[labels] * 0.9 + 0.1 / len(CLASS_NAMES)
    groups = np.array([f"group-{index}" for index in range(len(labels))])
    intervals = grouped_bootstrap_intervals(
        labels,
        probabilities,
        groups,
        iterations=30,
    )
    assert intervals["auc_roc_melanoma"]["estimate"] == pytest.approx(1.0)
    assert intervals["f1_macro"]["lower_95"] == pytest.approx(1.0)


def test_focal_loss_is_finite() -> None:
    criterion = FocalLoss(gamma=2.0)
    loss = criterion(torch.randn(8, len(CLASS_NAMES)), torch.arange(8) % len(CLASS_NAMES))
    assert torch.isfinite(loss)
