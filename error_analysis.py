"""Analisa erros de melanoma por metadados e prepara revisão especializada."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from skin_lesions.config import ARTIFACTS_DIR, CLASS_NAMES, get_device  # noqa: E402
from skin_lesions.data import HAM10000Dataset  # noqa: E402
from skin_lesions.engine import load_model_from_checkpoint  # noqa: E402
from skin_lesions.explain import save_explanations  # noqa: E402

MODEL_NAME = "efficientnet_b3"
EXPERIMENT = "efficientnet_b3_tuned"
REPORT_DIR = ROOT / "reports" / "experiments" / EXPERIMENT
FINAL_DIR = ROOT / "reports" / "final"
PROTOCOL_DIR = ROOT / "docs" / "protocols"


def _calibrated_melanoma_probability(
    predictions: pd.DataFrame,
    temperature: float,
) -> np.ndarray:
    columns = [f"prob_{name}" for name in CLASS_NAMES]
    raw = predictions[columns].to_numpy().clip(1e-12, 1.0)
    scaled = np.power(raw, 1.0 / temperature)
    calibrated = scaled / scaled.sum(axis=1, keepdims=True)
    return calibrated[:, CLASS_NAMES.index("mel")]


def _group_statistics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(column, dropna=False):
        tp = int(((group["actual_melanoma"]) & (group["predicted_melanoma"])).sum())
        fn = int(((group["actual_melanoma"]) & (~group["predicted_melanoma"])).sum())
        fp = int(((~group["actual_melanoma"]) & (group["predicted_melanoma"])).sum())
        tn = int(((~group["actual_melanoma"]) & (~group["predicted_melanoma"])).sum())
        positives = tp + fn
        negatives = tn + fp
        rows.append(
            {
                "subgroup": str(value),
                "samples": len(group),
                "melanoma_cases": positives,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "sensitivity": tp / positives if positives >= 5 else np.nan,
                "specificity": tn / negatives if negatives >= 20 else np.nan,
                "precision": tp / (tp + fp) if tp + fp >= 5 else np.nan,
                "interpretation_allowed": bool(len(group) >= 30 and positives >= 5),
            }
        )
    return pd.DataFrame(rows).sort_values("samples", ascending=False)


def _plot_subgroups(statistics: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, dimension in zip(axes, ("sex", "age_group")):
        data = statistics[dimension]
        valid = data[data["interpretation_allowed"]].copy()
        positions = np.arange(len(valid))
        axis.barh(positions - 0.18, valid["sensitivity"] * 100, 0.36, label="Sensibilidade")
        axis.barh(positions + 0.18, valid["specificity"] * 100, 0.36, label="Especificidade")
        axis.set_yticks(positions, valid["subgroup"])
        axis.set_xlim(0, 100)
        axis.set_xlabel("Pontuação (%)")
        axis.set_title(f"Melanoma por {dimension}")
        axis.grid(axis="x", alpha=0.25)
        axis.legend()
    fig.suptitle("Desempenho do limiar de triagem por subgrupo")
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "subgroup_performance.png", dpi=180)
    plt.close(fig)


def _prepare_review_cases(frame: pd.DataFrame) -> pd.DataFrame:
    false_negatives = (
        frame[frame["actual_melanoma"] & ~frame["predicted_melanoma"]]
        .sort_values("melanoma_probability")
        .head(15)
    )
    false_positives = (
        frame[~frame["actual_melanoma"] & frame["predicted_melanoma"]]
        .sort_values("melanoma_probability", ascending=False)
        .head(15)
    )
    cases = pd.concat(
        [
            false_negatives.assign(error_type="FN"),
            false_positives.assign(error_type="FP"),
        ],
        ignore_index=True,
    )
    cases.insert(0, "case_id", [f"CASE-{index:03d}" for index in range(1, len(cases) + 1)])
    return cases


def _generate_error_gradcams(cases: pd.DataFrame, device: torch.device) -> list[Path]:
    checkpoint = ROOT / "models" / "experiments" / EXPERIMENT / "best.pt"
    model, _ = load_model_from_checkpoint(checkpoint, device)
    dataset = HAM10000Dataset(cases, image_size=300, training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return save_explanations(
        model,
        MODEL_NAME,
        loader,
        device,
        FINAL_DIR / "error_gradcam",
        method="gradcam++",
        max_images=len(cases),
    )


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(REPORT_DIR / "predictions.csv")
    metadata = pd.read_csv(ARTIFACTS_DIR / "splits.csv")
    advanced = json.loads(
        (REPORT_DIR / "advanced" / "advanced_metrics.json").read_text(encoding="utf-8")
    )
    temperature = float(advanced["temperature"])
    threshold = float(advanced["threshold_validation"]["threshold"])

    frame = predictions.merge(metadata, on="image_id", how="left", validate="one_to_one")
    frame["melanoma_probability"] = _calibrated_melanoma_probability(
        frame,
        temperature,
    )
    frame["actual_melanoma"] = frame["true_label"] == "mel"
    frame["predicted_melanoma"] = frame["melanoma_probability"] >= threshold
    frame["age_group"] = pd.cut(
        frame["age"],
        bins=[-np.inf, 39, 59, 79, np.inf],
        labels=["<40", "40–59", "60–79", "80+"],
    ).astype("string").fillna("desconhecida")
    frame["sex"] = frame["sex"].fillna("unknown")
    frame["localization"] = frame["localization"].fillna("unknown")

    statistics = {
        dimension: _group_statistics(frame, dimension)
        for dimension in ("sex", "age_group", "localization")
    }
    for dimension, table in statistics.items():
        table.to_csv(FINAL_DIR / f"errors_by_{dimension}.csv", index=False)
    _plot_subgroups(statistics)

    cases = _prepare_review_cases(frame)
    device = get_device("auto")
    paths = _generate_error_gradcams(cases, device)
    path_by_image = {path.name.split("_real-")[0]: str(path) for path in paths}
    cases["gradcam_path"] = cases["image_id"].map(path_by_image)
    answer_key_columns = [
        "case_id",
        "image_id",
        "error_type",
        "true_label",
        "melanoma_probability",
        "age",
        "sex",
        "localization",
        "gradcam_path",
    ]
    cases[answer_key_columns].to_csv(
        PROTOCOL_DIR / "specialist_review_answer_key.csv",
        index=False,
    )
    blinded = cases[["case_id", "image_id", "gradcam_path"]].copy()
    blinded["heatmap_anatomically_plausible_1_to_5"] = ""
    blinded["focuses_lesion_not_artifact_yes_no"] = ""
    blinded["image_quality_1_to_5"] = ""
    blinded["comments"] = ""
    blinded.to_csv(PROTOCOL_DIR / "specialist_review_blinded.csv", index=False)
    print(f"Análise de erros concluída com {len(cases)} casos para revisão.")


if __name__ == "__main__":
    main()
