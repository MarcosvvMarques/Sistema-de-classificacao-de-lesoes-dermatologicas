"""Executa os experimentos pragmáticos da Entrega 2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from skin_lesions.advanced_evaluation import run_advanced_evaluation  # noqa: E402
from skin_lesions.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    MODELS_DIR,
    REPORTS_DIR,
    TrainConfig,
    get_device,
)
from skin_lesions.data import build_dataloaders, class_weights  # noqa: E402
from skin_lesions.engine import load_model_from_checkpoint, train_model  # noqa: E402
from skin_lesions.evaluation import calculate_metrics, evaluate_model  # noqa: E402

SPLITS_PATH = ARTIFACTS_DIR / "splits.csv"
EXPERIMENT_MODELS = MODELS_DIR / "experiments"
EXPERIMENT_REPORTS = REPORTS_DIR / "experiments"


def load_frame() -> pd.DataFrame:
    return pd.read_csv(SPLITS_PATH)


def make_loaders(frame: pd.DataFrame, config: TrainConfig):
    return build_dataloaders(
        frame,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        imbalance_strategy=config.imbalance_strategy,
    )


def train_or_load(
    name: str,
    frame: pd.DataFrame,
    config: TrainConfig,
    device: torch.device,
):
    model_dir = EXPERIMENT_MODELS / name
    checkpoint_path = model_dir / "best.pt"
    loaders = make_loaders(frame, config)
    started = time.perf_counter()
    if checkpoint_path.exists() and (model_dir / "history.json").exists():
        model, _ = load_model_from_checkpoint(checkpoint_path, device)
        reused = True
    else:
        weights = None
        if config.imbalance_strategy == "class_weights" or config.loss_name == "focal":
            weights = class_weights(frame[frame["split"] == "train"])
        model, _ = train_model(
            config,
            loaders,
            device,
            model_dir,
            loss_weights=weights,
        )
        reused = False
    elapsed = time.perf_counter() - started
    summary = {
        "name": name,
        "reused_checkpoint": reused,
        "runtime_seconds_this_invocation": elapsed,
        "config": asdict(config),
    }
    (model_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return model, loaders


def run_advanced(device: torch.device) -> None:
    frame = load_frame()
    for model_name in ("resnet18", "efficientnet_b3"):
        checkpoint = MODELS_DIR / model_name / "best.pt"
        model, _ = load_model_from_checkpoint(checkpoint, device)
        config = TrainConfig(
            model_name=model_name,
            batch_size=128 if model_name == "resnet18" else 64,
        )
        loaders = make_loaders(frame, config)
        run_advanced_evaluation(
            model,
            loaders["val"],
            loaders["test"],
            frame,
            device,
            REPORTS_DIR / model_name / "advanced",
        )
        print(f"Validação estatística concluída: {model_name}")


def _predictions_to_arrays(predictions: pd.DataFrame):
    labels = predictions["true_label"].map(CLASS_TO_IDX).to_numpy()
    probabilities = predictions[
        [f"prob_{name}" for name in CLASS_NAMES]
    ].to_numpy()
    return labels, probabilities


def run_cross_validation(device: torch.device) -> None:
    frame = load_frame()
    development = frame[frame["split"] != "test"].copy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics: list[dict[str, float]] = []
    predictions: list[pd.DataFrame] = []

    for fold, (_train_indices, val_indices) in enumerate(
        splitter.split(
            development,
            development["label"],
            groups=development["lesion_id"],
        ),
        start=1,
    ):
        fold_frame = frame.copy()
        fold_frame.loc[fold_frame["split"] != "test", "split"] = "train"
        validation_lesions = set(development.iloc[val_indices]["lesion_id"])
        fold_frame.loc[
            fold_frame["lesion_id"].isin(validation_lesions),
            "split",
        ] = "val"
        config = TrainConfig(
            model_name="resnet18",
            epochs=10,
            batch_size=128,
            patience=3,
            num_workers=0,
            selection_metric="auc_roc_melanoma",
        )
        name = f"cv_resnet18/fold_{fold}"
        model, loaders = train_or_load(name, fold_frame, config, device)
        report_dir = EXPERIMENT_REPORTS / name
        metrics = evaluate_model(model, loaders["val"], device, report_dir)
        fold_metrics.append(
            {
                "fold": fold,
                "auc_roc_melanoma": float(metrics["auc_roc_melanoma"]),
                "auc_roc_macro": float(metrics["auc_roc_macro"]),
                "f1_macro": float(metrics["f1_macro"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
            }
        )
        fold_predictions = pd.read_csv(report_dir / "predictions.csv")
        fold_predictions["fold"] = fold
        predictions.append(fold_predictions)
        print(f"Fold {fold}/5 concluído.")

    combined = pd.concat(predictions, ignore_index=True)
    labels, probabilities = _predictions_to_arrays(combined)
    aggregate = calculate_metrics(labels, probabilities)
    numeric_names = (
        "auc_roc_melanoma",
        "auc_roc_macro",
        "f1_macro",
        "balanced_accuracy",
    )
    summary = {
        "folds": fold_metrics,
        "out_of_fold_metrics": aggregate,
        "mean": {
            name: float(np.mean([fold[name] for fold in fold_metrics]))
            for name in numeric_names
        },
        "std": {
            name: float(np.std([fold[name] for fold in fold_metrics], ddof=1))
            for name in numeric_names
        },
    }
    output_dir = EXPERIMENT_REPORTS / "cv_resnet18"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def run_ablation(device: torch.device) -> None:
    frame = load_frame()
    experiments = {
        "weighted_loss": TrainConfig(
            model_name="resnet18",
            epochs=10,
            batch_size=128,
            patience=3,
            imbalance_strategy="class_weights",
            selection_metric="auc_roc_melanoma",
        ),
        "weighted_sampler": TrainConfig(
            model_name="resnet18",
            epochs=10,
            batch_size=128,
            patience=3,
            imbalance_strategy="sampler",
            selection_metric="auc_roc_melanoma",
        ),
        "focal_loss": TrainConfig(
            model_name="resnet18",
            epochs=10,
            batch_size=128,
            patience=3,
            imbalance_strategy="class_weights",
            loss_name="focal",
            focal_gamma=2.0,
            selection_metric="auc_roc_melanoma",
        ),
    }
    results = {}
    for name, config in experiments.items():
        experiment_name = f"ablation_resnet18/{name}"
        model, loaders = train_or_load(experiment_name, frame, config, device)
        results[name] = evaluate_model(
            model,
            loaders["test"],
            device,
            EXPERIMENT_REPORTS / experiment_name,
        )
        print(f"Ablação concluída: {name}")
    output_dir = EXPERIMENT_REPORTS / "ablation_resnet18"
    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


def run_efficientnet_tuning(device: torch.device) -> None:
    frame = load_frame()
    config = TrainConfig(
        model_name="efficientnet_b3",
        epochs=15,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=5e-4,
        patience=4,
        imbalance_strategy="class_weights",
        selection_metric="auc_roc_melanoma",
        gradual_unfreeze=True,
        drop_rate=0.3,
    )
    name = "efficientnet_b3_tuned"
    model, loaders = train_or_load(name, frame, config, device)
    metrics = evaluate_model(
        model,
        loaders["test"],
        device,
        EXPERIMENT_REPORTS / name,
    )
    run_advanced_evaluation(
        model,
        loaders["val"],
        loaders["test"],
        frame,
        device,
        EXPERIMENT_REPORTS / name / "advanced",
    )
    print(json.dumps({key: metrics[key] for key in metrics if key != "classification_report"}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("advanced", "cv", "ablation", "efficientnet", "all"),
        default="all",
        nargs="?",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = get_device(args.device)
    stages = (
        ("advanced", "cv", "ablation", "efficientnet")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        {
            "advanced": run_advanced,
            "cv": run_cross_validation,
            "ablation": run_ablation,
            "efficientnet": run_efficientnet_tuning,
        }[stage](device)


if __name__ == "__main__":
    main()
