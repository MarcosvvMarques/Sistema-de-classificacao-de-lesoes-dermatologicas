"""CLI do pipeline end-to-end do HAM10000."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from skin_lesions.config import (  # noqa: E402
    ARTIFACTS_DIR,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TrainConfig,
    ensure_output_dirs,
    get_device,
)
from skin_lesions.data import (  # noqa: E402
    build_dataloaders,
    class_weights,
    prepare_splits,
    validate_splits,
)
from skin_lesions.engine import load_model_from_checkpoint, train_model  # noqa: E402
from skin_lesions.evaluation import evaluate_model  # noqa: E402
from skin_lesions.explain import save_explanations  # noqa: E402

DEFAULT_SPLITS = ARTIFACTS_DIR / "splits.csv"
MODEL_CHOICES = ("resnet18", "efficientnet_b3")


def _load_splits(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Execute primeiro: python pipeline.py prepare"
        )
    frame = pd.read_csv(path)
    validate_splits(frame)
    return frame


def _make_loaders(frame: pd.DataFrame, config: TrainConfig):
    return build_dataloaders(
        frame,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        imbalance_strategy=config.imbalance_strategy,
    )


def _train_one(args: argparse.Namespace, model_name: str):
    frame = _load_splits(args.splits)
    config = TrainConfig(
        model_name=model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        imbalance_strategy=args.imbalance_strategy,
    )
    device = get_device(args.device)
    loaders = _make_loaders(frame, config)
    weights = None
    if config.imbalance_strategy == "class_weights":
        weights = class_weights(frame[frame["split"] == "train"])
    model, _history = train_model(
        config,
        loaders,
        device=device,
        output_dir=MODELS_DIR / model_name,
        loss_weights=weights,
    )
    return model, loaders, device


def command_prepare(args: argparse.Namespace) -> None:
    frame = prepare_splits(args.data_dir, args.output, seed=args.seed)
    counts = frame.groupby(["split", "dx"]).size().unstack(fill_value=0)
    print(f"Splits salvos em {args.output}\n{counts}")


def command_train(args: argparse.Namespace) -> None:
    _train_one(args, args.model)
    print(f"Melhor checkpoint salvo em {MODELS_DIR / args.model / 'best.pt'}")


def command_evaluate(args: argparse.Namespace) -> None:
    frame = _load_splits(args.splits)
    device = get_device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    config = TrainConfig(
        model_name=str(checkpoint["model_name"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loaders = _make_loaders(frame, config)
    metrics = evaluate_model(
        model,
        loaders["test"],
        device,
        REPORTS_DIR / config.model_name,
    )
    summary = {
        key: value
        for key, value in metrics.items()
        if key != "classification_report"
    }
    print(json.dumps(summary, indent=2))


def command_explain(args: argparse.Namespace) -> None:
    frame = _load_splits(args.splits)
    device = get_device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    model_name = str(checkpoint["model_name"])
    config = TrainConfig(
        model_name=model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loaders = _make_loaders(frame, config)
    methods = ("gradcam", "gradcam++") if args.method == "both" else (args.method,)
    for method in methods:
        paths = save_explanations(
            model,
            model_name,
            loaders["test"],
            device,
            REPORTS_DIR / model_name / "explanations" / method,
            method=method,
            max_images=args.max_images,
        )
        print(f"{len(paths)} explicações {method} salvas.")


def command_all(args: argparse.Namespace) -> None:
    if not args.splits.exists():
        prepare_splits(args.data_dir, args.splits, seed=args.seed)
    for model_name in MODEL_CHOICES:
        print(f"\n=== Treinando {model_name} ===")
        model, loaders, device = _train_one(args, model_name)
        metrics = evaluate_model(
            model,
            loaders["test"],
            device,
            REPORTS_DIR / model_name,
        )
        print(
            f"{model_name}: AUC melanoma={metrics['auc_roc_melanoma']:.4f}, "
            f"F1 macro={metrics['f1_macro']:.4f}"
        )
        for method in ("gradcam", "gradcam++"):
            save_explanations(
                model,
                model_name,
                loaders["test"],
                device,
                REPORTS_DIR / model_name / "explanations" / method,
                method=method,
                max_images=args.max_images,
            )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    _add_runtime_arguments(parser)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument(
        "--imbalance-strategy",
        choices=("class_weights", "sampler", "none"),
        default="class_weights",
    )
    parser.add_argument("--no-pretrained", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Validar dados e criar splits")
    prepare.add_argument("--data-dir", type=Path, default=DATA_DIR)
    prepare.add_argument("--output", type=Path, default=DEFAULT_SPLITS)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(function=command_prepare)

    train = subparsers.add_parser("train", help="Treinar um modelo")
    train.add_argument("--model", choices=MODEL_CHOICES, required=True)
    _add_training_arguments(train)
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Avaliar um checkpoint")
    evaluate.add_argument("--model", choices=MODEL_CHOICES, required=True)
    evaluate.add_argument("--checkpoint", type=Path)
    _add_runtime_arguments(evaluate)
    evaluate.set_defaults(function=command_evaluate)

    explain = subparsers.add_parser("explain", help="Gerar mapas Grad-CAM")
    explain.add_argument("--model", choices=MODEL_CHOICES, required=True)
    explain.add_argument("--checkpoint", type=Path)
    explain.add_argument(
        "--method",
        choices=("gradcam", "gradcam++", "both"),
        default="both",
    )
    explain.add_argument("--max-images", type=int, default=12)
    _add_runtime_arguments(explain)
    explain.set_defaults(function=command_explain)

    all_commands = subparsers.add_parser("all", help="Executar o MVP completo")
    all_commands.add_argument("--data-dir", type=Path, default=DATA_DIR)
    all_commands.add_argument("--max-images", type=int, default=12)
    _add_training_arguments(all_commands)
    all_commands.set_defaults(function=command_all)
    return parser


def main() -> None:
    ensure_output_dirs()
    parser = build_parser()
    args = parser.parse_args()
    if args.command in {"evaluate", "explain"} and args.checkpoint is None:
        args.checkpoint = MODELS_DIR / args.model / "best.pt"
    args.function(args)


if __name__ == "__main__":
    main()
