"""
Comparação side-by-side de todos os modelos treinados no HAM10000.

Suporta três famílias timm (EfficientNet / ResNet / MobileNet) e YOLOv8-cls.
Modelos sem checkpoint são silenciosamente ignorados — treine apenas os que
quiser comparar e execute este script para ver o resultado consolidado.

═══════════════════════════════════════════════════════════════════════════════
 COMO TREINAR CADA MODELO ANTES DE COMPARAR
═══════════════════════════════════════════════════════════════════════════════

 EfficientNet (B0→B7 via src/train.py):
   python src/train.py --model efficientnet_b0 --img-size 224 --batch-size 64
   python src/train.py --model efficientnet_b1 --img-size 240 --batch-size 48
   python src/train.py --model efficientnet_b2 --img-size 260 --batch-size 48
   python src/train.py --model efficientnet_b3 --img-size 300 --batch-size 32  # ★ spec
   python src/train.py --model efficientnet_b4 --img-size 380 --batch-size 16
   python src/train.py --model efficientnet_b5 --img-size 456 --batch-size  8
   python src/train.py --model efficientnet_b6 --img-size 528 --batch-size  4
   python src/train.py --model efficientnet_b7 --img-size 600 --batch-size  2

 ResNet (via src/train.py):
   python src/train.py --model resnet18  --batch-size 64   # ★ baseline spec
   python src/train.py --model resnet34  --batch-size 64
   python src/train.py --model resnet50  --batch-size 32
   python src/train.py --model resnet101 --batch-size 16
   python src/train.py --model resnet152 --batch-size  8

 MobileNet (via src/train.py):
   python src/train.py --model mobilenetv2_100       --batch-size 128
   python src/train.py --model mobilenetv3_small_100 --batch-size 128
   python src/train.py --model mobilenetv3_large_100 --batch-size  64
   python src/train.py --model mobilenetv3_large_125 --batch-size  64

 YOLOv8-cls (via src/train_yolo.py — API Ultralytics separada):
   python src/train_yolo.py --model yolov8n --batch-size 128   # nano
   python src/train_yolo.py --model yolov8s --batch-size  64   # small
   python src/train_yolo.py --model yolov8m --batch-size  32   # ★ medium (padrão)
   python src/train_yolo.py --model yolov8l --batch-size  16   # large
   python src/train_yolo.py --model yolov8x --batch-size   8   # extra-large

 Após treinar qualquer combinação, execute:
   python src/compare_models.py

 Saída salva em outputs/comparison/

═══════════════════════════════════════════════════════════════════════════════
 NOTA SOBRE ORDENAÇÃO DE CLASSES — YOLO vs. timm
═══════════════════════════════════════════════════════════════════════════════

 O Ultralytics ordena as classes pelo nome da pasta (ordem alfabética):
   YOLO:  0=akiec  1=bcc  2=bkl  3=df  4=mel  5=nv  6=vasc
   timm:  0=mel    1=nv   2=bcc  3=akiec  4=bkl  5=df  6=vasc

 Este script remapeia automaticamente as probabilidades YOLO para a ordem
 padrão do projeto, garantindo que as métricas sejam comparáveis.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from dataset import CLASSES, CLASS_TO_IDX, HAM10000Dataset, get_dataloaders
from evaluate import predict

OUTPUT_DIR = Path("outputs/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Catálogo completo de modelos.
#
# Formato: (model_name, checkpoint_path, família, params_M, img_size, tipo)
#   tipo = "timm"  → carregado via timm + state_dict (.pth)
#   tipo = "yolo"  → carregado via Ultralytics YOLO (.pt); checkpoint é o
#                    diretório base, o script adiciona /weights/best.pt
#
# Apenas modelos com checkpoint existente são avaliados.
# ─────────────────────────────────────────────────────────────────────────────
MODELS: list[tuple[str, str, str, float, int, str]] = [
    # ── EfficientNet ──────────────────────────────────────────────────────────
    ("efficientnet_b0", "checkpoints/efficientnet_b0_best.pth", "EfficientNet",  5.3, 224, "timm"),
    ("efficientnet_b1", "checkpoints/efficientnet_b1_best.pth", "EfficientNet",  7.8, 240, "timm"),
    ("efficientnet_b2", "checkpoints/efficientnet_b2_best.pth", "EfficientNet",  9.1, 260, "timm"),
    ("efficientnet_b3", "checkpoints/efficientnet_b3_best.pth", "EfficientNet", 12.2, 300, "timm"),  # ★ spec
    ("efficientnet_b4", "checkpoints/efficientnet_b4_best.pth", "EfficientNet", 19.3, 380, "timm"),
    ("efficientnet_b5", "checkpoints/efficientnet_b5_best.pth", "EfficientNet", 30.4, 456, "timm"),
    ("efficientnet_b6", "checkpoints/efficientnet_b6_best.pth", "EfficientNet", 43.0, 528, "timm"),
    ("efficientnet_b7", "checkpoints/efficientnet_b7_best.pth", "EfficientNet", 66.3, 600, "timm"),

    # ── ResNet ────────────────────────────────────────────────────────────────
    ("resnet18",  "checkpoints/resnet18_best.pth",  "ResNet", 11.7, 224, "timm"),  # ★ baseline spec
    ("resnet34",  "checkpoints/resnet34_best.pth",  "ResNet", 21.8, 224, "timm"),
    ("resnet50",  "checkpoints/resnet50_best.pth",  "ResNet", 25.6, 224, "timm"),
    ("resnet101", "checkpoints/resnet101_best.pth", "ResNet", 44.5, 224, "timm"),
    ("resnet152", "checkpoints/resnet152_best.pth", "ResNet", 60.2, 224, "timm"),

    # ── MobileNet ─────────────────────────────────────────────────────────────
    ("mobilenetv2_100",       "checkpoints/mobilenetv2_100_best.pth",       "MobileNet", 3.4, 224, "timm"),
    ("mobilenetv3_small_100", "checkpoints/mobilenetv3_small_100_best.pth", "MobileNet", 2.5, 224, "timm"),
    ("mobilenetv3_large_100", "checkpoints/mobilenetv3_large_100_best.pth", "MobileNet", 5.5, 224, "timm"),
    ("mobilenetv3_large_125", "checkpoints/mobilenetv3_large_125_best.pth", "MobileNet", 7.5, 224, "timm"),

    # ── YOLOv8-cls ────────────────────────────────────────────────────────────
    # checkpoint_path aponta para o diretório do projeto YOLO;
    # o script resolve automaticamente para weights/best.pt.
    # Treine com: python src/train_yolo.py --model yolov8<n|s|m|l|x>
    ("yolov8n", "checkpoints/yolo_yolov8n", "YOLOv8",  2.7, 224, "yolo"),
    ("yolov8s", "checkpoints/yolo_yolov8s", "YOLOv8",  6.4, 224, "yolo"),
    ("yolov8m", "checkpoints/yolo_yolov8m", "YOLOv8", 17.0, 224, "yolo"),  # ★ padrão train_yolo.py
    ("yolov8l", "checkpoints/yolo_yolov8l", "YOLOv8", 37.5, 224, "yolo"),
    ("yolov8x", "checkpoints/yolo_yolov8x", "YOLOv8", 57.4, 224, "yolo"),
]

FAMILY_COLORS = {
    "EfficientNet": "#2196F3",
    "ResNet":       "#4CAF50",
    "MobileNet":    "#FF9800",
    "YOLOv8":       "#9C27B0",
}


# ─────────────────────────────────────────────────────────────────────────────
# Funções de carregamento e predição
# ─────────────────────────────────────────────────────────────────────────────

def load_timm_model(model_name: str, checkpoint: str, device: torch.device) -> torch.nn.Module:
    """Carrega modelo timm a partir de checkpoint .pth gerado pelo train.py."""
    model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device)


def predict_yolo(
    yolo_project_dir: str,
    data_dir: str = "data/organized",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carrega um modelo YOLOv8-cls e gera predições no conjunto de teste.

    O Ultralytics ordena classes alfabeticamente; esta função remapeia as
    probabilidades para a ordem padrão do projeto (CLASSES em dataset.py).

    Args:
        yolo_project_dir: Caminho base do projeto YOLO, ex.:
                          'checkpoints/yolo_yolov8m'. O peso é resolvido
                          como '<yolo_project_dir>/weights/best.pt'.
        data_dir:         Raiz do dataset organizado (train/val/test/).

    Returns:
        (y_true, y_pred, y_probs) — arrays alinhados com a ordem de CLASSES.
    """
    from ultralytics import YOLO  # importação local para não exigir ultralytics quando não usado

    best_pt = Path(yolo_project_dir) / "weights" / "best.pt"
    yolo = YOLO(str(best_pt))

    # Remapeamento: índice YOLO (alfabético) → índice do projeto
    yolo_names: dict[int, str] = yolo.names  # ex.: {0: 'akiec', 1: 'bcc', ...}
    remap = {yolo_idx: CLASS_TO_IDX[name] for yolo_idx, name in yolo_names.items()}

    test_ds = HAM10000Dataset(data_dir, split="test")
    all_labels, all_preds, all_probs = [], [], []

    for img_path, true_label in test_ds.samples:
        result = yolo.predict(str(img_path), verbose=False, imgsz=224)[0]
        yolo_probs = result.probs.data.cpu().numpy()

        # Reordena probabilidades para a convenção do projeto
        reordered = np.zeros(len(CLASSES))
        for yolo_idx, proj_idx in remap.items():
            reordered[proj_idx] = yolo_probs[yolo_idx]

        all_labels.append(true_label)
        all_preds.append(int(reordered.argmax()))
        all_probs.append(reordered)

    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Métricas e plots
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> dict:
    """Calcula Accuracy, F1-macro, F1-weighted e AUC-ROC macro."""
    y_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
    return {
        "Accuracy":    accuracy_score(y_true, y_pred),
        "F1 Macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "F1 Weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "AUC Macro":   roc_auc_score(y_bin, y_probs, average="macro"),
    }


def plot_comparison(results: dict[str, dict], families: dict[str, str]) -> None:
    """Gráfico de barras agrupado por modelo, colorido por família."""
    metric_names = list(next(iter(results.values())).keys())
    model_names = list(results.keys())
    n_models = len(model_names)
    x = np.arange(len(metric_names))
    width = min(0.8 / n_models, 0.12)

    fig, ax = plt.subplots(figsize=(max(14, n_models * 2), 7))
    for i, name in enumerate(model_names):
        color = FAMILY_COLORS.get(families.get(name, ""), "#9E9E9E")
        values = [results[name][m] for m in metric_names]
        offset = x + (i - n_models / 2 + 0.5) * width
        bars = ax.bar(offset, values, width, label=name, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=6, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Comparação de Modelos — HAM10000 Test Set", fontsize=13)
    ax.axhline(0.85, color="red", linestyle="--", linewidth=0.8, label="Meta AUC ≥ 0.85")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7,
              ncol=max(1, n_models // 5 + 1), loc="upper right", framealpha=0.8)

    plt.tight_layout()
    out = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def plot_params_vs_auc(
    results: dict[str, dict],
    params_map: dict[str, float],
    families: dict[str, str],
) -> None:
    """Scatter: parâmetros × AUC-ROC macro — trade-off tamanho/desempenho."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for family, color in FAMILY_COLORS.items():
        names = [n for n in results if families.get(n) == family]
        if not names:
            continue
        xs = [params_map[n] for n in names]
        ys = [results[n]["AUC Macro"] for n in names]
        ax.scatter(xs, ys, color=color, s=90, label=family, zorder=3)
        for n, xi, yi in zip(names, xs, ys):
            ax.annotate(n, (xi, yi), textcoords="offset points",
                        xytext=(5, 3), fontsize=7)

    ax.axhline(0.85, color="red", linestyle="--", linewidth=0.8, label="Meta AUC ≥ 0.85")
    ax.set_xlabel("Parâmetros (M)", fontsize=11)
    ax.set_ylabel("AUC-ROC Macro", fontsize=11)
    ax.set_title("Trade-off: Tamanho do Modelo × AUC-ROC\n(EfficientNet / ResNet / MobileNet / YOLOv8)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    out = OUTPUT_DIR / "params_vs_auc.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def print_table(results: dict) -> None:
    """Tabela formatada no terminal com flag ✓ para modelos que batem a meta."""
    col_w = max(len(n) for n in results) + 2
    header = (
        f"{'Model':<{col_w}} {'Accuracy':>10} {'F1 Macro':>10} "
        f"{'F1 Weighted':>12} {'AUC Macro':>10}"
    )
    print("\n" + header)
    print("─" * len(header))
    for name, m in results.items():
        flag = " ✓" if m["AUC Macro"] >= 0.85 else "  "
        print(
            f"{name:<{col_w}} {m['Accuracy']:>10.4f} {m['F1 Macro']:>10.4f} "
            f"{m['F1 Weighted']:>12.4f} {m['AUC Macro']:>10.4f}{flag}"
        )
    print("\n✓ = atingiu a meta AUC-ROC ≥ 0.85")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = get_dataloaders("data/organized", batch_size=64)

    results:    dict[str, dict]  = {}
    families:   dict[str, str]   = {}
    params_map: dict[str, float] = {}

    for model_name, ckpt_path, family, n_params, _, model_type in MODELS:

        if model_type == "timm":
            if not Path(ckpt_path).exists():
                print(f"Pulando {model_name:30s} — checkpoint não encontrado: {ckpt_path}")
                continue
            print(f"[timm ] Avaliando {model_name}...")
            model = load_timm_model(model_name, ckpt_path, device)
            y_true, y_pred, y_probs = predict(model, test_loader, device)

        elif model_type == "yolo":
            best_pt = Path(ckpt_path) / "weights" / "best.pt"
            if not best_pt.exists():
                print(f"Pulando {model_name:30s} — checkpoint não encontrado: {best_pt}")
                continue
            print(f"[yolo ] Avaliando {model_name}...")
            y_true, y_pred, y_probs = predict_yolo(ckpt_path)

        else:
            print(f"Tipo desconhecido '{model_type}' para {model_name} — ignorado.")
            continue

        results[model_name]    = compute_metrics(y_true, y_pred, y_probs)
        families[model_name]   = family
        params_map[model_name] = n_params

    if not results:
        print(
            "\nNenhum modelo encontrado. Treine ao menos um modelo:\n"
            "  python src/train.py --model efficientnet_b3        # timm\n"
            "  python src/train.py --model resnet18               # timm baseline\n"
            "  python src/train_yolo.py --model yolov8m           # YOLO\n"
        )
        return

    print_table(results)
    plot_comparison(results, families)

    if len(results) > 1:
        plot_params_vs_auc(results, params_map, families)


if __name__ == "__main__":
    main()
