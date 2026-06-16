"""
Treinamento de modelos timm (EfficientNet / ResNet / MobileNet) no HAM10000.

Este script cobre as famílias baseadas em PyTorch/timm.
Para treinar modelos YOLOv8, use src/train_yolo.py (ver seção YOLO abaixo).

═══════════════════════════════════════════════════════════════════════════════
 FAMÍLIA EfficientNet  (escala B0→B7: mais parâmetros, maior acurácia)
═══════════════════════════════════════════════════════════════════════════════

  Modelo              | Params  | Img padrão | Batch 8GB | Observação
  --------------------|---------|------------|-----------|------------------
  efficientnet_b0     |  5.3 M  |   224 px   |    64     | Mais leve da família
  efficientnet_b1     |  7.8 M  |   240 px   |    48     |
  efficientnet_b2     |  9.1 M  |   260 px   |    48     |
  efficientnet_b3     | 12.2 M  |   300 px   |    32     | ★ MODELO PRINCIPAL (spec)
  efficientnet_b4     | 19.3 M  |   380 px   |    16     | Alta acurácia, lento
  efficientnet_b5     | 30.4 M  |   456 px   |     8     | GPU 16 GB+
  efficientnet_b6     | 43.0 M  |   528 px   |     4     | GPU 24 GB+
  efficientnet_b7     | 66.3 M  |   600 px   |     2     | GPU 40 GB+

  Exemplos:
      python src/train.py --model efficientnet_b0 --img-size 224 --batch-size 64
      python src/train.py --model efficientnet_b3 --img-size 300 --batch-size 32
      python src/train.py --model efficientnet_b4 --img-size 380 --batch-size 16

═══════════════════════════════════════════════════════════════════════════════
 FAMÍLIA ResNet  (clássica; residual connections; muito estável)
═══════════════════════════════════════════════════════════════════════════════

  Modelo   | Params  | Img padrão | Batch 8GB | Observação
  ---------|---------|------------|-----------|------------------
  resnet18 | 11.7 M  |   224 px   |    64     | ★ BASELINE (spec) — treina rápido
  resnet34 | 21.8 M  |   224 px   |    64     |
  resnet50 | 25.6 M  |   224 px   |    32     | Boa relação custo/benefício
  resnet101| 44.5 M  |   224 px   |    16     |
  resnet152| 60.2 M  |   224 px   |     8     | Lento; ganho marginal sobre 101

  Exemplos:
      python src/train.py --model resnet18  --epochs 50
      python src/train.py --model resnet50  --epochs 50
      python src/train.py --model resnet101 --batch-size 16

═══════════════════════════════════════════════════════════════════════════════
 FAMÍLIA MobileNet  (otimizados para edge/CPU; menor footprint)
═══════════════════════════════════════════════════════════════════════════════

  Modelo                   | Params | Img padrão | Batch 8GB | Observação
  -------------------------|--------|------------|-----------|------------------
  mobilenetv2_100          |  3.4 M |   224 px   |    128    | Muito leve
  mobilenetv3_small_100    |  2.5 M |   224 px   |    128    | O menor da família
  mobilenetv3_large_100    |  5.5 M |   224 px   |    64     | Equilíbrio leve/acurácia
  mobilenetv3_large_125    |  7.5 M |   224 px   |    64     | Versão ampliada do large

  Exemplos:
      python src/train.py --model mobilenetv2_100       --batch-size 128
      python src/train.py --model mobilenetv3_small_100 --batch-size 128
      python src/train.py --model mobilenetv3_large_100 --batch-size 64

═══════════════════════════════════════════════════════════════════════════════
 FAMÍLIA YOLOv8-cls  ─── use src/train_yolo.py, NÃO este script ───
═══════════════════════════════════════════════════════════════════════════════

  YOLOv8 usa a API Ultralytics, que gerencia o loop de treino internamente.
  Este script (train.py) não suporta YOLO; use train_yolo.py.

  Modelo      | Params  | Img padrão | Batch 8GB | Observação
  ------------|---------|------------|-----------|------------------
  yolov8n-cls |  2.7 M  |   224 px   |    128    | Nano — mais rápido
  yolov8s-cls |  6.4 M  |   224 px   |    64     | Small
  yolov8m-cls | 17.0 M  |   224 px   |    32     | ★ padrão do train_yolo.py
  yolov8l-cls | 37.5 M  |   224 px   |    16     | Large
  yolov8x-cls | 57.4 M  |   224 px   |     8     | Extra-large

  Exemplos:
      python src/train_yolo.py --model yolov8n --epochs 50 --batch-size 128
      python src/train_yolo.py --model yolov8m --epochs 50
      python src/train_yolo.py --model yolov8x --epochs 50 --batch-size 8

  Checkpoint salvo em: checkpoints/yolo_<model>/weights/best.pt
  Compare todos os modelos (timm + YOLO) com: python src/compare_models.py

═══════════════════════════════════════════════════════════════════════════════
 DICAS GERAIS
═══════════════════════════════════════════════════════════════════════════════

  • Modelos B4+ exigem --img-size maior para extrair o máximo de desempenho.
  • Em CPU ou GPU < 8 GB, prefira B0, resnet18 ou mobilenet.
  • Todos os checkpoints vão para checkpoints/<model_name>_best.pth
  • Use compare_models.py para comparar todos os modelos treinados (timm + YOLO).

Uso rápido:
    python src/train.py --model efficientnet_b3 --img-size 300 --batch-size 32
    python src/train.py --model resnet18        --batch-size 64
    python src/train_yolo.py --model yolov8m    --batch-size 32
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import CLASSES, get_dataloaders

SEED = 42

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Fixa todas as sementes para reprodutibilidade determinística."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    """Executa uma época de treino com mixed precision (AMP).

    Retorna:
        (loss_media, acuracia) sobre o epoch inteiro.
    """
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast("cuda"):
            loss = criterion(model(imgs), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = model(imgs).argmax(1)
        total_loss += loss.item() * len(labels)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Avalia o modelo sem gradientes.

    Retorna:
        (loss_media, acuracia) sobre o loader fornecido.
    """
    model.eval()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def train(args) -> None:
    """Pipeline completo de treino: setup → loop de epochs → early stopping → checkpoint.

    O melhor modelo (por val_acc) é salvo em checkpoints/<model_name>_best.pth.
    O arquivo de checkpoint contém: epoch, model_state, val_acc.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _, class_weights = get_dataloaders(
        args.data_dir, args.batch_size, args.img_size
    )

    model = timm.create_model(args.model, pretrained=True, num_classes=len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Parameters: {n_params:,} | img_size: {args.img_size}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    best_val_acc = 0.0
    patience_counter = 0
    ckpt_path = CKPT_DIR / f"{args.model}_best.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f} | "
            f"{time.time() - t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc}, ckpt_path)
            print(f"  -> Saved checkpoint (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nBest val_acc={best_val_acc:.4f} | {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina um modelo timm no HAM10000.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modelos disponíveis por família (ver docstring do módulo para detalhes):\n"
            "  EfficientNet : efficientnet_b0 .. efficientnet_b7\n"
            "  ResNet       : resnet18 resnet34 resnet50 resnet101 resnet152\n"
            "  MobileNet    : mobilenetv2_100 mobilenetv3_small_100\n"
            "                 mobilenetv3_large_100 mobilenetv3_large_125\n"
        ),
    )
    parser.add_argument(
        "--model", default="efficientnet_b3",
        help=(
            "Nome do modelo timm. Padrão: efficientnet_b3 (modelo principal da spec). "
            "Outros exemplos: resnet18 (baseline), mobilenetv3_large_100."
        ),
    )
    parser.add_argument(
        "--img-size", type=int, default=224,
        help=(
            "Resolução de entrada em pixels. "
            "Recomendado por família: B0=224, B1=240, B2=260, B3=300, B4=380, "
            "B5=456; ResNet/MobileNet=224."
        ),
    )
    parser.add_argument("--data-dir", default="data/organized", help="Diretório com train/val/test.")
    parser.add_argument("--epochs", type=int, default=50, help="Número máximo de épocas.")
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help=(
            "Tamanho do batch. Ajuste conforme a GPU: "
            "B0/resnet18/mobilenet→64-128; B3→32; B4→16; B5+→8 ou menos."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Taxa de aprendizado inicial (AdamW).")
    parser.add_argument("--patience", type=int, default=10, help="Épocas sem melhoria antes de parar.")
    parser.add_argument("--seed", type=int, default=SEED, help="Semente global para reprodutibilidade.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
