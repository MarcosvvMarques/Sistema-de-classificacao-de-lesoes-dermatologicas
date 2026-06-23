"""
Treinamento de YOLOv8 para classificação de lesões dermatológicas (HAM10000).

A API Ultralytics gerencia o loop de treino, augmentation e scheduler
internamente. Este script configura os hiperparâmetros e delega a execução.

Para modelos baseados em PyTorch/timm (EfficientNet, ResNet, MobileNet),
use src/train.py. Após treinar ambos, compare com src/compare_models.py.

═══════════════════════════════════════════════════════════════════════════════
 MODELOS YOLOv8-cls disponíveis
═══════════════════════════════════════════════════════════════════════════════

  Argumento --model | Arquivo base     | Params  | Batch 8GB | Observação
  ------------------|------------------|---------|-----------|------------------
  yolov8n           | yolov8n-cls.pt   |  2.7 M  |   128     | Nano — mais rápido
  yolov8s           | yolov8s-cls.pt   |  6.4 M  |    64     | Small
  yolov8m           | yolov8m-cls.pt   | 17.0 M  |    32     | ★ padrão (spec)
  yolov8l           | yolov8l-cls.pt   | 37.5 M  |    16     | Large
  yolov8x           | yolov8x-cls.pt   | 57.4 M  |     8     | Extra-large

  O peso base (yolov8*-cls.pt) é baixado automaticamente do hub Ultralytics
  na primeira execução. Requer conexão com a internet no primeiro uso.

═══════════════════════════════════════════════════════════════════════════════
 ESTRUTURA DO CHECKPOINT
═══════════════════════════════════════════════════════════════════════════════

  Após o treino, os artefatos ficam em:
      checkpoints/yolo_<model>/
          weights/
              best.pt      ← melhor modelo (por val/top1)
              last.pt      ← último epoch
          args.yaml        ← hiperparâmetros usados
          results.csv      ← histórico de métricas por epoch
          confusion_matrix.png
          results.png      ← curvas de treino/val

  O compare_models.py sabe onde procurar os pesos:
      checkpoints/yolo_<model>/weights/best.pt

═══════════════════════════════════════════════════════════════════════════════
 DIFERENÇAS EM RELAÇÃO AO train.py (timm)
═══════════════════════════════════════════════════════════════════════════════

  | Aspecto            | train.py (timm)          | train_yolo.py (Ultralytics)|
  |--------------------|--------------------------|----------------------------|
  | Loop de treino     | Manual (PyTorch)         | Gerenciado pela API YOLO   |
  | Augmentation       | Albumentations custom    | Augmentation interno YOLO  |
  | Mixed precision    | torch.amp explícito      | Automático                 |
  | Checkpoint         | .pth (state_dict)        | .pt (modelo completo)      |
  | Grad-CAM           | Suportado (gradcam.py)   | Não suportado diretamente  |
  | Classe ordering    | mel/nv/bcc/akiec/bkl/df/vasc | Alfabética (akiec/bcc/...)  |

  IMPORTANTE: O YOLO ordena as classes em ordem alfabética pelo nome da pasta.
  O compare_models.py remapeia automaticamente para a ordem padrão do projeto.

═══════════════════════════════════════════════════════════════════════════════
 EXEMPLOS DE USO
═══════════════════════════════════════════════════════════════════════════════

  # Nano — experimento rápido (< 5 min em GPU)
  python src/train_yolo.py --model yolov8n --epochs 30 --batch-size 128

  # Small — equilíbrio velocidade/acurácia
  python src/train_yolo.py --model yolov8s --epochs 50 --batch-size 64

  # Medium — padrão recomendado
  python src/train_yolo.py --model yolov8m --epochs 50 --batch-size 32

  # Large — melhor acurácia; GPU com >= 8 GB
  python src/train_yolo.py --model yolov8l --epochs 50 --batch-size 16

  # Extra-large — GPU com >= 8 GB; batch menor
  python src/train_yolo.py --model yolov8x --epochs 50 --batch-size 8

  # CPU (lento, apenas para teste)
  python src/train_yolo.py --model yolov8n --device cpu --epochs 5 --batch-size 16

  # Comparar com modelos timm após o treino
  python src/compare_models.py
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    """Configura e executa o treino de um modelo YOLOv8-cls no HAM10000.

    O Ultralytics gerencia augmentation, scheduler cosine, early stopping
    e salvamento do melhor checkpoint automaticamente.

    Saída:
        checkpoints/yolo_<model>/weights/best.pt  ← melhor peso (por val/top1)
        checkpoints/yolo_<model>/weights/last.pt  ← último epoch
        checkpoints/yolo_<model>/results.csv      ← métricas por epoch
    """
    parser = argparse.ArgumentParser(
        description="Treina um modelo YOLOv8-cls no HAM10000.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modelos disponíveis (do mais leve ao mais pesado):\n"
            "  yolov8n  ~2.7M params  — nano, mais rápido\n"
            "  yolov8s  ~6.4M params  — small\n"
            "  yolov8m  ~17M  params  — medium (padrão)\n"
            "  yolov8l  ~37.5M params — large\n"
            "  yolov8x  ~57.4M params — extra-large\n\n"
            "Para modelos timm (EfficientNet/ResNet/MobileNet), use src/train.py.\n"
            "Para comparar todos os modelos treinados, use src/compare_models.py."
        ),
    )
    parser.add_argument(
        "--model", default="yolov8m",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help=(
            "Variante YOLOv8-cls. Dica de batch por GPU 8GB: "
            "yolov8n→128, yolov8s→64, yolov8m→32, yolov8l→16, yolov8x→8."
        ),
    )
    parser.add_argument(
        "--data-dir", default="data/organized",
        help="Diretório raiz com subpastas train/val/test/<classe>/. "
             "Gerado por src/prepare_dataset.py.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Número máximo de épocas (early stopping com patience=10).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Tamanho do batch. Ajuste conforme a memória da GPU.",
    )
    parser.add_argument(
        "--img-size", type=int, default=224,
        help="Resolução de entrada em pixels (padrão 224).",
    )
    parser.add_argument(
        "--device", default="0",
        help="ID da GPU (ex.: '0', '0,1') ou 'cpu'.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Diretório de dados não encontrado: {data_dir}\n"
            "Execute primeiro: python src/prepare_dataset.py"
        )

    print(
        f"\nIniciando treino YOLOv8-cls\n"
        f"  Modelo   : {args.model} ({args.model}-cls.pt)\n"
        f"  Dados    : {data_dir.resolve()}\n"
        f"  Epochs   : {args.epochs}\n"
        f"  Batch    : {args.batch_size}\n"
        f"  Img size : {args.img_size} px\n"
        f"  Device   : {args.device}\n"
        f"  Saída    : checkpoints/yolo_{args.model}/\n"
    )

    model = YOLO(f"{args.model}-cls.pt")

    model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        project="checkpoints",
        name=f"yolo_{args.model}",
        cos_lr=True,
        augment=True,
        patience=10,
        save=True,
        exist_ok=True,
    )

    print("\nAvaliando no conjunto de teste...")
    metrics = model.val(data=str(data_dir), split="test")
    print(f"\nResultados no teste:\n{metrics}")

    best_path = Path("checkpoints") / f"yolo_{args.model}" / "weights" / "best.pt"
    print(f"\nMelhor checkpoint: {best_path.resolve()}")
    print("Para comparar com outros modelos: python src/compare_models.py")


if __name__ == "__main__":
    main()
