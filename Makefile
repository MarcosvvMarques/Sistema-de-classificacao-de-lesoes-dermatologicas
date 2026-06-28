.PHONY: setup download prepare eda \
        train-baseline train-main \
        train-efficientnet-all train-resnet-all train-mobilenet-all \
        evaluate-baseline evaluate-main compare gradcam all clean

PYTHON   := python
DATA_DIR := data/organized
EPOCHS   := 50
SEED     := 42

# ── Ambiente ──────────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ── Pipeline de dados ─────────────────────────────────────────────────────────
download:
	$(PYTHON) download_data.py

prepare:
	$(PYTHON) src/prepare_dataset.py

eda:
	$(PYTHON) src/eda.py --data-dir data --out-dir docs/figs

# ── Modelos principais (spec) ─────────────────────────────────────────────────
train-baseline:
	# ResNet18 — baseline conforme especificação do projeto
	$(PYTHON) src/train.py --model resnet18 --img-size 224 --batch-size 64 \
		--epochs $(EPOCHS) --seed $(SEED)

train-main:
	# EfficientNet-B3 — modelo principal conforme especificação do projeto
	$(PYTHON) src/train.py --model efficientnet_b3 --img-size 300 --batch-size 32 \
		--epochs $(EPOCHS) --seed $(SEED)

# ── Família EfficientNet completa (B0 → B7) ───────────────────────────────────
# Ajuste --batch-size conforme a memória da GPU disponível.
# GPU 8 GB: B0-B3 ok; B4→16; B5→8; B6/B7 exigem 16GB+.
train-efficientnet-all:
	$(PYTHON) src/train.py --model efficientnet_b0 --img-size 224 --batch-size 64 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model efficientnet_b1 --img-size 240 --batch-size 48 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model efficientnet_b2 --img-size 260 --batch-size 48 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model efficientnet_b3 --img-size 300 --batch-size 32 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model efficientnet_b4 --img-size 380 --batch-size 16 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model efficientnet_b5 --img-size 456 --batch-size  8 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model efficientnet_b6 --img-size 528 --batch-size  4 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model efficientnet_b7 --img-size 600 --batch-size  2 --epochs $(EPOCHS) --seed $(SEED)

# ── Família ResNet completa (18 → 152) ────────────────────────────────────────
train-resnet-all:
	$(PYTHON) src/train.py --model resnet18  --img-size 224 --batch-size 64 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model resnet34  --img-size 224 --batch-size 64 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model resnet50  --img-size 224 --batch-size 32 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model resnet101 --img-size 224 --batch-size 16 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model resnet152 --img-size 224 --batch-size  8 --epochs $(EPOCHS) --seed $(SEED)

# ── Família MobileNet completa ────────────────────────────────────────────────
train-mobilenet-all:
	$(PYTHON) src/train.py --model mobilenetv2_100       --img-size 224 --batch-size 128 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model mobilenetv3_small_100 --img-size 224 --batch-size 128 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model mobilenetv3_large_100 --img-size 224 --batch-size  64 --epochs $(EPOCHS) --seed $(SEED)
	$(PYTHON) src/train.py --model mobilenetv3_large_125 --img-size 224 --batch-size  64 --epochs $(EPOCHS) --seed $(SEED)

# ── Avaliação ─────────────────────────────────────────────────────────────────
evaluate-baseline:
	$(PYTHON) src/evaluate.py \
		--model-name resnet18 \
		--checkpoint checkpoints/resnet18_best.pth \
		--data-dir $(DATA_DIR)

evaluate-main:
	$(PYTHON) src/evaluate.py \
		--model-name efficientnet_b3 \
		--checkpoint checkpoints/efficientnet_b3_best.pth \
		--data-dir $(DATA_DIR)

compare:
	# Compara todos os modelos com checkpoint disponível
	$(PYTHON) src/compare_models.py

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
gradcam:
	$(PYTHON) src/gradcam.py \
		--model-name efficientnet_b3 \
		--checkpoint checkpoints/efficientnet_b3_best.pth \
		--data-dir $(DATA_DIR) \
		--method gradcam++

# ── Família YOLOv8-cls completa (nano → extra-large) ─────────────────────────
# Usa src/train_yolo.py (API Ultralytics). Ajuste --batch-size conforme GPU.
# GPU 8 GB: yolov8n→128, yolov8s→64, yolov8m→32, yolov8l→16, yolov8x→8.
train-yolo-all:
	$(PYTHON) src/train_yolo.py --model yolov8n --epochs $(EPOCHS) --batch-size 128
	$(PYTHON) src/train_yolo.py --model yolov8s --epochs $(EPOCHS) --batch-size  64
	$(PYTHON) src/train_yolo.py --model yolov8m --epochs $(EPOCHS) --batch-size  32
	$(PYTHON) src/train_yolo.py --model yolov8l --epochs $(EPOCHS) --batch-size  16
	$(PYTHON) src/train_yolo.py --model yolov8x --epochs $(EPOCHS) --batch-size   8

# ── Pipeline mínimo (spec: baseline + main) ───────────────────────────────────
pipeline-spec: download prepare eda train-baseline train-main \
               evaluate-baseline evaluate-main compare gradcam

# ── Pipeline completo (todas as famílias, incluindo YOLO) ─────────────────────
all: download prepare eda \
     train-efficientnet-all train-resnet-all train-mobilenet-all \
     train-yolo-all \
     compare gradcam

# ── Limpeza ───────────────────────────────────────────────────────────────────
clean:
	rm -rf checkpoints/ outputs/ docs/figs/
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
