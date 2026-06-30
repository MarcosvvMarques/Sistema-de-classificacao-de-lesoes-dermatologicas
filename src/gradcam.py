"""
GradCAM and GradCAM++ visualization for timm CNN models and YOLOv8n-cls.

Usage:
    # Single image — um modelo
    python src/gradcam.py --model-name efficientnet_b0 \
        --checkpoint checkpoints/efficientnet_b0_best.pth \
        --image data/organized/test/mel/ISIC_0024306.jpg

    # Batch por classe — um modelo
    python src/gradcam.py --model-name efficientnet_b0 \
        --checkpoint checkpoints/efficientnet_b0_best.pth \
        --data-dir data/organized

    # Painel comparativo (original + 3 modelos numa imagem só)
    python src/gradcam.py --compare --data-dir data/organized

    # Painel comparativo com todos os 4 modelos (inclui YOLOv8n)
    python src/gradcam.py --compare --include-yolo --data-dir data/organized

Outputs saved to outputs/gradcam/
"""
import argparse
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD

OUTPUT_DIR = Path("outputs/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_MODELS = [
    ("efficientnet_b0",       "checkpoints/efficientnet_b0_best.pth"),
    ("mobilenetv3_small_100", "checkpoints/mobilenetv3_small_100_best.pth"),
    ("resnet18",              "checkpoints/resnet18_best.pth"),
]

YOLO_CHECKPOINT = "checkpoints/yolo_yolov8n/weights/best.pt"


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        target_layer.register_forward_hook(
            lambda _, __, out: setattr(self, "_acts", out.detach().clone())
        )
        target_layer.register_full_backward_hook(
            lambda _, __, grad_out: setattr(self, "_grads", grad_out[0].detach().clone())
        )

    def __call__(self, img_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        x = img_tensor.unsqueeze(0)
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        probs = logits.softmax(1).squeeze().detach().cpu().numpy()
        return cam, class_idx, probs


class GradCAMPlusPlus(GradCAM):
    def __call__(self, img_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        x = img_tensor.unsqueeze(0)
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads, acts = self._grads, self._acts
        g2, g3 = grads ** 2, grads ** 3
        denom = 2 * g2 + acts * g3.sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        probs = logits.softmax(1).squeeze().detach().cpu().numpy()
        return cam, class_idx, probs


def _get_target_layer(model, model_name: str) -> torch.nn.Module:
    if "yolo" in model_name:
        return _get_yolo_target_layer(model)
    if "efficientnet" in model_name:
        return model.conv_head
    if "resnet" in model_name:
        last = model.layer4[-1]
        return last.conv3 if hasattr(last, "conv3") else last.conv2
    if "mobilenet" in model_name:
        # conv_head é seguido por hardswish inplace e produz mapa plano;
        # o último bloco de inverted residuals tem gradientes espaciais mais ricos
        return model.blocks[-1][-1]
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise ValueError(f"Cannot determine target layer for {model_name}")


def _load_model(model_name: str, checkpoint: str, device: torch.device):
    model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    return model


class _YOLOWrapper(nn.Module):
    """Wraps the inner YOLO nn.Module to normalize output to a plain logit tensor."""
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        return out[0] if isinstance(out, tuple) else out

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)


def _load_yolo_model(checkpoint: str, device: torch.device) -> nn.Module:
    from ultralytics import YOLO
    yolo = YOLO(checkpoint)
    inner = yolo.model.to(device)
    for m in inner.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    inner.eval()
    for p in inner.parameters():
        p.requires_grad_(True)
    return _YOLOWrapper(inner)


def _get_yolo_target_layer(model: _YOLOWrapper) -> nn.Module:
    # model.inner.model[8] é o último bloco C2f antes do head Classify
    return model.inner.model[8]


def _load_image(path: str, img_size: int = 224):
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor = transform(image=img_rgb)["image"]
    return tensor, img_resized


def _overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * heatmap + (1 - alpha) * img)


def visualize(
    image_path: str,
    model_name: str,
    checkpoint: str,
    true_label: str | None = None,
    method: str = "gradcam++",
    img_size: int = 224,
    save: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_name, checkpoint, device)
    target_layer = _get_target_layer(model, model_name)
    cam_fn = GradCAMPlusPlus(model, target_layer) if method == "gradcam++" else GradCAM(model, target_layer)

    img_tensor, img_np = _load_image(image_path, img_size)
    cam, pred_idx, probs = cam_fn(img_tensor.to(device))

    pred_class = CLASSES[pred_idx]
    overlay = _overlay(img_np, cam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np);          axes[0].set_title("Original");          axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title(f"{method.upper()}"); axes[1].axis("off")

    title = f"Pred: {pred_class} ({probs[pred_idx]:.1%})"
    if true_label:
        title += f"\nTrue: {true_label}"
    axes[2].imshow(overlay); axes[2].set_title(title); axes[2].axis("off")

    plt.tight_layout()
    if save:
        out = OUTPUT_DIR / f"{Path(image_path).stem}_{model_name}_{method}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


def visualize_comparison(
    image_path: str,
    models_config: list[tuple[str, str]],
    true_label: str | None = None,
    method: str = "gradcam++",
    img_size: int = 224,
):
    """Painel único: imagem original + overlay GradCAM++ de cada modelo lado a lado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor, img_np = _load_image(image_path, img_size)

    n = len(models_config)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

    gt_label = true_label.upper() if true_label else "?"
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original\nGT: {gt_label}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for i, (model_name, checkpoint) in enumerate(models_config):
        if "yolo" in model_name:
            model = _load_yolo_model(checkpoint, device)
        else:
            model = _load_model(model_name, checkpoint, device)
        target_layer = _get_target_layer(model, model_name)
        cam_fn = GradCAMPlusPlus(model, target_layer) if method == "gradcam++" else GradCAM(model, target_layer)

        cam, pred_idx, probs = cam_fn(img_tensor.to(device))
        pred_class = CLASSES[pred_idx]
        overlay = _overlay(img_np, cam)

        correct = "✓" if true_label and pred_class == true_label else "✗"
        color = "green" if true_label and pred_class == true_label else "red"
        title = f"{model_name}\nPred: {pred_class.upper()} ({probs[pred_idx]:.1%}) {correct}"
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(title, fontsize=9, color=color)
        axes[i + 1].axis("off")

    stem = Path(image_path).stem
    gt_str = f"GT: {gt_label}"
    plt.suptitle(f"GradCAM++ — {stem} | {gt_str}", fontsize=11, y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / f"{stem}_comparison_{method}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def visualize_batch(data_dir: str, model_name: str, checkpoint: str, n_per_class: int = 2, method: str = "gradcam++"):
    test_dir = Path(data_dir) / "test"
    for cls in CLASSES:
        for img_path in list((test_dir / cls).glob("*.jpg"))[:n_per_class]:
            visualize(str(img_path), model_name, checkpoint, true_label=cls, method=method)


def visualize_comparison_batch(
    data_dir: str,
    models_config: list[tuple[str, str]],
    n_per_class: int = 2,
    method: str = "gradcam++",
):
    test_dir = Path(data_dir) / "test"
    for cls in CLASSES:
        for img_path in list((test_dir / cls).glob("*.jpg"))[:n_per_class]:
            visualize_comparison(str(img_path), models_config, true_label=cls, method=method)


def main():
    parser = argparse.ArgumentParser(
        description="GradCAM / GradCAM++ para modelos timm no HAM10000.",
    )
    parser.add_argument("--model-name", default="efficientnet_b0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--data-dir", default="data/organized")
    parser.add_argument("--method", default="gradcam++", choices=["gradcam", "gradcam++"])
    parser.add_argument("--n-per-class", type=int, default=2)
    parser.add_argument(
        "--compare", action="store_true",
        help="Gera painel comparativo com todos os modelos em COMPARISON_MODELS.",
    )
    parser.add_argument(
        "--include-yolo", action="store_true",
        help="Adiciona YOLOv8n ao painel comparativo (requer --compare).",
    )
    args = parser.parse_args()

    if args.compare:
        available = [(n, c) for n, c in COMPARISON_MODELS if Path(c).exists()]
        if args.include_yolo and Path(YOLO_CHECKPOINT).exists():
            available.append(("yolov8n", YOLO_CHECKPOINT))
        if not available:
            print("Nenhum checkpoint encontrado para comparação.")
            return
        if args.image:
            # Infere true_label a partir do diretório pai se a imagem estiver em data/organized/*/<cls>/
            img_path = Path(args.image)
            true_label = img_path.parent.name if img_path.parent.name in CLASSES else None
            visualize_comparison(args.image, available, true_label=true_label, method=args.method)
        else:
            visualize_comparison_batch(args.data_dir, available, args.n_per_class, args.method)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint é obrigatório no modo single-model.")
        if args.image:
            img_path = Path(args.image)
            true_label = img_path.parent.name if img_path.parent.name in CLASSES else None
            visualize(args.image, args.model_name, args.checkpoint, true_label=true_label, method=args.method)
        else:
            visualize_batch(args.data_dir, args.model_name, args.checkpoint, args.n_per_class, args.method)


if __name__ == "__main__":
    main()
