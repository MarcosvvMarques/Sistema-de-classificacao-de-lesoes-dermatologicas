"""
GradCAM and GradCAM++ visualization for any timm CNN model.

Usage:
    # Single image
    python src/gradcam.py --model-name efficientnet_b4 \
        --checkpoint checkpoints/efficientnet_b4_best.pth \
        --image data/organized/test/mel/ISIC_0024306.jpg

    # All test classes (2 images per class)
    python src/gradcam.py --model-name efficientnet_b4 \
        --checkpoint checkpoints/efficientnet_b4_best.pth \
        --data-dir data/organized

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
import torch.nn.functional as F

from dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD

OUTPUT_DIR = Path("outputs/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        target_layer.register_forward_hook(lambda _, __, out: setattr(self, "_acts", out.detach()))
        target_layer.register_full_backward_hook(
            lambda _, __, grad_out: setattr(self, "_grads", grad_out[0].detach())
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
    if "efficientnet" in model_name:
        return model.conv_head
    if "resnet" in model_name:
        last = model.layer4[-1]
        return last.conv3 if hasattr(last, "conv3") else last.conv2
    if "mobilenet" in model_name:
        return model.conv_head
    # Generic fallback: last Conv2d
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise ValueError(f"Cannot determine target layer for {model_name}")


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
    method: str = "gradcam",
    img_size: int = 224,
    save: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    target_layer = _get_target_layer(model, model_name)
    cam_fn = GradCAMPlusPlus(model, target_layer) if method == "gradcam++" else GradCAM(model, target_layer)

    img_tensor, img_np = _load_image(image_path, img_size)
    cam, pred_idx, probs = cam_fn(img_tensor.to(device))

    pred_class = CLASSES[pred_idx]
    overlay = _overlay(img_np, cam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np);          axes[0].set_title("Original");           axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title(f"{method.upper()}");  axes[1].axis("off")

    title = f"Pred: {pred_class} ({probs[pred_idx]:.1%})"
    if true_label:
        title += f"\nTrue: {true_label}"
    axes[2].imshow(overlay); axes[2].set_title(title); axes[2].axis("off")

    plt.tight_layout()
    if save:
        out = OUTPUT_DIR / f"{Path(image_path).stem}_{method}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()


def visualize_batch(data_dir: str, model_name: str, checkpoint: str, n_per_class: int = 2, method: str = "gradcam"):
    test_dir = Path(data_dir) / "test"
    for cls in CLASSES:
        for img_path in list((test_dir / cls).glob("*.jpg"))[:n_per_class]:
            visualize(str(img_path), model_name, checkpoint, true_label=cls, method=method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="efficientnet_b4")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--data-dir", default="data/organized")
    parser.add_argument("--method", default="gradcam", choices=["gradcam", "gradcam++"])
    parser.add_argument("--n-per-class", type=int, default=2)
    args = parser.parse_args()

    if args.image:
        visualize(args.image, args.model_name, args.checkpoint, method=args.method)
    else:
        visualize_batch(args.data_dir, args.model_name, args.checkpoint, args.n_per_class, args.method)


if __name__ == "__main__":
    main()
