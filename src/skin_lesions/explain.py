"""Grad-CAM e Grad-CAM++ para inspeção das predições."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CLASS_NAMES
from .models import get_gradcam_target_layer


class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        method: Literal["gradcam", "gradcam++"] = "gradcam",
    ):
        self.model = model
        self.method = method
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handle = target_layer.register_forward_hook(self._capture_activations)

    def _capture_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output
        output.register_hook(self._capture_gradients)

    def _capture_gradients(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def __call__(
        self,
        image: torch.Tensor,
        class_index: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        probabilities = torch.softmax(logits, dim=1)
        if class_index is None:
            class_index = int(probabilities.argmax(dim=1).item())
        logits[0, class_index].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Não foi possível capturar ativações e gradientes.")

        activations = self.activations
        gradients = self.gradients
        if self.method == "gradcam":
            weights = gradients.mean(dim=(2, 3), keepdim=True)
        elif self.method == "gradcam++":
            gradient_sq = gradients.pow(2)
            gradient_cube = gradient_sq * gradients
            denominator = (
                2.0 * gradient_sq
                + (activations * gradient_cube).sum(dim=(2, 3), keepdim=True)
            )
            alphas = gradient_sq / (denominator + 1e-8)
            weights = (alphas * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        else:
            raise ValueError("Método deve ser 'gradcam' ou 'gradcam++'.")

        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        confidence = float(probabilities[0, class_index].detach().cpu())
        return cam.detach().cpu().numpy(), class_index, confidence

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor((0.485, 0.456, 0.406), device=tensor.device)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=tensor.device)[:, None, None]
    image = (tensor * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)


def _overlay(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)


def save_explanations(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    method: Literal["gradcam", "gradcam++"] = "gradcam",
    max_images: int = 12,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = get_gradcam_target_layer(model, model_name)
    saved: list[Path] = []
    with GradCAM(model, target_layer, method=method) as explainer:
        for batch in loader:
            for index in range(batch["image"].shape[0]):
                image = batch["image"][index : index + 1].to(device)
                cam, predicted, confidence = explainer(image)
                real = int(batch["label"][index])
                visualization = _overlay(_denormalize(image[0]), cam)
                image_id = batch["image_id"][index]
                filename = (
                    f"{image_id}_real-{CLASS_NAMES[real]}_"
                    f"pred-{CLASS_NAMES[predicted]}_{confidence:.3f}_{method}.png"
                )
                output_path = output_dir / filename
                bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                success, encoded = cv2.imencode(".png", bgr)
                if not success:
                    raise RuntimeError(f"Falha ao codificar {output_path}")
                encoded.tofile(output_path)
                saved.append(output_path)
                if len(saved) >= max_images:
                    return saved
    return saved
