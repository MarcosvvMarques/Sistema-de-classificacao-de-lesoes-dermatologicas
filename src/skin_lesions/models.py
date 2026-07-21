"""Fábrica dos modelos comparados no projeto."""

from __future__ import annotations

import torch.nn as nn
import timm

from .config import CLASS_NAMES, MODEL_INPUT_SIZES


def create_model(
    model_name: str,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> nn.Module:
    if model_name not in MODEL_INPUT_SIZES:
        supported = ", ".join(MODEL_INPUT_SIZES)
        raise ValueError(f"Modelo inválido. Use: {supported}")
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=len(CLASS_NAMES),
        drop_rate=drop_rate,
    )


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    if not trainable:
        classifier = model.get_classifier()
        for parameter in classifier.parameters():
            parameter.requires_grad = True


def set_gradual_unfreeze_stage(
    model: nn.Module,
    model_name: str,
    stage: int,
) -> None:
    """Libera classificador, bloco final e, por fim, todo o backbone."""
    set_backbone_trainable(model, trainable=False)
    if stage <= 0:
        return
    if model_name == "resnet18":
        modules = (model.layer4, model.fc)
    elif model_name == "efficientnet_b3":
        modules = (model.blocks[-2:], model.conv_head, model.bn2, model.classifier)
    else:
        raise ValueError(f"Unfreezing gradual não configurado para {model_name}.")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    if stage >= 2:
        set_backbone_trainable(model, trainable=True)


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "efficientnet_b3":
        return model.blocks[-1]
    raise ValueError(f"Grad-CAM não configurado para {model_name}.")
