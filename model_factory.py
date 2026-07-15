"""ResNet-18 factory helpers for training and inference."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from config import UNFREEZE_FROM_LAYER


def build_model(
    *,
    pretrained: bool = True,
    unfreeze_from_layer: int = UNFREEZE_FROM_LAYER,
) -> nn.Module:
    """Build ResNet-18 binary classifier with selective unfreezing."""
    weights = None
    if pretrained:
        try:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        except Exception as error:
            print(
                "Warning: could not download ImageNet weights "
                f"({error}). Training from random initialization."
            )
            model = resnet18(weights=None)
    else:
        model = resnet18(weights=None)

    for parameter in model.parameters():
        parameter.requires_grad = False

    layers_to_unfreeze: list[nn.Module] = []
    if unfreeze_from_layer <= 4:
        layers_to_unfreeze.append(model.layer4)
    if unfreeze_from_layer <= 3:
        layers_to_unfreeze.append(model.layer3)
    if unfreeze_from_layer <= 2:
        layers_to_unfreeze.append(model.layer2)
    if unfreeze_from_layer <= 1:
        layers_to_unfreeze.extend([model.layer1, model.bn1, model.conv1])

    for module in layers_to_unfreeze:
        for parameter in module.parameters():
            parameter.requires_grad = True

    feature_count = model.fc.in_features
    model.fc = nn.Linear(feature_count, 1)
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


def load_model(
    model_path: Path,
    device: torch.device,
    *,
    unfreeze_from_layer: int = UNFREEZE_FROM_LAYER,
) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find model at {model_path}. Run train.py first."
        )

    model = build_model(pretrained=False, unfreeze_from_layer=unfreeze_from_layer)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def trainable_parameter_groups(
    model: nn.Module,
    learning_rate: float,
    backbone_lr_factor: float,
) -> list[dict]:
    backbone_params = []
    head_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    groups = []
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": learning_rate * backbone_lr_factor,
            }
        )
    if head_params:
        groups.append({"params": head_params, "lr": learning_rate})
    return groups
