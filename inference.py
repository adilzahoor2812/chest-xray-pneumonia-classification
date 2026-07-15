"""Inference helpers: predict, Grad-CAM, report generation."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import IMAGE_SIZE
from transforms_util import build_eval_transform


def score_band(probability: float) -> str:
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    if probability < 0.75:
        return "Elevated"
    return "High"


def prepare_image(
    image: Image.Image,
    device: torch.device,
) -> tuple[Image.Image, np.ndarray, torch.Tensor]:
    rgb_image = image.convert("RGB")
    resized = rgb_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(resized).astype(np.float32) / 255.0
    transform = build_eval_transform()
    input_tensor = transform(rgb_image).unsqueeze(0).to(device)
    return rgb_image, image_array, input_tensor


def predict(model: nn.Module, input_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        logit = model(input_tensor)
        return float(torch.sigmoid(logit).item())


def create_gradcam(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        return cam(input_tensor=input_tensor, targets=None)[0]


def create_overlay(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    intensity: float,
) -> np.ndarray:
    overlay = show_cam_on_image(image_array, heatmap, use_rgb=True).astype(
        np.float32
    ) / 255.0
    blended = (1.0 - intensity) * image_array + intensity * overlay
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def image_to_base64(image: Image.Image | np.ndarray, fmt: str = "PNG") -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def build_report(
    *,
    filename: str,
    dimensions: tuple[int, int],
    probability: float,
    threshold: float,
    classification: str,
    metrics: dict[str, Any],
    device: torch.device,
) -> str:
    return f"""
CHEST X-RAY PNEUMONIA CLASSIFICATION — RESEARCH REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMAGE
Filename: {filename}
Original dimensions: {dimensions[0]} x {dimensions[1]} pixels
Model input dimensions: {IMAGE_SIZE} x {IMAGE_SIZE} pixels

MODEL OUTPUT
Classification: {classification}
Pneumonia-class score: {probability:.6f}
Displayed score: {probability:.1%}
Decision threshold: {threshold:.2f}
Score band: {score_band(probability)}

MODEL
Architecture: ResNet-18
Framework: PyTorch
Runtime device: {device}

HELD-OUT TEST PERFORMANCE
ROC-AUC: {metrics.get("auc")}
Sensitivity: {metrics.get("sensitivity")}
Specificity: {metrics.get("specificity")}
Pneumonia precision: {metrics.get("precision")}
Optimal threshold: {metrics.get("optimal_threshold")}
Brier score: {metrics.get("brier_score")}
ECE: {metrics.get("ece")}

LIMITATIONS
This is an educational research prototype, not a medical device.
It must not be used for diagnosis, screening, or treatment decisions.
The model score is not a calibrated clinical probability.
Grad-CAM does not identify lesions or establish valid medical reasoning.
""".strip()
