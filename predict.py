"""Predict pneumonia class for a single chest X-ray (CLI)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from config import METRICS_PATH, MODELS_DIR, MODEL_PATH
from inference import (
    create_gradcam,
    create_overlay,
    prepare_image,
    predict,
    score_band,
)
from metrics_io import load_metrics
from model_factory import load_model


def load_image(path: Path) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        try:
            import numpy as np
            import pydicom
        except ImportError as error:
            raise SystemExit(
                "DICOM support requires pydicom. pip install pydicom"
            ) from error

        dataset = pydicom.dcmread(str(path))
        pixels = dataset.pixel_array.astype("float32")
        pixels -= pixels.min()
        if pixels.max() > 0:
            pixels /= pixels.max()
        pixels = (pixels * 255).astype("uint8")
        return Image.fromarray(pixels).convert("RGB")

    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError as error:
        raise SystemExit(f"Could not open image: {path}") from error


def main() -> None:
    default_image = Path(__file__).resolve().parent / "data" / "images_demo" / "00000001_000.png"

    parser = argparse.ArgumentParser(
        description="Predict Pneumonia vs No Finding from a chest X-ray"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(default_image),
        help=f"Path to PNG/JPG/DICOM image (default: {default_image.name})",
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to trained .pt weights",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (default: optimal from metrics.json or 0.5)",
    )
    parser.add_argument(
        "--save-gradcam",
        action="store_true",
        default=True,
        help="Save Grad-CAM overlay under models/ (default: on)",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_false",
        dest="save_gradcam",
        help="Skip Grad-CAM overlay",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.6,
        help="Grad-CAM overlay intensity",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(
            f"Image not found: {image_path}\n"
            "In PyCharm: Run → Edit Configurations → Parameters, e.g.\n"
            "  data/images_demo/00000001_000.png"
        )

    metrics = load_metrics(METRICS_PATH)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(metrics.get("optimal_threshold") or 0.5)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model), device)

    image = load_image(image_path)
    _, image_array, input_tensor = prepare_image(image, device)
    probability = predict(model, input_tensor)
    classification = "Pneumonia" if probability >= threshold else "No Finding"

    print(f"Image: {image_path}")
    print(f"Prediction: {classification}")
    print(f"Pneumonia score: {probability:.2%}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Score band: {score_band(probability)}")
    print(f"Device: {device}")

    if args.save_gradcam:
        heatmap = create_gradcam(model, input_tensor)
        overlay = create_overlay(image_array, heatmap, args.intensity)
        out_path = MODELS_DIR / f"gradcam_{image_path.stem}.png"
        Image.fromarray(overlay).save(out_path)
        print(f"Saved Grad-CAM: {out_path}")


if __name__ == "__main__":
    main()
