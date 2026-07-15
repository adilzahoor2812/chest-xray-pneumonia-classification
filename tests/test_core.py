"""Unit tests that do not require the NIH dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from dataset import ChestXrayDataset
from metrics_io import DEFAULT_METRICS, load_metrics, save_metrics
from model_factory import build_model
from transforms_util import build_eval_transform, build_train_transform


@pytest.fixture()
def tiny_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rows = []
    for index, label in enumerate([0, 1, 0, 1]):
        name = f"img_{index:04d}.png"
        image = Image.fromarray(
            np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        ).convert("RGB")
        image.save(data_dir / name)
        rows.append(
            {
                "Image Index": name,
                "Finding Labels": "Pneumonia" if label else "No Finding",
                "label": label,
                "Patient ID": index,
            }
        )

    csv_path = data_dir / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return data_dir


def test_transforms_output_shape():
    image = Image.fromarray(np.zeros((300, 280, 3), dtype=np.uint8))
    train_tensor = build_train_transform()(image)
    eval_tensor = build_eval_transform()(image)
    assert train_tensor.shape == (3, 224, 224)
    assert eval_tensor.shape == (3, 224, 224)


def test_build_model_trainable_layers():
    model = build_model(pretrained=False, unfreeze_from_layer=3)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any(name.startswith("fc.") for name in trainable)
    assert any(name.startswith("layer4.") for name in trainable)
    assert any(name.startswith("layer3.") for name in trainable)
    assert not any(name.startswith("layer1.") for name in trainable)


def test_dataset_loads_images(tiny_data_dir: Path):
    dataset = ChestXrayDataset(
        csv_file=tiny_data_dir / "train.csv",
        data_folder=tiny_data_dir,
        transform=build_eval_transform(),
        quiet=True,
    )
    assert len(dataset) == 4
    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert label in (0, 1)


def test_metrics_roundtrip(tmp_path: Path):
    path = tmp_path / "metrics.json"
    payload = {"auc": 0.91, "precision": 0.4, "note": "test"}
    save_metrics(path, payload)
    loaded = load_metrics(path)
    assert loaded["auc"] == 0.91
    assert loaded["precision"] == 0.4
    assert "sensitivity" in loaded  # merged defaults


def test_model_forward_pass():
    model = build_model(pretrained=False, unfreeze_from_layer=5)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 1)


def test_default_metrics_keys():
    assert "auc" in DEFAULT_METRICS
    assert "optimal_threshold" in DEFAULT_METRICS
