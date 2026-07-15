"""Chest X-ray dataset with PNG/JPG support."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        data_folder: str | Path,
        transform=None,
        *,
        quiet: bool = False,
    ):
        self.data_folder = Path(data_folder)
        self.transform = transform

        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find CSV file: {csv_path}")
        if not self.data_folder.exists():
            raise FileNotFoundError(
                f"Could not find data folder: {self.data_folder}"
            )

        self.labels = pd.read_csv(csv_path)

        required_columns = {"Image Index", "Finding Labels", "label"}
        missing_columns = required_columns - set(self.labels.columns)
        if missing_columns:
            raise KeyError(
                f"CSV is missing required columns: {missing_columns}. "
                "Run prepare_data.py again."
            )

        image_files = [
            *self.data_folder.rglob("*.png"),
            *self.data_folder.rglob("*.jpg"),
            *self.data_folder.rglob("*.jpeg"),
            *self.data_folder.rglob("*.PNG"),
            *self.data_folder.rglob("*.JPG"),
            *self.data_folder.rglob("*.JPEG"),
        ]

        self.image_paths: dict[str, Path] = {}
        for image_path in image_files:
            # Prefer first occurrence; warn via overwrite only if identical name
            # from different folders (rare on NIH).
            self.image_paths.setdefault(image_path.name, image_path)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No PNG/JPG images were found inside {self.data_folder}."
            )

        self.labels = self.labels[
            self.labels["Image Index"].isin(self.image_paths)
        ].copy()
        self.labels = self.labels.reset_index(drop=True)

        if len(self.labels) == 0:
            raise RuntimeError(
                "No CSV rows matched the image files. "
                "Check that the X-ray folders are inside the data folder."
            )

        self.labels["label"] = self.labels["label"].astype(int)

        if not quiet:
            print(f"Loaded {len(self.labels)} labeled X-rays.")
            label_counts = self.labels["label"].value_counts().sort_index()
            print("Class counts:")
            print(f"No Finding: {label_counts.get(0, 0)}")
            print(f"Pneumonia:  {label_counts.get(1, 0)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.labels):
            raise IndexError(f"Index {index} is outside dataset range.")

        row = self.labels.iloc[index]
        image_name = row["Image Index"]
        image_path = self.image_paths[image_name]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Could not open image: {image_path}") from error

        label = int(row["label"])
        if self.transform is not None:
            image = self.transform(image)

        return image, label
