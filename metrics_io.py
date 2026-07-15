"""Metrics persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS: dict[str, Any] = {
    "auc": 0.7744,
    "sensitivity": 0.7633,
    "specificity": 0.6508,
    "precision": 0.0481,
    "f1": None,
    "threshold": 0.5,
    "optimal_threshold": 0.5,
    "brier_score": None,
    "ece": None,
    "n_test": None,
    "note": (
        "Placeholder metrics from the original shipped model. "
        "Re-run evaluate.py after training to refresh."
    ),
}


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_METRICS)

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    merged = dict(DEFAULT_METRICS)
    merged.update(data)
    return merged


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
