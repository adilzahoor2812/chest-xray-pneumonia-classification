"""Matplotlib / seaborn plots saved under models/ (brain-tumor style)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_training_curves(history: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Training loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["val_auc"], label="Validation AUC", color="#2bb7a8")
    axes[1].set_title("Validation ROC-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc(labels, probabilities, auc: float, path: Path) -> None:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, probabilities)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Chest X-Ray Model ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pr(labels, probabilities, ap: float, path: Path) -> None:
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion(matrix: np.ndarray, path: Path, threshold: float) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Finding", "Pneumonia"],
        yticklabels=["No Finding", "Pneumonia"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_calibration(labels, probabilities, path: Path) -> None:
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(
        labels,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )
    plt.figure(figsize=(7, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_class_metrics(
    sensitivity: float,
    specificity: float,
    precision: float,
    f1: float,
    path: Path,
) -> None:
    names = ["Sensitivity", "Specificity", "Precision", "F1"]
    values = [sensitivity, specificity, precision, f1]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=["#2bb7a8", "#3a7ca5", "#d6a15c", "#6c8eae"])
    plt.ylim(0, 1.05)
    plt.title("Test-Set Class Metrics (Pneumonia-oriented)")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_results_dashboard(
    *,
    history: pd.DataFrame | None,
    labels: np.ndarray,
    probabilities: np.ndarray,
    matrix: np.ndarray,
    auc: float,
    ap: float,
    sensitivity: float,
    specificity: float,
    precision: float,
    f1: float,
    threshold: float,
    path: Path,
) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if history is not None and len(history) > 0:
        ax.plot(history["epoch"], history["train_loss"], label="Train loss")
        ax.plot(history["epoch"], history["val_loss"], label="Val loss")
        ax.set_title("Training Curves")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No training history yet", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(labels, probabilities)
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["No Finding", "Pneumonia"],
        yticklabels=["No Finding", "Pneumonia"],
    )
    ax.set_title(f"Confusion Matrix @ {threshold:.2f}")

    ax = axes[1, 1]
    names = ["Sens", "Spec", "Prec", "F1", "AP"]
    values = [sensitivity, specificity, precision, f1, ap]
    ax.bar(names, values, color="#2bb7a8")
    ax.set_ylim(0, 1.05)
    ax.set_title("Summary Metrics")
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center")

    fig.suptitle("Chest X-ray Pneumonia — Results Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
