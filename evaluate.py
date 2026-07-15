"""Evaluate trained model and save graphs under models/."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CALIBRATION_PATH,
    CLASS_METRICS_PATH,
    CM_PATH,
    DATA_DIR,
    DEFAULT_THRESHOLD,
    METRICS_PATH,
    MODEL_PATH,
    PR_CURVE_PATH,
    RESULTS_DASHBOARD_PATH,
    ROC_CURVE_PATH,
    TEST_CSV,
    TRAINING_LOG_PATH,
)
from dataset import ChestXrayDataset
from metrics_io import save_metrics
from model_factory import load_model
from plots import (
    plot_calibration,
    plot_class_metrics,
    plot_confusion,
    plot_pr,
    plot_results_dashboard,
    plot_roc,
)
from transforms_util import build_eval_transform


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probabilities > start) & (probabilities <= end)
        if not np.any(mask):
            continue
        bin_confidence = float(np.mean(probabilities[mask]))
        bin_accuracy = float(np.mean(labels[mask]))
        ece += (np.sum(mask) / len(probabilities)) * abs(
            bin_accuracy - bin_confidence
        )
    return float(ece)


def youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    j_scores = tpr - fpr
    best_index = int(np.argmax(j_scores))
    if len(thresholds) == 0:
        return float(DEFAULT_THRESHOLD)
    best_index = min(best_index, len(thresholds) - 1)
    return float(thresholds[best_index])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = ChestXrayDataset(
        csv_file=TEST_CSV,
        data_folder=DATA_DIR,
        transform=build_eval_transform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = load_model(MODEL_PATH, device)

    all_labels: list[float] = []
    all_probabilities: list[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.sigmoid(logits).squeeze(1)
            all_labels.extend(labels.numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    labels_np = np.asarray(all_labels, dtype=np.float64)
    probs_np = np.asarray(all_probabilities, dtype=np.float64)

    auc = float(roc_auc_score(labels_np, probs_np))
    ap = float(average_precision_score(labels_np, probs_np))
    optimal_threshold = youden_threshold(labels_np, probs_np)

    predictions = (probs_np >= optimal_threshold).astype(int)
    matrix = confusion_matrix(labels_np, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    sensitivity = float(recall_score(labels_np, predictions, zero_division=0))
    specificity = float(tn / max(tn + fp, 1))
    precision = float(precision_score(labels_np, predictions, zero_division=0))
    f1 = float(f1_score(labels_np, predictions, zero_division=0))
    brier = float(brier_score_loss(labels_np, probs_np))
    ece = expected_calibration_error(labels_np, probs_np)

    print("\nTest results (optimal Youden threshold)")
    print(f"ROC-AUC:            {auc:.4f}")
    print(f"Average precision:  {ap:.4f}")
    print(f"Optimal threshold:  {optimal_threshold:.4f}")
    print(f"Sensitivity:        {sensitivity:.4f}")
    print(f"Specificity:        {specificity:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"F1:                 {f1:.4f}")
    print(f"Brier score:        {brier:.4f}")
    print(f"ECE:                {ece:.4f}")
    print("\nConfusion matrix")
    print(matrix)
    print("\nClassification report")
    print(
        classification_report(
            labels_np,
            predictions,
            target_names=["No Finding", "Pneumonia"],
            digits=4,
            zero_division=0,
        )
    )

    history = None
    if TRAINING_LOG_PATH.exists():
        history = pd.read_csv(TRAINING_LOG_PATH)

    plot_roc(labels_np, probs_np, auc, ROC_CURVE_PATH)
    plot_pr(labels_np, probs_np, ap, PR_CURVE_PATH)
    plot_confusion(matrix, CM_PATH, optimal_threshold)
    plot_calibration(labels_np, probs_np, CALIBRATION_PATH)
    plot_class_metrics(
        sensitivity,
        specificity,
        precision,
        f1,
        CLASS_METRICS_PATH,
    )
    plot_results_dashboard(
        history=history,
        labels=labels_np,
        probabilities=probs_np,
        matrix=matrix,
        auc=auc,
        ap=ap,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=f1,
        threshold=optimal_threshold,
        path=RESULTS_DASHBOARD_PATH,
    )

    metrics = {
        "auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "threshold": DEFAULT_THRESHOLD,
        "optimal_threshold": round(optimal_threshold, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "n_test": int(len(labels_np)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "note": "Generated by evaluate.py after training.",
    }
    save_metrics(METRICS_PATH, metrics)

    print(f"\nSaved graphs:")
    print(f"  {RESULTS_DASHBOARD_PATH}")
    print(f"  {ROC_CURVE_PATH}")
    print(f"  {PR_CURVE_PATH}")
    print(f"  {CM_PATH}")
    print(f"  {CALIBRATION_PATH}")
    print(f"  {CLASS_METRICS_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
