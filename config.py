"""Shared configuration for chest-xray-pneumonia-classification."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

SOURCE_CSV = DATA_DIR / "Data_Entry_2017.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "best_model.pt"
METRICS_PATH = MODELS_DIR / "metrics.json"
TRAINING_LOG_PATH = MODELS_DIR / "training_history.csv"

TRAINING_CURVES_PATH = MODELS_DIR / "training_curves.png"
ROC_CURVE_PATH = MODELS_DIR / "roc_curve.png"
PR_CURVE_PATH = MODELS_DIR / "pr_curve.png"
CM_PATH = MODELS_DIR / "confusion_matrix.png"
CALIBRATION_PATH = MODELS_DIR / "calibration_curve.png"
RESULTS_DASHBOARD_PATH = MODELS_DIR / "results_dashboard.png"
CLASS_METRICS_PATH = MODELS_DIR / "class_metrics.png"

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
BACKBONE_LR_FACTOR = 0.1
WEIGHT_DECAY = 1e-4
UNFREEZE_FROM_LAYER = 3
RANDOM_SEED = 42

DEFAULT_THRESHOLD = 0.5
