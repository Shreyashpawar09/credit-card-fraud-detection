"""
config.py
=========
Central configuration for the Credit Card Fraud Detection project.

All paths are resolved relative to this file's location (BASE_DIR), so the
application does NOT depend on the current working directory from which it
is launched. No absolute, machine-specific paths are hardcoded.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths (resolved relative to this file, never the CWD)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = BASE_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"

# Expected default location of the real Kaggle dataset.
RAW_DATASET_PATH: Path = DATA_RAW_DIR / "creditcard.csv"
KAGGLE_DATASET_URL: str = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

# Model artifacts
MODEL_PATH: Path = MODELS_DIR / "model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

# Evaluation artifacts
METRICS_JSON_PATH: Path = REPORTS_DIR / "metrics.json"
METRICS_CSV_PATH: Path = REPORTS_DIR / "metrics.csv"
CONFUSION_MATRIX_PLOT: Path = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_PLOT: Path = REPORTS_DIR / "roc_curve.png"
PR_CURVE_PLOT: Path = REPORTS_DIR / "precision_recall_curve.png"
MODEL_COMPARISON_PLOT: Path = REPORTS_DIR / "model_comparison.png"

# SQLite database (file-backed; path can be overridden via environment
# variable for isolated test databases).
DATABASE_PATH: Path = Path(
    os.environ.get("FRAUD_APP_DB_PATH", str(BASE_DIR / "data" / "predictions.db"))
)

# ---------------------------------------------------------------------------
# Feature schema (single source of truth, shared by training & inference)
# ---------------------------------------------------------------------------
# The Kaggle "Credit Card Fraud Detection" dataset exposes Time, V1..V28
# (PCA-transformed features) and Amount as inputs, with Class as the target.
V_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time"] + V_FEATURES + ["Amount"]
TARGET_COLUMN = "Class"
ALL_REQUIRED_COLUMNS_FOR_TRAINING = FEATURE_COLUMNS + [TARGET_COLUMN]

# Valid values for the target column.
VALID_TARGET_LABELS = {0, 1}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Train / validation / test split configuration
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20   # held out, touched only for final evaluation
VALIDATION_SIZE: float = 0.20  # fraction of the remaining train+val data

# ---------------------------------------------------------------------------
# Decision threshold
# ---------------------------------------------------------------------------
DEFAULT_DECISION_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Synthetic data (demo / testing ONLY — never evidence of real performance)
# ---------------------------------------------------------------------------
SYNTHETIC_DATASET_LABEL = "synthetic_demo"
REAL_DATASET_LABEL = "kaggle_creditcard_real"

# ---------------------------------------------------------------------------
# Flask / web application configuration
# ---------------------------------------------------------------------------
class Config:
    """Base Flask configuration. Values are environment-overridable."""

    _env_secret = os.environ.get("FRAUD_APP_SECRET_KEY")
    if _env_secret:
        SECRET_KEY = _env_secret
        SECRET_KEY_IS_DEV_FALLBACK = False
    else:
        # Development-only fallback. NEVER use this in a real deployment.
        SECRET_KEY = "dev-only-insecure-secret-key-change-me"
        SECRET_KEY_IS_DEV_FALLBACK = True

    # Upload handling for CSV batch analysis
    MAX_CONTENT_LENGTH = int(os.environ.get("FRAUD_APP_MAX_UPLOAD_BYTES", 5 * 1024 * 1024))  # 5 MB
    ALLOWED_UPLOAD_EXTENSIONS = {"csv"}
    MAX_BATCH_ROWS = int(os.environ.get("FRAUD_APP_MAX_BATCH_ROWS", 20000))

    # Only relevant for the demonstration synthetic training path.
    SYNTHETIC_DEMO_SAMPLES = int(os.environ.get("FRAUD_APP_SYNTHETIC_SAMPLES", 20000))

    JSON_SORT_KEYS = False


class TestingConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False


def ensure_directories() -> None:
    """Create all runtime directories if they do not already exist."""
    for directory in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
