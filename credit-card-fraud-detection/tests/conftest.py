"""
tests/conftest.py
==================
Shared pytest fixtures. All fixtures use temporary directories/databases so
tests never touch the real project database, real model artifacts, or
require the real Kaggle dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pytest

# Ensure the project root is importable regardless of the directory pytest
# was invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import model_training, prediction, preprocessing  # noqa: E402
from src.data_loader import generate_synthetic_dataset  # noqa: E402


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_predictions.db"


@pytest.fixture
def app_instance(tmp_db_path: Path):
    from app import create_app

    application = create_app(db_path=tmp_db_path, testing=True)
    application.config.update(TESTING=True)
    yield application


@pytest.fixture
def client(app_instance):
    return app_instance.test_client()


@pytest.fixture
def synthetic_df():
    """A small, clearly-synthetic dataset matching the real schema, for tests only."""
    return generate_synthetic_dataset(n_samples=600, fraud_ratio=0.05, random_state=1)


@pytest.fixture
def trained_demo_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_df):
    """
    Train a small pipeline on synthetic data and point src.prediction's
    model artifact paths at temporary files, so tests exercise real
    load/predict code paths without touching the project's real model
    artifacts.
    """
    cleaned, _ = preprocessing.clean_dataset(synthetic_df)
    X, y = preprocessing.split_features_target(cleaned)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.stratified_random_split(X, y)

    pipeline = model_training.get_candidate_pipelines()["logistic_regression"]
    pipeline.fit(X_train, y_train)

    val_scores = model_training.get_positive_class_scores(pipeline, X_val)
    threshold_info = model_training.find_best_threshold(y_val, val_scores)

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": "logistic_regression",
        "model_version": "test",
        "feature_columns": config.FEATURE_COLUMNS,
        "target_column": config.TARGET_COLUMN,
        "decision_threshold": threshold_info["threshold"],
        "is_demo": True,
        "is_calibrated": False,
        "dataset_source": config.SYNTHETIC_DATASET_LABEL,
        "candidate_validation_comparison": {
            "logistic_regression": {"validation_average_precision": 0.5, "validation_f1_at_0.5": 0.5}
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    monkeypatch.setattr(prediction, "MODEL_PATH", model_path)
    monkeypatch.setattr(prediction, "MODEL_METADATA_PATH", metadata_path)

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
