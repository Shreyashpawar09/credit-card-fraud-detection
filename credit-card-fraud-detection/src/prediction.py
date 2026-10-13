"""
src/prediction.py
==================
Safe model loading and inference for both single-transaction and batch
predictions.

Security notes:
  * Only the application's own trusted artifact at config.MODEL_PATH is ever
    loaded. There is no route or function that deserializes an arbitrary,
    user-supplied model file.
  * Loading failures (missing file, corrupted file, incompatible schema)
    are handled gracefully and surfaced as clear, structured errors rather
    than raw stack traces.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, MODEL_METADATA_PATH, MODEL_PATH, TARGET_COLUMN

logger = logging.getLogger(__name__)


class ModelUnavailableError(Exception):
    """Raised when no usable trained model artifact is available."""


class ModelSchemaError(Exception):
    """Raised when the loaded model's expected feature schema does not match config."""


@dataclass
class LoadedModel:
    pipeline: Any
    metadata: dict[str, Any]


def model_artifacts_exist() -> bool:
    return MODEL_PATH.is_file() and MODEL_METADATA_PATH.is_file()


def load_model() -> LoadedModel:
    """
    Load the trained pipeline and its metadata from disk.

    Raises:
        ModelUnavailableError: if the artifact files are missing or cannot
            be deserialized.
        ModelSchemaError: if the saved feature schema does not match the
            application's current expected schema.
    """
    if not MODEL_PATH.is_file():
        raise ModelUnavailableError(
            f"No trained model found at {MODEL_PATH}. Train a model first "
            f"(see README.md / train_model.py)."
        )
    if not MODEL_METADATA_PATH.is_file():
        raise ModelUnavailableError(
            f"Model file exists at {MODEL_PATH} but its metadata file is missing "
            f"at {MODEL_METADATA_PATH}. Retrain the model to regenerate both files."
        )

    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - we deliberately convert any load failure
        raise ModelUnavailableError(f"Failed to load model artifact: {exc}") from exc

    try:
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(f"Failed to load model metadata: {exc}") from exc

    saved_features = metadata.get("feature_columns")
    if saved_features != FEATURE_COLUMNS:
        raise ModelSchemaError(
            "The saved model's feature schema does not match the application's "
            "current expected feature schema. Retrain the model to regenerate a "
            "compatible artifact."
        )

    return LoadedModel(pipeline=pipeline, metadata=metadata)


def predict_single(loaded: LoadedModel, features: dict[str, float]) -> dict[str, Any]:
    """Run inference for a single transaction (already-validated feature dict)."""
    row = pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    threshold = float(loaded.metadata.get("decision_threshold", 0.5))
    score = _get_positive_class_score(loaded.pipeline, row)[0]
    predicted_class = int(score >= threshold)

    return {
        "predicted_class": predicted_class,
        "label": "Fraud" if predicted_class == 1 else "Legitimate",
        "score": float(score),
        "decision_threshold": threshold,
        "is_calibrated_probability": bool(loaded.metadata.get("is_calibrated", False)),
    }


def predict_batch(loaded: LoadedModel, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference for a batch of already-validated rows (CSV upload path).
    The target column, if present, is ignored for inference and never
    required.
    """
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    threshold = float(loaded.metadata.get("decision_threshold", 0.5))
    scores = _get_positive_class_score(loaded.pipeline, X)
    predicted_classes = (scores >= threshold).astype(int)

    result = df.copy()
    result["predicted_class"] = predicted_classes
    result["predicted_label"] = np.where(predicted_classes == 1, "Fraud", "Legitimate")
    result["fraud_score"] = np.round(scores, 6)
    result["decision_threshold"] = threshold

    # Never echo the target column back as though it were a model output.
    if TARGET_COLUMN in result.columns:
        result = result.rename(columns={TARGET_COLUMN: f"{TARGET_COLUMN}_provided_in_upload"})

    return result


def _get_positive_class_score(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        s_min, s_max = float(np.min(scores)), float(np.max(scores))
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        return np.zeros(len(X))
    raise ModelUnavailableError("Loaded model does not support probability or decision scoring.")
