"""
src/model_training.py
======================
Candidate model pipelines, validation-based model selection, and decision
threshold tuning.

Design notes:
  * Logistic Regression uses a StandardScaler (fit on training data only,
    inside the pipeline) because it is scale-sensitive.
  * Random Forest does not require scaling (tree splits are scale-invariant),
    so no scaler is included for it.
  * Class imbalance is handled primarily via `class_weight="balanced"`.
    When imbalanced-learn is installed, an additional SMOTE-augmented
    Logistic Regression candidate is included for comparison. Because SMOTE
    is wrapped inside an imbalanced-learn `Pipeline`, the resampling step
    activates ONLY during `.fit()` and is automatically skipped during
    `.predict()` / `.predict_proba()`, which structurally prevents
    resampling from ever touching validation or test data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED

logger = logging.getLogger(__name__)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    _IMBLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when imblearn is absent
    _IMBLEARN_AVAILABLE = False


def imblearn_available() -> bool:
    return _IMBLEARN_AVAILABLE


def get_candidate_pipelines(random_state: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Return the named candidate model pipelines to compare on validation data."""
    candidates: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if _IMBLEARN_AVAILABLE:
        candidates["logistic_regression_smote"] = ImbPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=random_state)),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        )
    else:
        logger.info("imbalanced-learn not available; skipping SMOTE candidate pipeline.")

    return candidates


def get_positive_class_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Return model scores for the positive (fraud) class, preferring
    predict_proba and falling back to a min-max-scaled decision_function.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        return np.zeros_like(scores)
    raise AttributeError("Pipeline does not support predict_proba or decision_function.")


def evaluate_candidates_on_validation(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, dict[str, Any]]:
    """
    Fit each candidate on the TRAINING split only, score it on the
    VALIDATION split, and return per-candidate fitted pipeline + metrics.

    Model selection must be based on this validation comparison, not the
    held-out test set, and not on accuracy alone (which is misleading for
    severely imbalanced fraud data).
    """
    results: dict[str, dict[str, Any]] = {}
    for name, pipeline in pipelines.items():
        logger.info("Fitting candidate model: %s", name)
        pipeline.fit(X_train, y_train)
        val_scores = get_positive_class_scores(pipeline, X_val)
        val_preds = (val_scores >= 0.5).astype(int)

        results[name] = {
            "pipeline": pipeline,
            "validation_average_precision": float(average_precision_score(y_val, val_scores)),
            "validation_f1_at_0.5": float(f1_score(y_val, val_preds, zero_division=0)),
        }
        logger.info(
            "Candidate %s -> validation average precision: %.4f, F1@0.5: %.4f",
            name,
            results[name]["validation_average_precision"],
            results[name]["validation_f1_at_0.5"],
        )
    return results


def select_best_model(results: dict[str, dict[str, Any]]) -> str:
    """
    Select the best candidate by validation average precision (area under
    the precision-recall curve), which is the standard, accuracy-independent
    metric for severely imbalanced binary classification such as fraud
    detection.
    """
    return max(results, key=lambda name: results[name]["validation_average_precision"])


def find_best_threshold(y_true: pd.Series, scores: np.ndarray) -> dict[str, Any]:
    """
    Scan the precision-recall curve (computed on VALIDATION data only) to
    choose the decision threshold that maximizes F1. Returns the chosen
    threshold plus the precision/recall achieved at that threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    f1_scores = np.zeros_like(thresholds, dtype=float)
    for i, t in enumerate(thresholds):
        p, r = precision[i], recall[i]
        f1_scores[i] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    best_idx = int(np.argmax(f1_scores))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1_scores[best_idx]),
    }
