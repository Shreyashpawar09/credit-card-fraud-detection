#!/usr/bin/env python3
"""
evaluate_model.py
==================
Compute and save FINAL evaluation metrics on the held-out TEST split that
was set aside (and never touched) during train_model.py.

This script must be run only after train_model.py has produced:
  - models/model.joblib
  - models/model_metadata.json
  - data/processed/held_out_test_split.csv

It never fabricates metrics: if any prerequisite artifact is missing, it
exits with a clear, actionable error instead of inventing numbers.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

import config
from src import evaluation, model_training, prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_model")


def main() -> int:
    config.ensure_directories()
    test_split_path = config.DATA_PROCESSED_DIR / "held_out_test_split.csv"

    missing = []
    if not config.MODEL_PATH.is_file():
        missing.append(str(config.MODEL_PATH))
    if not config.MODEL_METADATA_PATH.is_file():
        missing.append(str(config.MODEL_METADATA_PATH))
    if not test_split_path.is_file():
        missing.append(str(test_split_path))

    if missing:
        print(
            "\nERROR: Cannot run final evaluation — missing required artifact(s):\n  "
            + "\n  ".join(missing)
            + "\n\nRun 'python train_model.py' first (optionally with --synthetic-demo "
            "for a demonstration-only run).\n",
            file=sys.stderr,
        )
        return 1

    try:
        loaded = prediction.load_model()
    except (prediction.ModelUnavailableError, prediction.ModelSchemaError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1

    test_df = pd.read_csv(test_split_path)
    X_test = test_df[config.FEATURE_COLUMNS]
    y_test = test_df[config.TARGET_COLUMN].astype(int)

    threshold = float(loaded.metadata.get("decision_threshold", config.DEFAULT_DECISION_THRESHOLD))
    scores = model_training.get_positive_class_scores(loaded.pipeline, X_test)

    metrics = evaluation.compute_metrics(y_test, scores, threshold, split_name="test")
    metrics["model_name"] = loaded.metadata.get("model_name")
    metrics["dataset_source"] = loaded.metadata.get("dataset_source")
    metrics["is_demo"] = loaded.metadata.get("is_demo", False)
    metrics["model_version"] = loaded.metadata.get("model_version")

    evaluation.save_metrics(metrics, config.METRICS_JSON_PATH, config.METRICS_CSV_PATH)
    evaluation.plot_confusion_matrix(
        y_test, scores, threshold, config.CONFUSION_MATRIX_PLOT,
        title=f"Confusion Matrix — Held-out Test Split ({len(y_test)} samples)",
    )
    evaluation.plot_roc_curve(y_test, scores, config.ROC_CURVE_PLOT, title="ROC Curve — Held-out Test Split")
    evaluation.plot_precision_recall_curve(
        y_test, scores, config.PR_CURVE_PLOT, title="Precision-Recall Curve — Held-out Test Split"
    )

    print("\n=== FINAL held-out TEST evaluation ===")
    if metrics["is_demo"]:
        print("*** SYNTHETIC DEMO MODEL — these metrics do NOT reflect real fraud-detection performance ***")
    print(f"Model:                 {metrics['model_name']}")
    print(f"Dataset source:        {metrics['dataset_source']}")
    print(f"Test samples:          {metrics['n_samples']} (fraud: {metrics['n_fraud']}, "
          f"{metrics['fraud_prevalence_percent']}% prevalence)")
    print(f"Decision threshold:    {metrics['decision_threshold']:.4f}")
    print(f"Precision:             {metrics['precision']:.4f}")
    print(f"Recall:                {metrics['recall']:.4f}")
    print(f"F1-score:              {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:               {metrics['roc_auc']}")
    print(f"Average Precision (PR-AUC): {metrics['average_precision']}")
    print(f"Confusion matrix:      {metrics['confusion_matrix']}")
    print(f"\nSaved metrics to:      {config.METRICS_JSON_PATH}")
    print(f"Saved plots to:        {config.REPORTS_DIR}\n")
    print(
        "NOTE: Accuracy is intentionally not emphasized here — for a dataset this "
        "imbalanced, a model that labels every transaction as legitimate could still "
        "score a very high accuracy while catching zero fraud. Precision, recall, F1, "
        "ROC-AUC, and Average Precision are the meaningful metrics for this task.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
