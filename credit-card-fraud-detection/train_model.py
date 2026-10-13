#!/usr/bin/env python3
"""
train_model.py
===============
Train and select a fraud-detection model.

Usage (real dataset — the standard workflow for this project):
    python train_model.py

    Expects the real Kaggle dataset at data/raw/creditcard.csv. If it is
    missing, this script exits with a clear, non-fabricated error message
    and instructions, and does NOT create a fake model.

Usage (synthetic demo mode — ONLY for exercising the interface/tests when
the real dataset is unavailable):
    python train_model.py --synthetic-demo

    Generates a clearly-labeled synthetic dataset, trains on it, and saves
    a model artifact whose metadata is explicitly flagged as
    `"is_demo": true` / `"dataset_source": "synthetic_demo"`. The web
    application surfaces this flag prominently; synthetic results are never
    presented as evidence of real fraud-detection performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import joblib

import config
from src import evaluation, model_training, preprocessing
from src.data_loader import (
    DatasetNotFoundError,
    DatasetValidationError,
    generate_synthetic_dataset,
    load_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the credit card fraud detection model.")
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help=(
            "Train on a clearly-labeled synthetic dataset instead of the real "
            "Kaggle dataset. For interface demonstration and testing ONLY."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=config.Config.SYNTHETIC_DEMO_SAMPLES,
        help="Number of synthetic rows to generate when --synthetic-demo is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.ensure_directories()

    if args.synthetic_demo:
        logger.warning(
            "Running in SYNTHETIC DEMO mode. The dataset is randomly generated and "
            "carries NO real fraud signal. Resulting metrics are for interface "
            "demonstration and automated testing only, and must never be cited as "
            "real-world fraud-detection performance."
        )
        df = generate_synthetic_dataset(n_samples=args.samples, random_state=config.RANDOM_SEED)
        dataset_source = config.SYNTHETIC_DATASET_LABEL
        dataset_path_str = "<synthetically generated in-memory>"
    else:
        try:
            df, dataset_report = load_dataset(config.RAW_DATASET_PATH, source_label=config.REAL_DATASET_LABEL)
        except DatasetNotFoundError as exc:
            logger.error(str(exc))
            print(
                "\nERROR: Real dataset not found.\n"
                f"Expected file: {config.RAW_DATASET_PATH}\n\n"
                f"Download the dataset from Kaggle:\n  {config.KAGGLE_DATASET_URL}\n"
                f"and place the extracted 'creditcard.csv' at the path above.\n\n"
                "Alternatively, run:\n  python train_model.py --synthetic-demo\n"
                "to exercise the application end-to-end with clearly-labeled synthetic "
                "data (this does NOT produce real fraud-detection results).\n",
                file=sys.stderr,
            )
            return 1
        except DatasetValidationError as exc:
            logger.error("Dataset validation failed: %s", exc)
            print(f"\nERROR: Dataset failed validation: {exc}\n", file=sys.stderr)
            return 1

        dataset_source = config.REAL_DATASET_LABEL
        dataset_path_str = str(config.RAW_DATASET_PATH)
        for warning in dataset_report.warnings:
            logger.warning(warning)

    # --- Clean --------------------------------------------------------
    cleaned_df, cleaning_report = preprocessing.clean_dataset(df)
    logger.info(
        "Cleaned dataset: %d -> %d rows (%d duplicates, %d invalid rows removed)",
        cleaning_report.n_rows_before,
        cleaning_report.n_rows_after,
        cleaning_report.n_duplicates_removed,
        cleaning_report.n_invalid_rows_removed,
    )

    X, y = preprocessing.split_features_target(cleaned_df)

    # --- Split (train / validation / test) -----------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.stratified_random_split(X, y)
    logger.info(
        "Split sizes -> train: %d, validation: %d, test (held out): %d",
        len(X_train), len(X_val), len(X_test),
    )

    # --- Train & compare candidates on validation -----------------------
    candidates = model_training.get_candidate_pipelines(random_state=config.RANDOM_SEED)
    results = model_training.evaluate_candidates_on_validation(candidates, X_train, y_train, X_val, y_val)

    best_name = model_training.select_best_model(results)
    best_pipeline = results[best_name]["pipeline"]
    logger.info("Selected best model by validation average precision: %s", best_name)

    comparison_for_plot = {
        name: {
            "validation_average_precision": r["validation_average_precision"],
            "validation_f1_at_0.5": r["validation_f1_at_0.5"],
        }
        for name, r in results.items()
    }
    evaluation.plot_model_comparison(comparison_for_plot, config.MODEL_COMPARISON_PLOT)

    # --- Threshold tuning on validation only -----------------------------
    val_scores = model_training.get_positive_class_scores(best_pipeline, X_val)
    threshold_info = model_training.find_best_threshold(y_val, val_scores)
    logger.info(
        "Chosen decision threshold (validation F1-optimal): %.4f "
        "(precision=%.4f, recall=%.4f, F1=%.4f)",
        threshold_info["threshold"], threshold_info["precision"],
        threshold_info["recall"], threshold_info["f1"],
    )

    # --- Persist model + metadata ----------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, config.MODEL_PATH)

    metadata = {
        "model_name": best_name,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": config.FEATURE_COLUMNS,
        "target_column": config.TARGET_COLUMN,
        "decision_threshold": threshold_info["threshold"],
        "threshold_selection_method": "F1-optimal over validation precision-recall curve",
        "dataset_source": dataset_source,
        "dataset_path": dataset_path_str,
        "is_demo": args.synthetic_demo,
        "is_calibrated": False,
        "random_seed": config.RANDOM_SEED,
        "n_train_rows": int(len(X_train)),
        "n_validation_rows": int(len(X_val)),
        "n_test_rows": int(len(X_test)),
        "candidate_validation_comparison": comparison_for_plot,
        "imbalanced_learn_available": model_training.imblearn_available(),
    }
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Persist the test split to disk (features + target) so evaluate_model.py
    # can perform the FINAL evaluation without re-splitting (which would risk
    # an inconsistent or accidentally-reused test set across runs).
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    test_df = X_test.copy()
    test_df[config.TARGET_COLUMN] = y_test.values
    test_df.to_csv(config.DATA_PROCESSED_DIR / "held_out_test_split.csv", index=False)

    print("\n=== Training complete ===")
    print(f"Dataset source:        {dataset_source}{'  (SYNTHETIC — demo only)' if args.synthetic_demo else ''}")
    print(f"Selected model:        {best_name}")
    print(f"Decision threshold:    {threshold_info['threshold']:.4f}")
    print(f"Validation avg. precision (best model): {results[best_name]['validation_average_precision']:.4f}")
    print(f"Model artifact:        {config.MODEL_PATH}")
    print(f"Model metadata:        {config.MODEL_METADATA_PATH}")
    print(f"Held-out test split saved to: {config.DATA_PROCESSED_DIR / 'held_out_test_split.csv'}")
    print("Run 'python evaluate_model.py' next to compute final held-out test metrics.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
