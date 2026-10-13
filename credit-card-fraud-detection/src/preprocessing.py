"""
src/preprocessing.py
=====================
Reproducible preprocessing pipeline shared by training and (indirectly, via
a saved pipeline) inference.

Leakage-prevention rules enforced here:
  * The dataset is split into train/validation/test BEFORE any scaler or
    resampler is fit.
  * Scalers are fit on the training split only, then applied (transform
    only) to validation/test.
  * Resampling (e.g. SMOTE) is applied to the TRAINING split only, never to
    validation or test data.
  * Feature order is fixed by config.FEATURE_COLUMNS and preserved end to
    end so inference always sees columns in the same order used at fit time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import FEATURE_COLUMNS, RANDOM_SEED, TARGET_COLUMN, TEST_SIZE, VALIDATION_SIZE

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    n_rows_before: int
    n_rows_after: int
    n_duplicates_removed: int
    n_invalid_rows_removed: int


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Apply documented, deterministic cleaning steps:
      1. Drop exact duplicate rows (assumption: duplicates represent the
         same transaction recorded twice and would otherwise leak between
         train/test if left in place before splitting).
      2. Drop rows with missing, non-numeric, or infinite values in any
         required feature or target column (assumption: these are a tiny
         fraction of rows in the real dataset and are not safely imputable
         without domain knowledge of the anonymized PCA features).
    """
    n_before = len(df)

    working = df.copy()
    n_duplicates = int(working.duplicated().sum())
    working = working.drop_duplicates().reset_index(drop=True)

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    numeric = working[required_cols].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy()).all(axis=1)
    n_invalid = int((~finite_mask).sum())

    cleaned = working.loc[finite_mask].reset_index(drop=True)
    # Ensure numeric dtypes after coercion (defensive; avoids object dtype columns).
    cleaned[required_cols] = cleaned[required_cols].apply(pd.to_numeric, errors="coerce")

    report = CleaningReport(
        n_rows_before=n_before,
        n_rows_after=len(cleaned),
        n_duplicates_removed=n_duplicates,
        n_invalid_rows_removed=n_invalid,
    )
    logger.info(
        "Cleaning: %d rows -> %d rows (removed %d duplicates, %d invalid rows)",
        report.n_rows_before, report.n_rows_after, report.n_duplicates_removed, report.n_invalid_rows_removed,
    )
    return cleaned, report


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target, enforcing fixed column order."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y


def stratified_random_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_SEED,
):
    """
    Produce stratified train/validation/test splits.

    The TEST split is isolated here and must only be touched once, for the
    final evaluation (see evaluate_model.py). Model selection and threshold
    tuning must use only train/validation data.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=validation_size, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def chronological_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
):
    """
    Produce an OPTIONAL chronological (time-ordered) split using the `Time`
    column, for comparison against the random stratified split.

    Rationale: the dataset's transactions are recorded in time order. A
    random split lets future transactions leak into the training set, which
    can make reported performance optimistic relative to a real deployment
    where the model only ever sees past transactions at prediction time.
    A chronological split better approximates real deployment conditions,
    at the cost of a (possibly) different class balance in each split.
    """
    ordered = df.sort_values("Time").reset_index(drop=True)
    n = len(ordered)
    n_test = int(round(n * test_size))
    n_val = int(round(n * validation_size * (1 - test_size)))

    test_df = ordered.iloc[n - n_test :]
    remaining = ordered.iloc[: n - n_test]
    val_df = remaining.iloc[len(remaining) - n_val :]
    train_df = remaining.iloc[: len(remaining) - n_val]

    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)
    return X_train, X_val, X_test, y_train, y_val, y_test
