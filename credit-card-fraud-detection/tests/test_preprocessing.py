"""
tests/test_preprocessing.py
============================
Covers: data schema validation, missing/malformed dataset handling,
preprocessing correctness, feature ordering, and target-leakage prevention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import preprocessing
from src.data_loader import DatasetNotFoundError, DatasetValidationError, load_dataset


# ---------------------------------------------------------------------------
# Dataset loading: missing / malformed handling (no real Kaggle data needed)
# ---------------------------------------------------------------------------
def test_load_dataset_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(DatasetNotFoundError):
        load_dataset(missing_path)


def test_load_dataset_missing_columns_raises(tmp_path):
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"Time": [1, 2], "Amount": [10, 20]}).to_csv(bad_path, index=False)
    with pytest.raises(DatasetValidationError):
        load_dataset(bad_path)


def test_load_dataset_unexpected_target_labels_raises(tmp_path, synthetic_df):
    bad_path = tmp_path / "bad_labels.csv"
    df = synthetic_df.copy()
    df.loc[0, "Class"] = 7  # invalid label
    df.to_csv(bad_path, index=False)
    with pytest.raises(DatasetValidationError):
        load_dataset(bad_path)


def test_load_dataset_valid_file_reports_honest_summary(tmp_path, synthetic_df):
    valid_path = tmp_path / "valid.csv"
    synthetic_df.to_csv(valid_path, index=False)

    df, report = load_dataset(valid_path, source_label="synthetic_demo")

    assert len(df) == len(synthetic_df)
    assert report.n_rows == len(synthetic_df)
    assert set(report.class_distribution.keys()) <= {"0", "1"}
    assert 0.0 <= report.fraud_percentage <= 100.0


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def test_clean_dataset_removes_exact_duplicates(synthetic_df):
    with_dupes = pd.concat([synthetic_df, synthetic_df.iloc[:5]], ignore_index=True)
    cleaned, report = preprocessing.clean_dataset(with_dupes)

    assert report.n_duplicates_removed == 5
    assert len(cleaned) == len(synthetic_df)


def test_clean_dataset_removes_invalid_rows(synthetic_df):
    corrupted = synthetic_df.copy()
    corrupted.loc[0, "V1"] = np.nan
    corrupted.loc[1, "Amount"] = np.inf
    corrupted["V5"] = corrupted["V5"].astype(object)
    corrupted.loc[2, "V5"] = "not_a_number"

    cleaned, report = preprocessing.clean_dataset(corrupted)

    assert report.n_invalid_rows_removed == 3
    assert len(cleaned) == len(synthetic_df) - 3
    assert cleaned[config.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").notna().all().all()


# ---------------------------------------------------------------------------
# Feature/target separation and leakage prevention
# ---------------------------------------------------------------------------
def test_split_features_target_excludes_target_and_preserves_order(synthetic_df):
    cleaned, _ = preprocessing.clean_dataset(synthetic_df)
    X, y = preprocessing.split_features_target(cleaned)

    assert config.TARGET_COLUMN not in X.columns
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


def test_stratified_random_split_isolates_test_set_and_preserves_balance(synthetic_df):
    cleaned, _ = preprocessing.clean_dataset(synthetic_df)
    X, y = preprocessing.split_features_target(cleaned)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.stratified_random_split(X, y)

    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(X)

    # No row index should appear in more than one split.
    train_idx, val_idx, test_idx = set(X_train.index), set(X_val.index), set(X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)

    overall_fraud_rate = y.mean()
    test_fraud_rate = y_test.mean()
    # Stratified split should keep the fraud rate reasonably close across splits.
    assert abs(overall_fraud_rate - test_fraud_rate) < 0.05


def test_chronological_split_respects_time_order(synthetic_df):
    cleaned, _ = preprocessing.clean_dataset(synthetic_df)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.chronological_split(cleaned)

    assert len(X_train) + len(X_val) + len(X_test) == len(cleaned)
    if len(X_train) and len(X_test):
        assert X_train["Time"].max() <= X_test["Time"].min()
