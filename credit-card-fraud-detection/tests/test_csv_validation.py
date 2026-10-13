"""
tests/test_csv_validation.py
==============================
Covers: uploaded filename/extension checks, CSV schema validation,
malformed and oversized batch handling.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src import validators


def test_validate_uploaded_filename_rejects_missing_filename():
    errors = validators.validate_uploaded_filename(None)
    assert errors and "No file" in errors[0]


def test_validate_uploaded_filename_rejects_wrong_extension():
    errors = validators.validate_uploaded_filename("transactions.txt")
    assert errors and "Unsupported file extension" in errors[0]


def test_validate_uploaded_filename_accepts_csv():
    errors = validators.validate_uploaded_filename("transactions.csv")
    assert errors == []


def test_validate_batch_dataframe_rejects_empty_dataframe():
    df = pd.DataFrame(columns=config.FEATURE_COLUMNS)
    errors = validators.validate_batch_dataframe(df, max_rows=1000)
    assert any("no data rows" in e for e in errors)


def test_validate_batch_dataframe_rejects_missing_columns():
    df = pd.DataFrame({"Time": [1, 2], "Amount": [10, 20]})  # missing V1..V28
    errors = validators.validate_batch_dataframe(df, max_rows=1000)
    assert any("missing required column" in e for e in errors)


def test_validate_batch_dataframe_rejects_oversized_batch(synthetic_df):
    errors = validators.validate_batch_dataframe(synthetic_df, max_rows=10)
    assert any("exceeds the maximum" in e for e in errors)


def test_validate_batch_dataframe_rejects_non_numeric_values(synthetic_df):
    corrupted = synthetic_df.head(20).copy()
    corrupted["V2"] = corrupted["V2"].astype(object)
    corrupted.loc[0, "V2"] = "not_numeric"
    corrupted.loc[1, "V3"] = float("inf")

    errors = validators.validate_batch_dataframe(corrupted, max_rows=1000)
    assert any("non-numeric, missing, or infinite" in e for e in errors)


def test_validate_batch_dataframe_rejects_negative_time_or_amount(synthetic_df):
    corrupted = synthetic_df.head(20).copy()
    corrupted.loc[0, "Time"] = -5.0

    errors = validators.validate_batch_dataframe(corrupted, max_rows=1000)
    assert any("negative" in e for e in errors)


def test_validate_batch_dataframe_accepts_valid_data(synthetic_df):
    valid = synthetic_df.head(50).copy()
    errors = validators.validate_batch_dataframe(valid, max_rows=1000)
    assert errors == []


def test_validate_batch_dataframe_does_not_require_target_column(synthetic_df):
    valid = synthetic_df.head(10).drop(columns=[config.TARGET_COLUMN]).copy()
    errors = validators.validate_batch_dataframe(valid, max_rows=1000)
    assert errors == []
