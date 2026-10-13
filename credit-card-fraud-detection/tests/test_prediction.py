"""
tests/test_prediction.py
=========================
Covers: single prediction success/invalid inputs, batch prediction output
schema, and target-column exclusion from inference.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src import prediction, validators


def test_predict_single_returns_expected_keys(trained_demo_model):
    payload = {col: 0.1 for col in config.FEATURE_COLUMNS}
    payload["Time"] = 1000.0
    payload["Amount"] = 50.0

    validation = validators.validate_transaction_payload(payload)
    assert validation.is_valid

    loaded = prediction.load_model()
    result = prediction.predict_single(loaded, validation.cleaned)

    assert set(result.keys()) == {
        "predicted_class", "label", "score", "decision_threshold", "is_calibrated_probability",
    }
    assert result["predicted_class"] in (0, 1)
    assert result["label"] in ("Fraud", "Legitimate")
    assert 0.0 <= result["score"] <= 1.0


def test_predict_single_invalid_payload_is_rejected():
    payload = {col: 0.1 for col in config.FEATURE_COLUMNS}
    del payload["V10"]  # missing field

    validation = validators.validate_transaction_payload(payload)

    assert not validation.is_valid
    assert any("V10" in err for err in validation.errors)


def test_predict_single_rejects_non_numeric_and_infinite_values():
    payload = {col: 0.1 for col in config.FEATURE_COLUMNS}
    payload["V3"] = "abc"
    payload["V4"] = float("inf")

    validation = validators.validate_transaction_payload(payload)

    assert not validation.is_valid
    assert len(validation.errors) == 2


def test_predict_batch_ignores_target_column_for_inference(trained_demo_model):
    loaded = prediction.load_model()

    rows = {col: [0.1, 0.2, -0.3] for col in config.FEATURE_COLUMNS}
    rows["Time"] = [10.0, 20.0, 30.0]
    rows["Amount"] = [5.0, 15.0, 500.0]
    rows[config.TARGET_COLUMN] = [1, 0, 1]  # present but must not be required or trusted
    df = pd.DataFrame(rows)

    result_df = prediction.predict_batch(loaded, df)

    assert "predicted_class" in result_df.columns
    assert "predicted_label" in result_df.columns
    assert "fraud_score" in result_df.columns
    assert "decision_threshold" in result_df.columns
    # Original target column must be renamed, never silently used as if it were a prediction.
    assert config.TARGET_COLUMN not in result_df.columns
    assert f"{config.TARGET_COLUMN}_provided_in_upload" in result_df.columns
    assert len(result_df) == 3
    assert result_df["predicted_class"].isin([0, 1]).all()


def test_predict_batch_works_without_target_column(trained_demo_model):
    loaded = prediction.load_model()
    rows = {col: [0.0, 0.5] for col in config.FEATURE_COLUMNS}
    rows["Time"] = [1.0, 2.0]
    rows["Amount"] = [1.0, 2.0]
    df = pd.DataFrame(rows)  # no Class column at all

    result_df = prediction.predict_batch(loaded, df)
    assert len(result_df) == 2
    assert "predicted_class" in result_df.columns
