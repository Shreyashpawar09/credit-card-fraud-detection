"""
tests/test_model_loading.py
=============================
Covers: model loading, missing-model behavior, corrupted artifact handling,
and feature-schema compatibility checks.
"""

from __future__ import annotations

import json

import joblib
import pytest

from src import prediction


def test_model_artifacts_exist_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "MODEL_PATH", tmp_path / "missing_model.joblib")
    monkeypatch.setattr(prediction, "MODEL_METADATA_PATH", tmp_path / "missing_meta.json")
    assert prediction.model_artifacts_exist() is False


def test_model_artifacts_exist_true_when_present(trained_demo_model):
    assert prediction.model_artifacts_exist() is True


def test_load_model_missing_file_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "MODEL_PATH", tmp_path / "missing_model.joblib")
    monkeypatch.setattr(prediction, "MODEL_METADATA_PATH", tmp_path / "missing_meta.json")

    with pytest.raises(prediction.ModelUnavailableError):
        prediction.load_model()


def test_load_model_missing_metadata_raises(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    joblib.dump({"not": "a real pipeline, just exercising missing-metadata path"}, model_path)

    monkeypatch.setattr(prediction, "MODEL_PATH", model_path)
    monkeypatch.setattr(prediction, "MODEL_METADATA_PATH", tmp_path / "missing_meta.json")

    with pytest.raises(prediction.ModelUnavailableError):
        prediction.load_model()


def test_load_model_corrupted_file_is_handled_gracefully(tmp_path, monkeypatch):
    model_path = tmp_path / "corrupted_model.joblib"
    metadata_path = tmp_path / "meta.json"
    model_path.write_bytes(b"this is not a valid joblib/pickle file at all")
    metadata_path.write_text(json.dumps({"feature_columns": []}))

    monkeypatch.setattr(prediction, "MODEL_PATH", model_path)
    monkeypatch.setattr(prediction, "MODEL_METADATA_PATH", metadata_path)

    with pytest.raises(prediction.ModelUnavailableError):
        prediction.load_model()


def test_load_model_schema_mismatch_raises(tmp_path, monkeypatch, trained_demo_model):
    # Reuse the trained demo model's pipeline file, but with mismatched metadata.
    metadata_path = tmp_path / "mismatched_meta.json"
    metadata_path.write_text(json.dumps({"feature_columns": ["Time", "Amount"]}))

    monkeypatch.setattr(prediction, "MODEL_PATH", trained_demo_model["model_path"])
    monkeypatch.setattr(prediction, "MODEL_METADATA_PATH", metadata_path)

    with pytest.raises(prediction.ModelSchemaError):
        prediction.load_model()


def test_load_model_success_returns_pipeline_and_metadata(trained_demo_model):
    loaded = prediction.load_model()
    assert loaded.pipeline is not None
    assert loaded.metadata["is_demo"] is True
    assert hasattr(loaded.pipeline, "predict_proba") or hasattr(loaded.pipeline, "decision_function")
