"""
tests/test_routes.py
======================
Covers: all page routes, the /health endpoint, prediction form handling,
CSV batch upload handling, downloadable results, and error handlers.
"""

from __future__ import annotations

import io

import pandas as pd

import config
from src import database


def test_health_route_returns_structured_json(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "dataset_available" in payload
    assert "model_available" in payload
    assert "database_reachable" in payload
    assert payload["database_reachable"] is True


def test_index_route_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Dashboard" in response.data


def test_predict_get_renders_form(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Time" in response.data
    assert b"V1" in response.data


def test_predict_post_missing_fields_shows_errors_not_crash(client):
    response = client.post("/predict", data={"Time": "100"})  # most fields missing
    assert response.status_code == 200
    assert b"Missing required field" in response.data or b"fix the following" in response.data


def test_predict_post_without_model_shows_unavailable_message(client, monkeypatch, tmp_path):
    # Explicitly point the prediction module at nonexistent artifacts so this
    # test is correct regardless of whether a real model has been trained
    # elsewhere in the project (e.g. via train_model.py).
    from src import prediction

    monkeypatch.setattr(prediction, "MODEL_PATH", tmp_path / "no_model_here.joblib")
    monkeypatch.setattr(prediction, "MODEL_METADATA_PATH", tmp_path / "no_meta_here.json")

    payload = {col: "0.1" for col in config.FEATURE_COLUMNS}
    payload["Time"] = "100"
    payload["Amount"] = "50"
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    assert b"Prediction unavailable" in response.data


def test_predict_post_valid_payload_with_model_logs_history(client, app_instance, trained_demo_model, tmp_db_path):
    payload = {col: "0.1" for col in config.FEATURE_COLUMNS}
    payload["Time"] = "100"
    payload["Amount"] = "50"

    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    assert b"Prediction Result" in response.data

    records = database.get_recent_predictions(tmp_db_path, limit=5)
    assert len(records) == 1
    assert records[0]["source"] == "single"


def test_batch_get_renders_form(client):
    response = client.get("/batch")
    assert response.status_code == 200
    assert b"CSV" in response.data


def test_batch_analyze_without_file_shows_error(client):
    response = client.post("/batch/analyze", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"No file was selected" in response.data


def test_batch_analyze_wrong_extension_shows_error(client):
    data = {"csv_file": (io.BytesIO(b"not,a,csv\n1,2,3"), "transactions.txt")}
    response = client.post("/batch/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Unsupported file extension" in response.data


def test_batch_analyze_without_model_shows_unavailable_message(client, monkeypatch, tmp_path):
    from src import prediction

    monkeypatch.setattr(prediction, "MODEL_PATH", tmp_path / "no_model_here.joblib")
    monkeypatch.setattr(prediction, "MODEL_METADATA_PATH", tmp_path / "no_meta_here.json")

    df = pd.DataFrame({col: [0.1, 0.2] for col in config.FEATURE_COLUMNS})
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    data = {"csv_file": (buffer, "transactions.csv")}
    response = client.post("/batch/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Prediction unavailable" in response.data


def test_batch_analyze_and_download_full_flow(client, trained_demo_model):
    df = pd.DataFrame({col: [0.1, -0.2, 0.3] for col in config.FEATURE_COLUMNS})
    df["Time"] = [1.0, 2.0, 3.0]
    df["Amount"] = [10.0, 20.0, 30.0]
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    data = {"csv_file": (buffer, "transactions.csv")}
    response = client.post("/batch/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Batch Summary" in response.data
    assert b"Download results CSV" in response.data

    # Extract the download token from the rendered link.
    html = response.data.decode("utf-8")
    start = html.find("/batch/download/")
    token = html[start + len("/batch/download/"): start + len("/batch/download/") + 32]

    download_response = client.get(f"/batch/download/{token}")
    assert download_response.status_code == 200
    assert download_response.mimetype == "text/csv"
    assert b"predicted_class" in download_response.data

    # File should be deleted after being downloaded once.
    second_attempt = client.get(f"/batch/download/{token}")
    assert second_attempt.status_code == 404


def test_batch_download_rejects_malformed_token(client):
    response = client.get("/batch/download/not-a-valid-token")
    assert response.status_code == 400


def test_performance_route_without_model_shows_unavailable(client, monkeypatch, tmp_path):
    from src import prediction

    monkeypatch.setattr(prediction, "MODEL_PATH", tmp_path / "no_model_here.joblib")
    monkeypatch.setattr(prediction, "MODEL_METADATA_PATH", tmp_path / "no_meta_here.json")

    response = client.get("/performance")
    assert response.status_code == 200
    assert b"No trained model is available" in response.data


def test_performance_route_with_model(client, trained_demo_model):
    response = client.get("/performance")
    assert response.status_code == 200
    assert b"Validation Model Comparison" in response.data


def test_history_route_empty_state(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert b"No predictions have been recorded yet" in response.data


def test_about_route_renders(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert b"Kaggle" in response.data


def test_unknown_route_returns_404_page(client):
    response = client.get("/this-route-does-not-exist")
    assert response.status_code == 404
    assert b"404" in response.data


def test_report_file_route_404_when_missing(client):
    response = client.get("/reports/does_not_exist.png")
    assert response.status_code == 404
