#!/usr/bin/env python3
"""
app.py
======
Flask web application for the Credit Card Fraud Detection demo project.

Run directly for local development:
    python app.py

Or via the PowerShell launcher:
    .\\run.ps1
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

import config
from src import database, evaluation, prediction, validators
from src.data_loader import dataset_exists

logger = logging.getLogger("fraud_app")

BATCH_RESULTS_DIR = Path(tempfile.gettempdir()) / "fraud_app_batch_results"
_UUID_TOKEN_LENGTH = 32  # uuid4().hex length


def create_app(db_path: Path | str | None = None, testing: bool = False) -> Flask:
    """Application factory. Accepts an explicit db_path for isolated testing."""
    app = Flask(__name__)
    app.config.from_object(config.TestingConfig if testing else config.Config)

    resolved_db_path = Path(db_path) if db_path is not None else config.DATABASE_PATH
    app.config["DATABASE_PATH"] = resolved_db_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config.ensure_directories()
    BATCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    database.init_db(resolved_db_path)

    if app.config.get("SECRET_KEY_IS_DEV_FALLBACK"):
        logger.warning(
            "FRAUD_APP_SECRET_KEY is not set; using an insecure development fallback "
            "secret key. Set FRAUD_APP_SECRET_KEY before any real deployment."
        )

    _register_routes(app)
    _register_error_handlers(app)
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _try_load_model():
    """Attempt to load the model; return (LoadedModel | None, error_message | None)."""
    try:
        return prediction.load_model(), None
    except (prediction.ModelUnavailableError, prediction.ModelSchemaError) as exc:
        return None, str(exc)


def _dashboard_status(db_path) -> dict:
    loaded, model_error = _try_load_model()
    metrics = evaluation.load_metrics(config.METRICS_JSON_PATH)

    return {
        "dataset_available": dataset_exists(config.RAW_DATASET_PATH),
        "dataset_path": str(config.RAW_DATASET_PATH),
        "model_available": loaded is not None,
        "model_error": model_error,
        "model_metadata": loaded.metadata if loaded else None,
        "metrics_available": metrics is not None,
        "metrics": metrics,
        "recent_prediction_count": database.count_predictions(db_path),
    }


def _validate_batch_token(token: str) -> str:
    if len(token) != _UUID_TOKEN_LENGTH or not all(c in "0123456789abcdef" for c in token):
        abort(400, description="Invalid download token.")
    return token


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def _register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        status = _dashboard_status(app.config["DATABASE_PATH"])
        return render_template("index.html", active_page="dashboard", status=status)

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        feature_columns = config.FEATURE_COLUMNS
        result = None
        submitted_values = {col: "" for col in feature_columns}
        form_errors: list[str] = []

        if request.method == "POST":
            submitted_values = {col: request.form.get(col, "") for col in feature_columns}
            validation = validators.validate_transaction_payload(submitted_values)

            if not validation.is_valid:
                form_errors = validation.errors
            else:
                loaded, model_error = _try_load_model()
                if loaded is None:
                    form_errors = [
                        f"Prediction unavailable: {model_error}"
                    ]
                else:
                    result = prediction.predict_single(loaded, validation.cleaned)
                    database.insert_prediction(
                        app.config["DATABASE_PATH"],
                        source="single",
                        predicted_class=result["predicted_class"],
                        predicted_label=result["label"],
                        score=result["score"],
                        decision_threshold=result["decision_threshold"],
                        model_is_demo=bool(loaded.metadata.get("is_demo", False)),
                    )
                    result["model_is_demo"] = bool(loaded.metadata.get("is_demo", False))

        return render_template(
            "predict.html",
            active_page="predict",
            feature_columns=feature_columns,
            v_features=config.V_FEATURES,
            submitted_values=submitted_values,
            result=result,
            form_errors=form_errors,
        )

    @app.route("/batch", methods=["GET"])
    def batch_analysis():
        return render_template(
            "batch_analysis.html",
            active_page="batch",
            summary=None,
            download_token=None,
            form_errors=[],
            required_columns=config.FEATURE_COLUMNS,
            max_rows=app.config["MAX_BATCH_ROWS"],
            max_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
        )

    @app.route("/batch/analyze", methods=["POST"])
    def batch_analyze():
        import pandas as pd  # local import keeps module import graph lean at startup

        errors: list[str] = []
        summary = None
        download_token = None

        uploaded_file = request.files.get("csv_file")
        filename = uploaded_file.filename if uploaded_file else None
        errors.extend(validators.validate_uploaded_filename(filename))

        loaded, model_error = (None, None)
        if not errors:
            loaded, model_error = _try_load_model()
            if loaded is None:
                errors.append(f"Prediction unavailable: {model_error}")

        if not errors:
            safe_name = secure_filename(filename) or "upload.csv"
            with tempfile.TemporaryDirectory(prefix="fraud_upload_") as tmp_dir:
                tmp_path = Path(tmp_dir) / safe_name
                uploaded_file.save(tmp_path)
                try:
                    df = pd.read_csv(tmp_path)
                except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
                    errors.append(f"Could not parse the uploaded file as CSV: {exc}")
                    df = None

                if df is not None:
                    errors.extend(validators.validate_batch_dataframe(df, app.config["MAX_BATCH_ROWS"]))

                if not errors and df is not None:
                    result_df = prediction.predict_batch(loaded, df)

                    download_token = uuid.uuid4().hex
                    result_path = BATCH_RESULTS_DIR / f"{download_token}.csv"
                    result_df.to_csv(result_path, index=False)

                    n_total = len(result_df)
                    n_fraud = int((result_df["predicted_class"] == 1).sum())
                    summary = {
                        "n_total": n_total,
                        "n_fraud": n_fraud,
                        "n_legitimate": n_total - n_fraud,
                        "fraud_percentage": round((n_fraud / n_total) * 100, 3) if n_total else 0.0,
                        "model_is_demo": bool(loaded.metadata.get("is_demo", False)),
                        "decision_threshold": float(loaded.metadata.get("decision_threshold", 0.5)),
                    }
                # tmp_dir (and tmp_path within it) is removed automatically on exit.

        return render_template(
            "batch_analysis.html",
            active_page="batch",
            summary=summary,
            download_token=download_token,
            form_errors=errors,
            required_columns=config.FEATURE_COLUMNS,
            max_rows=app.config["MAX_BATCH_ROWS"],
            max_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
        )

    @app.route("/batch/download/<token>")
    def batch_download(token: str):
        token = _validate_batch_token(token)
        result_path = BATCH_RESULTS_DIR / f"{token}.csv"
        if not result_path.is_file():
            abort(404, description="This result file is no longer available. Please re-run the batch analysis.")

        # Read the full (size-limited) result into memory and delete the temp
        # file immediately, rather than relying on a streaming callback whose
        # timing relative to the WSGI response body is not guaranteed across
        # servers/test clients. This makes cleanup deterministic: by the time
        # this function returns, the temp file is already gone.
        csv_bytes = result_path.read_bytes()
        result_path.unlink(missing_ok=True)

        response = Response(csv_bytes, mimetype="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=fraud_batch_results.csv"
        return response

    @app.route("/performance")
    def model_performance():
        metrics = evaluation.load_metrics(config.METRICS_JSON_PATH)
        loaded, model_error = _try_load_model()
        comparison = None
        if loaded is not None:
            comparison = loaded.metadata.get("candidate_validation_comparison")

        plot_files = {
            "confusion_matrix": config.CONFUSION_MATRIX_PLOT.name if config.CONFUSION_MATRIX_PLOT.is_file() else None,
            "roc_curve": config.ROC_CURVE_PLOT.name if config.ROC_CURVE_PLOT.is_file() else None,
            "pr_curve": config.PR_CURVE_PLOT.name if config.PR_CURVE_PLOT.is_file() else None,
            "model_comparison": config.MODEL_COMPARISON_PLOT.name if config.MODEL_COMPARISON_PLOT.is_file() else None,
        }

        return render_template(
            "model_performance.html",
            active_page="performance",
            metrics=metrics,
            model_metadata=loaded.metadata if loaded else None,
            model_error=model_error,
            comparison=comparison,
            plot_files=plot_files,
        )

    @app.route("/reports/<path:filename>")
    def report_file(filename: str):
        return send_from_directory(config.REPORTS_DIR, filename)

    @app.route("/history")
    def transaction_history():
        records = database.get_recent_predictions(app.config["DATABASE_PATH"], limit=50)
        return render_template("transaction_history.html", active_page="history", records=records)

    @app.route("/about")
    def about():
        return render_template(
            "about.html",
            active_page="about",
            feature_columns=config.FEATURE_COLUMNS,
            kaggle_url=config.KAGGLE_DATASET_URL,
        )

    @app.route("/health")
    def health():
        loaded, model_error = _try_load_model()
        payload = {
            "status": "ok",
            "dataset_available": dataset_exists(config.RAW_DATASET_PATH),
            "model_available": loaded is not None,
            "model_is_demo": bool(loaded.metadata.get("is_demo", False)) if loaded else None,
            "database_reachable": _check_database(app.config["DATABASE_PATH"]),
        }
        return jsonify(payload), 200


def _check_database(db_path) -> bool:
    try:
        database.count_predictions(db_path)
        return True
    except Exception:  # noqa: BLE001 - health check must never raise
        return False


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(400)
    def bad_request(error):
        return render_template("500.html", message="Bad request.", code=400), 400

    @app.errorhandler(404)
    def not_found(error):
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def too_large(error):
        return render_template(
            "500.html",
            message=f"Uploaded file is too large. Maximum allowed size is "
            f"{app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB.",
            code=413,
        ), 413

    @app.errorhandler(500)
    def internal_error(error):
        logger.exception("Internal server error")
        return render_template("500.html", message="An unexpected error occurred.", code=500), 500


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
