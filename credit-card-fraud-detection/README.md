# Credit Card Fraud Detection Using Machine Learning

A final-year B.Tech CSE project: a locally runnable Flask web application that trains,
evaluates, and serves machine learning models for detecting fraudulent credit card
transactions, built around the public Kaggle "Credit Card Fraud Detection" dataset.

> **Educational prototype.** This is not a production payment-security or financial
> decision system. No PCI DSS or regulatory compliance is claimed. See
> [Security Considerations](#security-considerations) and [Limitations](#limitations).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Features](#features)
4. [Technology Stack](#technology-stack)
5. [Folder Structure](#folder-structure)
6. [Prerequisites](#prerequisites)
7. [Windows PowerShell Installation](#windows-powershell-installation)
8. [Dataset Download and Expected Path](#dataset-download-and-expected-path)
9. [Model Training](#model-training)
10. [Model Evaluation](#model-evaluation)
11. [Application Launch](#application-launch)
12. [Testing](#testing)
13. [Batch CSV Format](#batch-csv-format)
14. [API and Health-Check Endpoints](#api-and-health-check-endpoints)
15. [Troubleshooting](#troubleshooting)
16. [Security Considerations](#security-considerations)
17. [Limitations](#limitations)
18. [Future Enhancements](#future-enhancements)

---

## Project Overview

The application has three parts:

- **Training pipeline** (`train_model.py`, `evaluate_model.py`, `src/`) — loads and
  validates the dataset, cleans it, splits it (stratified train/validation/held-out test),
  trains and compares Logistic Regression and Random Forest (optionally SMOTE-augmented),
  selects the best model on validation data, tunes a decision threshold, and computes
  genuine final metrics on the held-out test split.
- **Flask web application** (`app.py`, `templates/`, `static/`) — a fintech-styled
  dashboard for single-transaction prediction, CSV batch analysis, model performance
  reporting, and a demonstration prediction history.
- **Automated tests** (`tests/`) — pytest coverage for preprocessing, prediction,
  validation, the database layer, model loading, and Flask routes, all using temporary/
  isolated fixtures so tests never require the real dataset or touch real artifacts.

## Problem Statement

Fraudulent transactions are a tiny fraction of all card transactions, making this a
severely imbalanced binary classification problem. A naive model that always predicts
"legitimate" can score very high **accuracy** while catching zero fraud — so this project
deliberately favors Precision, Recall, F1-score, ROC-AUC, and Average Precision (PR-AUC)
over accuracy when selecting and reporting on models.

## Features

- Robust dataset loader with explicit validation (required columns, valid target labels,
  missing/infinite values, duplicates) and an honest, non-fabricated dataset report.
- Leakage-safe preprocessing: train/validation/test splits happen before any scaling or
  resampling; resampling (when used) only ever touches the training fold.
- Two baseline/comparison models (Logistic Regression, Random Forest), plus an optional
  SMOTE-augmented Logistic Regression candidate when `imbalanced-learn` is installed.
- Validation-based model selection and F1-optimal decision-threshold tuning — the test
  set is touched only once, for final reporting.
- A fintech-styled, responsive, accessible dashboard with genuine KPI cards (never
  invented placeholder numbers).
- Single-transaction prediction form with full client- and server-side validation.
- CSV batch analysis with file-size/row limits, strict schema validation, safe temporary
  file handling, and a downloadable results CSV — uploaded data is never retained.
- Model performance page with real confusion matrix / ROC / Precision-Recall plots.
- SQLite-backed demonstration history storing only derived prediction outputs (never raw
  feature vectors or payment details).
- `/health` JSON endpoint reporting dataset/model/database availability.
- A clearly-labeled synthetic-data demo mode (`--synthetic-demo`) for exercising the full
  application and automated tests without the real dataset — never presented as real
  fraud-detection evidence.

## Technology Stack

- Python 3.11+ (tested on 3.12)
- Flask (backend/web)
- pandas, NumPy (data handling)
- scikit-learn (machine learning)
- imbalanced-learn (optional, for SMOTE)
- joblib (model persistence)
- SQLite (demonstration prediction history)
- HTML5, custom CSS3 (no external UI framework dependency, works offline), vanilla JS
- pytest (automated testing)
- matplotlib, seaborn (evaluation plots)

## Folder Structure

```
credit-card-fraud-detection/
├── app.py                     # Flask application (routes, app factory)
├── train_model.py             # Training entry point
├── evaluate_model.py          # Final held-out test evaluation entry point
├── config.py                  # Central paths, feature schema, settings
├── requirements.txt
├── README.md
├── .gitignore
├── run.ps1                     # Windows launcher: run the web app
├── train.ps1                   # Windows launcher: train + evaluate
├── pytest.ini
├── src/
│   ├── data_loader.py          # Real dataset loading/validation + synthetic generator
│   ├── preprocessing.py        # Cleaning, splitting, leakage prevention
│   ├── model_training.py       # Candidate pipelines, selection, threshold tuning
│   ├── evaluation.py           # Metrics computation, plots, save/load
│   ├── prediction.py           # Safe model loading + single/batch inference
│   ├── database.py             # SQLite data-access layer
│   └── validators.py           # Input validation (single + batch)
├── templates/                  # Jinja2 templates (dashboard, predict, batch, etc.)
├── static/css/style.css        # Custom fintech-style CSS
├── static/js/{app,charts}.js   # Vanilla JS: nav, form validation, SVG chart
├── tests/                       # pytest suite (isolated fixtures, no real data needed)
├── data/raw/                    # Place creditcard.csv here
├── data/processed/              # Held-out test split written by train_model.py
├── models/                      # model.joblib + model_metadata.json
├── reports/                     # metrics.json/csv + PNG plots
└── docs/final_year_project_report.md
```

## Prerequisites

- Windows 10/11 with PowerShell (instructions below), or any OS with Python 3.11+ and a
  terminal (adapt the manual commands accordingly).
- Python 3.11 or a compatible version actually installed on your machine (this project
  was developed and tested against Python 3.12).
- No internet access is required to run the app once dependencies are installed, except
  to download the dataset from Kaggle for real training.
- No paid services, cloud accounts, or API keys are required.

## Windows PowerShell Installation

Shortest path:

```powershell
cd credit-card-fraud-detection
.\run.ps1
```

`run.ps1` resolves its own directory, detects Python, creates `.venv` if missing,
installs `requirements.txt`, checks dataset/model status, and starts the app.

If PowerShell blocks script execution with an error like *"running scripts is
disabled on this system"*, use a **session-scoped** workaround (this does not change
any system-wide policy):

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### Manual workflow (equivalent, step by step)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

(If `py -3.11` is not available, substitute whichever compatible Python you have, e.g.
`python -m venv .venv`.)

## Dataset Download and Expected Path

This project does **not** download any dataset automatically and never requires Kaggle
credentials. To train on real data:

1. Visit the official Kaggle dataset page:
   <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
2. Download and extract `creditcard.csv`.
3. Place it at exactly:
   ```
   credit-card-fraud-detection/data/raw/creditcard.csv
   ```

The dataset contains `Time`, `V1`–`V28` (PCA-transformed features), `Amount`, and
`Class` (1 = fraud, 0 = legitimate). If the file is missing or fails validation
(wrong columns, unexpected target labels, unparsable CSV), training stops with a clear
error — it never fabricates a result.

**Optional synthetic demo mode:** if you want to explore the full application before
obtaining the real dataset, run training with `--synthetic-demo` (see below). This uses
a clearly-labeled, randomly generated dataset with the same schema, exclusively for
interface demonstration and automated testing. Its metrics are never evidence of real
fraud-detection performance.

## Model Training

```powershell
.\train.ps1
```

or manually:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

This loads and validates `data/raw/creditcard.csv`, cleans it (drops exact duplicates and
invalid rows, with the reasoning logged), splits it into stratified train/validation/test
sets, trains Logistic Regression and Random Forest (plus a SMOTE-augmented Logistic
Regression candidate if `imbalanced-learn` is installed), selects the best model by
validation Average Precision, tunes a decision threshold on validation data, and saves:

- `models/model.joblib` — the selected trained pipeline
- `models/model_metadata.json` — feature schema, threshold, dataset source, training config
- `data/processed/held_out_test_split.csv` — the isolated test split, for `evaluate_model.py`

**Synthetic demo mode** (no real dataset required):

```powershell
.\train.ps1 -SyntheticDemo
```

or manually: `python train_model.py --synthetic-demo`

## Model Evaluation

```powershell
.\.venv\Scripts\python.exe evaluate_model.py
```

Computes final metrics on the held-out test split (never used for model selection or
threshold tuning) and saves:

- `reports/metrics.json`, `reports/metrics.csv`
- `reports/confusion_matrix.png`, `reports/roc_curve.png`,
  `reports/precision_recall_curve.png`, `reports/model_comparison.png`

If training has not been run yet, this script exits with a clear error rather than
inventing numbers.

## Application Launch

```powershell
.\run.ps1
```

or manually:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open <http://127.0.0.1:5000> in a browser. The app starts even without a trained
model or dataset — the dashboard and relevant pages explain exactly what is missing and
how to fix it.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

The suite uses temporary directories, isolated SQLite databases, and a small in-memory
trained demo pipeline (via fixtures in `tests/conftest.py`), so **no test requires the
real Kaggle dataset**. Also run a compilation check:

```powershell
.\.venv\Scripts\python.exe -m compileall -q .
```

## Batch CSV Format

The uploaded CSV must contain exactly these columns (order does not matter, extra
columns such as `Class` are ignored for inference):

```
Time, V1, V2, V3, ..., V28, Amount
```

Limits (configurable via environment variables):

- Maximum rows per file: `FRAUD_APP_MAX_BATCH_ROWS` (default 20,000)
- Maximum upload size: `FRAUD_APP_MAX_UPLOAD_BYTES` (default 5 MB)

The output CSV adds `predicted_class`, `predicted_label`, `fraud_score`, and
`decision_threshold` columns with a stable schema. Uploaded files are processed in a
temporary directory that is deleted immediately after analysis; results are written to a
short-lived, randomly-named temp file and deleted immediately after download.

## API and Health-Check Endpoints

| Route                     | Method | Description                                                   |
|---------------------------|--------|-----------------------------------------------------------------|
| `/`                       | GET    | Dashboard                                                        |
| `/predict`                | GET/POST | Single-transaction prediction form                            |
| `/batch`                  | GET    | Batch CSV upload form                                            |
| `/batch/analyze`          | POST   | Runs batch inference, returns a summary + download link          |
| `/batch/download/<token>` | GET    | Downloads the batch result CSV once, then deletes it              |
| `/performance`            | GET    | Model performance report (metrics + plots)                      |
| `/history`                | GET    | Demonstration prediction history                                 |
| `/about`                  | GET    | Methodology, limitations, security notes                         |
| `/health`                 | GET    | JSON: `{status, dataset_available, model_available, model_is_demo, database_reachable}` |

## Troubleshooting

- **"running scripts is disabled on this system"** — use
  `powershell -ExecutionPolicy Bypass -File .\run.ps1` (session-scoped only).
- **`py -3.11` not found** — install Python 3.11+ from python.org, or edit the launcher's
  Python-detection candidates to match your installed version.
- **Dataset validation error** — confirm the file is the unmodified Kaggle CSV with
  columns `Time, V1..V28, Amount, Class`, saved at `data/raw/creditcard.csv`.
- **Model unavailable in the app** — train a model first (`.\train.ps1` or
  `.\train.ps1 -SyntheticDemo`).
- **Port 5000 already in use** — stop the other process, or edit the `app.run(...)` call
  at the bottom of `app.py` to use a different port.
- **Upload rejected as too large / too many rows** — reduce the file size or raise
  `FRAUD_APP_MAX_UPLOAD_BYTES` / `FRAUD_APP_MAX_BATCH_ROWS`.

## Security Considerations

- No real card numbers, CVVs, expiry dates, or cardholder names are collected anywhere.
- `FRAUD_APP_SECRET_KEY` should be set via an environment variable for any shared/
  deployed use; an insecure development fallback is used otherwise and a warning is
  logged.
- All SQL uses parameterized queries. Only derived prediction outputs are stored in the
  demonstration history — never raw feature vectors.
- Only the application's own trusted model artifact is ever deserialized; there is no
  route that loads an arbitrary user-supplied model file.
- Uploaded CSVs are validated, size/row-limited, processed in temporary storage, and
  deleted immediately after use; uploaded content is never executed.
- Error pages never expose stack traces or internal file paths.
- **This project makes no claim of PCI DSS compliance, production readiness, or
  guaranteed fraud detection.** It is an educational prototype.

## Limitations

- Random stratified evaluation may be optimistic relative to a real deployment, where the
  model only ever sees past transactions; `src/preprocessing.chronological_split` is
  provided for an optional, more deployment-realistic comparison.
- Model scores are raw outputs, not calibrated real-world probabilities, unless
  explicitly calibrated (this project does not calibrate probabilities).
- Without the real Kaggle dataset, the application can only demonstrate its interface and
  software behavior using synthetic data — genuine fraud-detection results require the
  real dataset.
- This is a single-machine demo; it is not designed for concurrent production traffic,
  horizontal scaling, or real-time transaction-stream integration.

## Future Enhancements

- Probability calibration (e.g. Platt scaling / isotonic regression) with calibration
  curve reporting.
- Additional models (gradient boosting, XGBoost/LightGBM) and hyperparameter search.
- A chronological-split evaluation surfaced directly in the web dashboard alongside the
  random-split results.
- Role-based authentication for any shared/multi-user deployment.
- Streaming/real-time scoring integration.
