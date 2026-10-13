# Credit Card Fraud Detection Using Machine Learning

**Final-Year B.Tech CSE Project Report**

**Project Type:** Academic / Educational Prototype
**Status of this document:** Reflects the actual, executed state of the implementation at
the time of writing. Any figure in Section 13 (Results) is a genuinely computed value from
the run described there, never an estimate presented as a measurement.

---

## Abstract

Credit card fraud is rare relative to the volume of legitimate transactions, which makes
it a severely imbalanced binary classification problem: a model that never predicts fraud
can still score very high accuracy while being useless in practice. This project implements
a complete, locally runnable system — a data-validation and preprocessing pipeline, two
compared machine learning models (Logistic Regression and Random Forest, with an optional
SMOTE-augmented variant), a leakage-aware evaluation methodology, and a Flask web
application — for exploring supervised fraud detection on the public Kaggle "Credit Card
Fraud Detection" dataset. At the time of this report, the real Kaggle dataset was not
available in the development environment; the full pipeline and application were instead
exercised end-to-end using a clearly-labeled, randomly generated synthetic dataset matching
the real schema. All synthetic-mode results in this report are explicitly marked as such and
must not be read as evidence of real-world fraud-detection performance. Section 13 reports
exactly what was computed, and Section 14 states plainly what remains pending real data.

## 1. Introduction and Problem Statement

Financial institutions process enormous volumes of card transactions, a small fraction of
which are fraudulent. Manually reviewing every transaction is infeasible, and simple
rule-based systems struggle to keep pace with evolving fraud patterns. Supervised machine
learning offers a way to learn patterns distinguishing fraudulent from legitimate
transactions from historical, labeled data. The central technical challenge is severe class
imbalance: fraudulent transactions are a tiny minority, so naive optimization for accuracy
is actively misleading.

## 2. Objectives and Scope

Objectives:

1. Build a robust, validated data-loading pipeline for the Kaggle dataset schema.
2. Implement a reproducible, leakage-safe preprocessing pipeline.
3. Train and compare at least two models (Logistic Regression baseline, Random Forest),
   handling class imbalance via class weighting and/or training-only resampling.
4. Evaluate using metrics appropriate for imbalanced classification (precision, recall,
   F1, ROC-AUC, Average Precision), not accuracy alone.
5. Build a usable, secure, locally runnable web application exposing prediction, batch
   analysis, and reporting features.
6. Provide automated tests that do not require the real dataset.

Scope is explicitly limited to an educational prototype: no production payment-security
claims, no real-time transaction-stream integration, no regulatory compliance claims.

## 3. Background and Literature Overview

The reference dataset originates from work on calibrating probabilities for unbalanced
classification in the context of credit card fraud detection, produced by the Machine
Learning Group at Université Libre de Bruxelles (ULB) in collaboration with Worldline,
and distributed publicly via Kaggle [1]. Random Forests, used here as the comparison
model, were introduced by Breiman as an ensemble of decision trees trained on bootstrap
samples with randomized feature selection [2]. The Synthetic Minority Over-sampling
Technique (SMOTE), used here as an optional training-only resampling strategy, was
introduced by Chawla et al. as a way to generate synthetic minority-class examples rather
than naively duplicating them [3]. Both scikit-learn [4] and imbalanced-learn, its
imbalance-focused companion library, are used directly in this project's implementation.

## 4. Dataset Description

The target dataset is the public Kaggle "Credit Card Fraud Detection" dataset [1],
containing transactions made by European cardholders in September 2013. It has 31
columns: `Time` (seconds elapsed since the first transaction), `V1`–`V28` (the result of
a PCA transformation applied by the original authors for confidentiality, so individual
feature meanings are not interpretable), `Amount`, and the binary target `Class` (1 =
fraud, 0 = legitimate). Fraud accounts for roughly 0.17% of transactions in the real
dataset — a very severe imbalance. This project's data loader (`src/data_loader.py`)
validates exactly this schema and computes an honest summary (row/column counts, class
distribution, fraud percentage, descriptive statistics) rather than assuming the file is
well-formed.

**Dataset availability at the time of this report:** the real dataset file was not present
in the development environment (expected at `data/raw/creditcard.csv`). To exercise the
full pipeline and application, a synthetic dataset generator (`generate_synthetic_dataset`
in `src/data_loader.py`) was used instead, producing data with the identical column
schema and a similar fraud prevalence (~1.7%–5%, configurable) but with feature values
drawn from simple Gaussian/Gamma distributions — carrying no real fraud signal. This is
labeled `synthetic_demo` throughout the codebase, metadata, and UI.

## 5. System Requirements

- Python 3.11+ (developed and tested against Python 3.12)
- Flask, pandas, NumPy, scikit-learn, joblib, matplotlib, seaborn, pytest
- imbalanced-learn (optional; the SMOTE candidate model is skipped gracefully if absent)
- No GPU, cloud account, paid API, or Kaggle credentials required
- Approximately 150 MB disk space for dependencies; the real dataset itself is ~150 MB

## 6. System Architecture

The project separates concerns into:

- **`config.py`** — single source of truth for paths, the feature/target schema, split
  ratios, the random seed, and Flask settings.
- **`src/data_loader.py`** — real-dataset loading/validation and the synthetic generator.
- **`src/preprocessing.py`** — cleaning, feature/target separation, stratified random
  split, and an optional chronological split.
- **`src/model_training.py`** — candidate pipeline construction, validation-based
  comparison, and threshold tuning.
- **`src/evaluation.py`** — metric computation, plot generation, and metrics
  persistence/loading.
- **`src/prediction.py`** — safe model artifact loading and single/batch inference.
- **`src/database.py`** — SQLite access layer for the demonstration history.
- **`src/validators.py`** — input validation for both the web form and CSV uploads.
- **`train_model.py`** / **`evaluate_model.py`** — CLI entry points orchestrating the
  above for training and final evaluation, respectively.
- **`app.py`** — the Flask application factory and routes, consuming the `src/` modules.

This separation means training and inference share the same feature schema and
preprocessing logic (no duplicated logic between the two), and the web layer never embeds
ML logic directly.

## 7. Data Preprocessing Methodology

1. **Loading & validation** — required columns are checked explicitly; unexpected target
   labels or unparsable CSVs raise a clear, typed error rather than failing silently.
2. **Cleaning** — exact duplicate rows are dropped (to prevent the same transaction
   leaking across train/test), and rows with missing, non-numeric, or infinite values in
   any required column are dropped, with counts logged.
3. **Splitting** — a fixed random seed (42) is used throughout. The data is split into
   stratified train/validation/held-out test sets. The test set is isolated and touched
   only once, during final evaluation.
4. **Scaling/resampling** — fit on the training split only; Logistic Regression uses a
   `StandardScaler` inside its pipeline, while Random Forest uses none (tree splits are
   scale-invariant). When `imbalanced-learn` is available, a SMOTE-augmented Logistic
   Regression candidate wraps SMOTE inside an imbalanced-learn `Pipeline`, which
   structurally skips the resampling step during `.predict()`/`.predict_proba()`, so
   validation and test data are never resampled.
5. **Optional chronological split** — `src/preprocessing.chronological_split` orders
   transactions by `Time` and splits accordingly, to illustrate that a random split can be
   optimistic relative to a deployment where the model only ever sees past transactions.

## 8. Algorithm Explanations

- **Logistic Regression (baseline):** a linear model estimating the log-odds of fraud as
  a linear combination of (standardized) features, trained with `class_weight="balanced"`
  to compensate for class imbalance without altering the training data directly.
- **Random Forest (comparison model):** an ensemble of randomized decision trees [2],
  also trained with `class_weight="balanced"`, which can capture non-linear feature
  interactions without requiring feature scaling.
- **Logistic Regression + SMOTE (optional):** the same linear model, but trained on a
  resampled training fold where synthetic minority-class examples are generated via
  SMOTE [3] instead of class weighting.

## 9. System Implementation

All code is organized under `src/` as described in Section 6, with CLI entry points
`train_model.py` and `evaluate_model.py`, and a Flask app in `app.py`. Model artifacts are
persisted with `joblib` as a single fitted scikit-learn `Pipeline` object (so preprocessing
and the estimator travel together), alongside a JSON metadata file recording the exact
feature column order, the chosen decision threshold, the dataset source label
(`kaggle_creditcard_real` or `synthetic_demo`), training row counts, and whether the model
is a demo model (`is_demo: true/false`). `src/prediction.py` refuses to load a model whose
saved feature schema does not match the application's current schema, raising a typed
`ModelSchemaError` instead of silently mispredicting.

## 10. Web Application Modules

- **Dashboard (`/`):** live dataset/model/evaluation-report status; never shows invented
  KPI values — each card reflects an actual on-disk check.
- **Single Transaction Prediction (`/predict`):** a form for all 30 required features,
  validated client-side (JS) and server-side (`src/validators.py`); returns the predicted
  class, raw model score, configured threshold, and a plain-language interpretation. No
  card numbers, CVVs, expiry dates, or names are collected.
- **CSV Batch Analysis (`/batch`, `/batch/analyze`, `/batch/download/<token>`):** validates
  the uploaded file's extension, row count, and per-cell numeric/finite constraints;
  processes the file in a temporary directory that is deleted immediately after analysis;
  returns a downloadable results CSV via a one-time random token, deleted immediately
  after it is served.
- **Model Performance (`/performance`):** displays the validation model comparison and the
  final held-out test metrics/plots generated by `evaluate_model.py`, or an explicit
  "not yet evaluated" state with exact next steps if they do not exist.
- **Transaction History (`/history`):** lists recent single-transaction predictions from
  SQLite (timestamp, predicted class, score) — never raw feature vectors.
- **About (`/about`):** methodology, dataset, limitations, and security notes.
- **Health Check (`/health`):** JSON status for dataset/model/database availability.

## 11. Evaluation Methodology

Model **selection** compares candidates on the **validation** split only, using Average
Precision (the area under the Precision-Recall curve) as the primary metric, because it
remains meaningful under severe class imbalance while accuracy does not. The **decision
threshold** is then tuned, also on validation data only, by scanning the
precision-recall curve for the F1-maximizing threshold. The **held-out test split** is
evaluated exactly once, after model and threshold are both fixed, using
`evaluate_model.py`, which computes precision, recall, F1, ROC-AUC, Average Precision, and
the full confusion matrix, and saves genuine PNG plots (confusion matrix, ROC curve,
precision-recall curve, and a validation model-comparison bar chart).

## 12. Results and Interpretation

> **These results were produced in SYNTHETIC DEMO mode** (`train_model.py
> --synthetic-demo`), because the real Kaggle dataset was not present in the development
> environment at the time of this report. The synthetic dataset's "fraud" class is a
> deliberately separable, shifted Gaussian distribution used only to exercise the software
> pipeline end to end. **These numbers are not evidence of real-world fraud-detection
> performance** and should not be cited as such.

Run configuration: 20,000 synthetic rows, 1.72% synthetic fraud rate, random seed 42,
stratified split into 12,800 train / 3,200 validation / 4,000 held-out test rows.

**Validation comparison (Average Precision / F1 @ 0.5):**

| Model | Validation Avg. Precision (PR-AUC) | Validation F1 @ 0.5 |
|---|---|---|
| Logistic Regression | 0.9842 | 0.8780 |
| Random Forest | 1.0000 | 0.5333 |
| Logistic Regression + SMOTE | 0.9852 | 0.9310 |

Random Forest was selected (highest validation Average Precision). Its F1-optimal
decision threshold on validation data was **0.16** (precision 1.0000, recall 1.0000, F1
1.0000 at that threshold on validation).

**Final held-out test evaluation** (4,000 samples; 69 synthetic "fraud" rows, 1.725%
prevalence; threshold 0.16):

| Metric | Value |
|---|---|
| Precision | 1.0000 |
| Recall | 0.9855 |
| F1-score | 0.9927 |
| ROC-AUC | 0.99998 |
| Average Precision (PR-AUC) | 0.9988 |
| Confusion matrix | TN=3931, FP=0, FN=1, TP=68 |

**Interpretation (synthetic data only):** the near-perfect separation reflects that the
synthetic generator draws "fraud" rows from a distribution shifted well away from
"legitimate" rows on most of the 28 features simultaneously — an artificially easy
pattern that exists specifically so the full pipeline (training, threshold tuning,
evaluation, plotting, and the web UI) could be exercised and verified without the real
dataset. A model achieving this level of separation on the real, PCA-anonymized dataset
would be unusual; real-world fraud patterns are considerably more subtle and overlap much
more with legitimate behavior. These figures therefore demonstrate that the *software*
works correctly end-to-end, not that the *approach* achieves this performance on real
fraud.

## 13. Screenshots

Genuine PNG plots generated by this run are saved under `reports/` and are also rendered
live in the application's Model Performance page (`/performance`):
`confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, and
`model_comparison.png`. No screenshots of the web UI itself are embedded in this document;
the application can be run locally (see `README.md`) to view them directly.

## 14. Limitations

- **Empirical results on real data are pending.** The figures in Section 12 are from a
  synthetic run only. To obtain genuine results, download the real dataset from Kaggle
  [1], place it at `data/raw/creditcard.csv`, and run `python train_model.py` followed by
  `python evaluate_model.py` — no code changes are required.
- A random stratified split may be optimistic relative to a real deployment, where the
  model only ever sees past transactions; `src/preprocessing.chronological_split` is
  provided as an alternative but was not the split used for the numbers above.
- Model scores are raw outputs, not calibrated real-world probabilities.
- The synthetic generator exists solely to exercise the software and intentionally does
  not attempt to mimic the real dataset's actual (far more subtle) fraud signal.
- This is a single-machine demo, not designed for concurrent production traffic or
  real-time transaction-stream integration.

## 15. Security and Ethical Considerations

No real payment-card data (card numbers, CVVs, expiry dates, cardholder names) is
collected anywhere in this application. All SQL access is parameterized. Only derived
prediction outputs — never raw feature vectors — are stored in the demonstration history.
Only the application's own trusted model artifact is ever deserialized; there is no route
that loads an arbitrary user-supplied model file. Uploaded CSVs are size/row-limited,
validated, processed in temporary storage, and deleted immediately after use. Error pages
never expose stack traces or internal file paths. This project makes **no claim of PCI
DSS compliance, production readiness, or guaranteed fraud detection**, and should not be
used as the basis for real financial decisions.

## 16. Future Scope

- Probability calibration (Platt scaling / isotonic regression) with a calibration-curve
  report.
- Additional model families (gradient boosting) and systematic hyperparameter search.
- Surfacing the chronological-split evaluation directly in the dashboard alongside the
  random-split results, for a more deployment-realistic comparison.
- Authentication and role-based access for any shared/multi-user deployment.
- Streaming or real-time scoring integration.

## 17. Conclusion

This project delivers a complete, working, locally runnable fraud-detection pipeline and
web application: validated data loading, leakage-aware preprocessing, comparative model
training and selection on imbalance-appropriate metrics, genuine evaluation reporting, and
a secure, tested Flask interface. In the absence of the real Kaggle dataset in the
development environment, the system was fully exercised — and is reported on honestly —
using a clearly-labeled synthetic dataset; obtaining real-world results requires only
downloading the dataset and re-running the existing, unmodified training and evaluation
scripts.

## 18. References

[1] Machine Learning Group, Université Libre de Bruxelles (ULB), in collaboration with
Worldline. "Credit Card Fraud Detection." Kaggle.
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

[2] L. Breiman. "Random Forests." *Machine Learning*, 45(1):5–32, 2001.

[3] N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer. "SMOTE: Synthetic Minority
Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16:321–357, 2002.

[4] F. Pedregosa et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine
Learning Research*, 12:2825–2830, 2011.

---

*This report was generated as part of the project deliverable. Section 12's figures are
machine-computed outputs of the included code (`train_model.py --synthetic-demo` followed
by `evaluate_model.py`) at the time of writing, re-runnable by anyone with this repository.*
