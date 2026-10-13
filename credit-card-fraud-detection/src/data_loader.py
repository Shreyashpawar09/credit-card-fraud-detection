"""
src/data_loader.py
===================
Loading and validating the Credit Card Fraud Detection dataset.

Two data sources are supported:

1. REAL dataset: the public Kaggle "Credit Card Fraud Detection" dataset
   (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), expected at
   `data/raw/creditcard.csv`. This is never downloaded automatically.

2. SYNTHETIC dataset: an explicitly-labeled, randomly generated dataset used
   ONLY for exercising the application's interface and automated tests when
   the real dataset is not available. Synthetic results must never be
   presented as evidence of real-world fraud-detection performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ALL_REQUIRED_COLUMNS_FOR_TRAINING,
    FEATURE_COLUMNS,
    RAW_DATASET_PATH,
    TARGET_COLUMN,
    VALID_TARGET_LABELS,
)

logger = logging.getLogger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected dataset file cannot be found on disk."""


class DatasetValidationError(ValueError):
    """Raised when the dataset exists but fails schema/content validation."""


@dataclass
class DatasetReport:
    """Structured, honest summary of a loaded dataset. No invented values."""

    source_label: str
    path: str
    n_rows: int
    n_columns: int
    columns: list[str]
    class_distribution: dict[str, int]
    fraud_percentage: float
    n_missing_values: int
    n_duplicate_rows: int
    describe: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_label": self.source_label,
            "path": self.path,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": self.columns,
            "class_distribution": self.class_distribution,
            "fraud_percentage": self.fraud_percentage,
            "n_missing_values": self.n_missing_values,
            "n_duplicate_rows": self.n_duplicate_rows,
            "describe": self.describe,
            "warnings": self.warnings,
        }


def dataset_exists(path: Path = RAW_DATASET_PATH) -> bool:
    return path.is_file()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in ALL_REQUIRED_COLUMNS_FOR_TRAINING if c not in df.columns]
    if missing:
        raise DatasetValidationError(
            "Dataset is missing required column(s): " + ", ".join(missing)
        )

    unexpected_targets = set(pd.unique(df[TARGET_COLUMN].dropna())) - VALID_TARGET_LABELS
    if unexpected_targets:
        raise DatasetValidationError(
            f"Target column '{TARGET_COLUMN}' contains unexpected label(s): "
            f"{sorted(unexpected_targets)}. Expected only {sorted(VALID_TARGET_LABELS)}."
        )


def load_dataset(path: Path = RAW_DATASET_PATH, source_label: str = "kaggle_creditcard_real") -> tuple[pd.DataFrame, DatasetReport]:
    """
    Load and validate the real transaction dataset from `path`.

    Raises:
        DatasetNotFoundError: if the file does not exist at the expected path.
        DatasetValidationError: if the file exists but fails validation
            (missing columns, unexpected target labels, unparsable CSV).
    """
    if not dataset_exists(path):
        raise DatasetNotFoundError(
            f"Dataset not found at expected path: {path}. "
            f"Download the real dataset from Kaggle and place it at this exact "
            f"location before training. See README.md for instructions."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError(f"Dataset file at {path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetValidationError(f"Dataset file at {path} could not be parsed as CSV: {exc}") from exc

    _validate_columns(df)

    warnings: list[str] = []

    n_missing = int(df[ALL_REQUIRED_COLUMNS_FOR_TRAINING].isna().sum().sum())
    if n_missing > 0:
        warnings.append(
            f"Dataset contains {n_missing} missing value(s) across required columns. "
            f"Assumption: rows with missing required values are dropped before training "
            f"(see preprocessing.clean_dataset)."
        )

    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        warnings.append(
            f"Dataset contains {n_duplicates} fully duplicated row(s). "
            f"Assumption: exact duplicate rows are removed before training to avoid "
            f"leaking the same transaction into both the train and test splits."
        )

    non_finite_mask = (
        df[FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .isna()
        & df[FEATURE_COLUMNS].notna()
    )
    n_non_finite = int(non_finite_mask.sum().sum())
    if n_non_finite > 0:
        warnings.append(
            f"Dataset contains {n_non_finite} non-numeric or infinite value(s) in "
            f"feature columns. These rows will be dropped during preprocessing."
        )

    report = _build_report(df, source_label=source_label, path=str(path), warnings=warnings)
    logger.info("Loaded dataset from %s: %d rows, %d columns", path, report.n_rows, report.n_columns)
    return df, report


def _build_report(df: pd.DataFrame, source_label: str, path: str, warnings: list[str]) -> DatasetReport:
    class_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
    class_distribution = {str(int(k)) if pd.notna(k) else "NaN": int(v) for k, v in class_counts.items()}
    n_fraud = int(df[TARGET_COLUMN].fillna(-1).eq(1).sum())
    fraud_pct = round((n_fraud / len(df)) * 100, 4) if len(df) else 0.0

    describe = df[FEATURE_COLUMNS].describe().to_dict()

    return DatasetReport(
        source_label=source_label,
        path=path,
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        columns=list(df.columns),
        class_distribution=class_distribution,
        fraud_percentage=fraud_pct,
        n_missing_values=int(df[ALL_REQUIRED_COLUMNS_FOR_TRAINING].isna().sum().sum()),
        n_duplicate_rows=int(df.duplicated().sum()),
        describe=describe,
        warnings=warnings,
    )


def generate_synthetic_dataset(
    n_samples: int = 20000,
    fraud_ratio: float = 0.0172,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a clearly-synthetic dataset matching the real dataset's SCHEMA
    (Time, V1..V28, Amount, Class) for interface demonstration and automated
    testing purposes ONLY.

    The fraud_ratio defaults to roughly the same class imbalance as the real
    Kaggle dataset (~0.172%) so that the demo application exercises the same
    imbalanced-learning code paths, but the feature values themselves are
    randomly generated and carry NO real-world fraud signal. Any metrics
    computed from this data must be labeled as synthetic/demo and must never
    be presented as real fraud-detection performance.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(round(n_samples * fraud_ratio)))
    n_legit = n_samples - n_fraud

    time_values = rng.uniform(0, 172800, size=n_samples)  # ~48 hours, like the real dataset

    # Legitimate transactions: PCA-like features centered near 0.
    legit_v = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    legit_amount = rng.gamma(shape=1.5, scale=60.0, size=n_legit)

    # Synthetic "fraud" transactions: shifted distribution so a classifier
    # has *something* learnable, purely for demo purposes.
    fraud_v = rng.normal(loc=1.5, scale=2.2, size=(n_fraud, 28))
    fraud_amount = rng.gamma(shape=1.1, scale=40.0, size=n_fraud)

    v_columns = np.vstack([legit_v, fraud_v])
    amounts = np.concatenate([legit_amount, fraud_amount])
    labels = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    df = pd.DataFrame(v_columns, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.round(time_values, 2))
    df["Amount"] = np.round(amounts, 2)
    df["Class"] = labels

    # Shuffle rows so fraud is not grouped at the end.
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df
