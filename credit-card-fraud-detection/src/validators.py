"""
src/validators.py
==================
Input validation for single-transaction predictions and CSV batch uploads.

All validation here is defensive: it never trusts client input, never
executes uploaded content, and never leaks internal paths or stack traces
back to the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config import FEATURE_COLUMNS, Config


class ValidationError(Exception):
    """Raised when user-supplied input fails validation."""


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    cleaned: dict[str, float] | None = None


def _is_finite_number(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def validate_transaction_payload(payload: dict[str, Any]) -> ValidationResult:
    """
    Validate a single-transaction input dict against the required feature
    schema (Time, V1..V28, Amount).

    Rejects: missing fields, non-numeric values, NaN, infinite values.
    Applies documented range sanity-checks: Time and Amount must be
    non-negative (both are inherently non-negative in the source dataset).
    """
    errors: list[str] = []
    cleaned: dict[str, float] = {}

    missing = [col for col in FEATURE_COLUMNS if col not in payload or payload[col] in (None, "")]
    if missing:
        errors.append(f"Missing required field(s): {', '.join(missing)}")

    for col in FEATURE_COLUMNS:
        if col in missing:
            continue
        raw_value = payload[col]
        if not _is_finite_number(raw_value):
            errors.append(f"Field '{col}' must be a finite number (got: {raw_value!r})")
            continue
        value = float(raw_value)
        if col in ("Time", "Amount") and value < 0:
            errors.append(f"Field '{col}' must be non-negative (got: {value})")
            continue
        cleaned[col] = value

    if errors:
        return ValidationResult(is_valid=False, errors=errors, cleaned=None)
    return ValidationResult(is_valid=True, errors=[], cleaned=cleaned)


def validate_uploaded_filename(filename: str | None) -> list[str]:
    """Validate the uploaded file has an allowed extension and a safe name."""
    errors: list[str] = []
    if not filename:
        errors.append("No file was selected for upload.")
        return errors

    if "." not in filename:
        errors.append("Uploaded file has no extension; a .csv file is required.")
        return errors

    extension = filename.rsplit(".", 1)[-1].lower()
    if extension not in Config.ALLOWED_UPLOAD_EXTENSIONS:
        errors.append(
            f"Unsupported file extension '.{extension}'. Only .csv files are accepted."
        )
    return errors


def validate_batch_dataframe(df: pd.DataFrame, max_rows: int) -> list[str]:
    """
    Validate a parsed CSV DataFrame intended for batch inference.

    Checks: required columns present, target column not required, row-count
    limit, numeric + finite values in every feature column, no fully-empty
    file.
    """
    errors: list[str] = []

    if df.empty:
        errors.append("The uploaded CSV file contains no data rows.")
        return errors

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        errors.append(
            "The uploaded CSV is missing required column(s): "
            + ", ".join(missing_columns)
        )
        # Can't safely continue numeric checks without the columns present.
        return errors

    if len(df) > max_rows:
        errors.append(
            f"The uploaded CSV has {len(df)} rows, which exceeds the maximum "
            f"allowed of {max_rows} rows per batch."
        )
        return errors

    numeric_subset = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    non_numeric_mask = numeric_subset.isna()
    infinite_mask = numeric_subset.isin([float("inf"), float("-inf")])

    bad_rows = (non_numeric_mask.any(axis=1) | infinite_mask.any(axis=1))
    n_bad = int(bad_rows.sum())
    if n_bad > 0:
        bad_row_indices = df.index[bad_rows][:10].tolist()
        errors.append(
            f"{n_bad} row(s) contain non-numeric, missing, or infinite values in "
            f"required feature columns. First affected row indices: {bad_row_indices}"
        )

    negative_time_or_amount = (
        (numeric_subset["Time"] < 0).fillna(False) | (numeric_subset["Amount"] < 0).fillna(False)
    )
    n_negative = int(negative_time_or_amount.sum())
    if n_negative > 0:
        errors.append(
            f"{n_negative} row(s) have a negative 'Time' or 'Amount' value, which is invalid."
        )

    return errors
