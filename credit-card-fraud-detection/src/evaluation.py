"""
src/evaluation.py
==================
Computes and persists genuine evaluation metrics and plots for a trained
pipeline against a held-out split (validation or test).

No metric in this module is ever fabricated: every value returned is
computed directly from `y_true` and model-produced scores/predictions.
If no trained model / evaluation has been run, callers must display an
honest "unavailable" state rather than invoke this module with fake data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering, safe for a web server process
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    split_name: str,
) -> dict[str, Any]:
    """Compute the full metrics bundle for one evaluation split."""
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    n_fraud = int((y_true == 1).sum())
    n_total = int(len(y_true))

    metrics: dict[str, Any] = {
        "split": split_name,
        "n_samples": n_total,
        "n_fraud": n_fraud,
        "fraud_prevalence_percent": round((n_fraud / n_total) * 100, 4) if n_total else 0.0,
        "decision_threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, scores)) if len(set(y_true)) > 1 else None,
        "average_precision_metric_name": "Average Precision (area under the Precision-Recall curve)",
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    return metrics


def plot_confusion_matrix(y_true: pd.Series, scores: np.ndarray, threshold: float, output_path: Path, title: str = "Confusion Matrix") -> None:
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: pd.Series, scores: np.ndarray, output_path: Path, title: str = "ROC Curve") -> None:
    if len(set(y_true)) < 2:
        logger.warning("Cannot plot ROC curve: y_true has only one class present.")
        return
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_value = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#1f4e79", linewidth=2, label=f"ROC curve (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_precision_recall_curve(y_true: pd.Series, scores: np.ndarray, output_path: Path, title: str = "Precision-Recall Curve") -> None:
    if len(set(y_true)) < 2:
        logger.warning("Cannot plot Precision-Recall curve: y_true has only one class present.")
        return
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap_value = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="#1f4e79", linewidth=2, label=f"PR curve (AP = {ap_value:.4f})")
    baseline = float((y_true == 1).sum()) / len(y_true)
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline prevalence = {baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(comparison: dict[str, dict[str, float]], output_path: Path, title: str = "Validation Model Comparison") -> None:
    """
    comparison: {model_name: {"validation_average_precision": float, "validation_f1_at_0.5": float}}
    """
    names = list(comparison.keys())
    ap_values = [comparison[n]["validation_average_precision"] for n in names]
    f1_values = [comparison[n]["validation_f1_at_0.5"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, ap_values, width, label="Avg. Precision (PR-AUC)", color="#1f4e79")
    ax.bar(x + width / 2, f1_values, width, label="F1 @ 0.5", color="#5b9bd5")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics(metrics: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    flat = {
        "split": metrics.get("split"),
        "n_samples": metrics.get("n_samples"),
        "n_fraud": metrics.get("n_fraud"),
        "fraud_prevalence_percent": metrics.get("fraud_prevalence_percent"),
        "decision_threshold": metrics.get("decision_threshold"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "tn": metrics.get("confusion_matrix", {}).get("tn"),
        "fp": metrics.get("confusion_matrix", {}).get("fp"),
        "fn": metrics.get("confusion_matrix", {}).get("fn"),
        "tp": metrics.get("confusion_matrix", {}).get("tp"),
    }
    pd.DataFrame([flat]).to_csv(csv_path, index=False)


def load_metrics(json_path: Path) -> dict[str, Any] | None:
    if not json_path.is_file():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
