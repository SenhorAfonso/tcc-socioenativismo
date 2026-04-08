"""
Shared classification metrics for LOSO evaluation loops.

Operates on flat lists or arrays of integer class labels.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    trues: NDArray,
    preds: NDArray,
) -> dict[str, float]:
    """
    Compute accuracy, macro F1, macro precision and macro recall.

    Parameters
    ----------
    trues : array-like of int
        Ground-truth class labels.
    preds : array-like of int
        Predicted class labels.

    Returns
    -------
    dict with keys 'accuracy', 'f1', 'precision', 'recall'.
    """
    return {
        "accuracy":  accuracy_score(trues, preds),
        "f1":        f1_score(trues, preds, average="macro", zero_division=0),
        "precision": precision_score(trues, preds, average="macro", zero_division=0),
        "recall":    recall_score(trues, preds, average="macro", zero_division=0),
    }


def print_subject_metrics(subject_id: str | int, metrics: dict[str, float]) -> None:
    """Print a one-line per-subject metrics summary."""
    print(
        f"Subject {subject_id} → "
        f"Acc={metrics['accuracy']:.3f} | "
        f"F1={metrics['f1']:.3f} | "
        f"Precision={metrics['precision']:.3f} | "
        f"Recall={metrics['recall']:.3f}"
    )


def print_aggregate_metrics(all_metrics: list[dict[str, float]]) -> None:
    """Print mean ± std across subjects for each metric."""
    keys = ["accuracy", "f1", "precision", "recall"]
    labels = {"accuracy": "Accuracy", "f1": "F1 Score", "precision": "Precision", "recall": "Recall"}

    print("\n=== MÉDIAS FINAIS ===")
    for k in keys:
        values = np.array([m[k] for m in all_metrics])
        print(f"  {labels[k]:10s} = {values.mean():.3f} ± {values.std():.3f}")


def get_classification_report_df(trues: NDArray, preds: NDArray):
    """
    Return a per-class metrics DataFrame from sklearn's classification_report.

    Parameters
    ----------
    trues, preds : array-like of int

    Returns
    -------
    pd.DataFrame with columns precision, recall, f1-score, accuracy (global).
    """
    import pandas as pd

    report = classification_report(trues, preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df_classes = df.iloc[:-3].copy()
    df_classes["accuracy"] = accuracy_score(trues, preds)
    return df_classes
