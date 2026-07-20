"""
Shared classification metrics for LOSO evaluation loops.

Operates on flat lists or arrays of integer class labels.
"""

import csv
from pathlib import Path

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
    """Print mean, std, min and max across subjects for each metric."""
    keys = ["accuracy", "f1", "precision", "recall"]
    labels = {"accuracy": "Accuracy", "f1": "F1 Score", "precision": "Precision", "recall": "Recall"}

    print("\n=== MÉDIAS FINAIS ===")
    for k in keys:
        values = np.array([m[k] for m in all_metrics])
        print(
            f"  {labels[k]:10s}: "
            f"média={values.mean():.4f} | "
            f"std={values.std():.4f} | "
            f"min={values.min():.4f} | "
            f"max={values.max():.4f} | "
            f"faixa=[{values.min():.4f} – {values.max():.4f}]"
        )


def save_loso_csv(
    subjects: list,
    all_metrics: list[dict[str, float]],
    output_path: str | Path,
) -> None:
    """
    Persist per-subject LOSO metrics to a CSV file and print a summary table.

    The file contains one row per subject with columns:
        subject, accuracy, f1, precision, recall

    A footer block with mean, std, min and max is appended as comment lines
    (prefixed with ``#``) so the numeric rows remain machine-readable.

    Parameters
    ----------
    subjects : list
        Subject identifiers in the same order as ``all_metrics``.
    all_metrics : list[dict[str, float]]
        Per-subject metric dicts (keys: accuracy, f1, precision, recall).
    output_path : str or Path
        Destination CSV file.  Parent directories are created automatically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keys   = ["accuracy", "f1", "precision", "recall"]
    labels = {"accuracy": "Accuracy", "f1": "F1 Score", "precision": "Precision", "recall": "Recall"}

    # ── Print per-subject table ────────────────────────────────────────────
    col_w = 12
    header = f"{'Sujeito':<12}" + "".join(f"{labels[k]:>{col_w}}" for k in keys)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for subj, m in zip(subjects, all_metrics):
        row = f"{str(subj):<12}" + "".join(f"{m[k]:>{col_w}.4f}" for k in keys)
        print(row)
    print("=" * len(header))

    # ── Print aggregate summary ────────────────────────────────────────────
    print("\n── Resumo agregado ──")
    for k in keys:
        vals = np.array([m[k] for m in all_metrics])
        print(
            f"  {labels[k]:10s}: "
            f"média={vals.mean():.4f} | "
            f"std={vals.std():.4f} | "
            f"min={vals.min():.4f} | "
            f"max={vals.max():.4f}"
        )

    # ── Write CSV ─────────────────────────────────────────────────────────
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject"] + keys)
        for subj, m in zip(subjects, all_metrics):
            writer.writerow([subj] + [round(m[k], 6) for k in keys])

        # Aggregate footer (comment lines — ignored by pd.read_csv with comment='#')
        writer.writerow([])
        for stat_name, fn in [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
            row = [f"# {stat_name}"]
            for k in keys:
                vals = np.array([m[k] for m in all_metrics])
                row.append(round(float(fn(vals)), 6))
            writer.writerow(row)

    print(f"\n  CSV salvo em: {output_path}")


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
