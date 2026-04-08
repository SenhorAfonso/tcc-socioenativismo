"""
XGBoost training and evaluation on the WESAD dataset.

Validation strategy: Leave-One-Subject-Out (LOSO).

Usage:
    python wesad/models/xgboost/main.py

Expected input file (produced by the WESAD preprocessing pipeline):
    wesad/output/final_wesad.csv

Note: XGBoost requires 0-indexed labels, so the original 1–4 label range
is shifted by -1 before training.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from shared.classification.metrics import (
    compute_metrics,
    print_aggregate_metrics,
    print_subject_metrics,
)
from shared.classification.plots import plot_confusion_matrix, plot_per_class_metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_CSV = ROOT_DIR / "wesad" / "output" / "final_wesad.csv"

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
XGB_PARAMS: dict = {
    "n_estimators":  300,
    "learning_rate": 0.05,
    "max_depth":     6,
    "n_jobs":        -1,
    "random_state":  42,
}


def run_loso(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[list[dict], list, list]:
    """
    Run Leave-One-Subject-Out cross-validation with XGBoost.

    Returns
    -------
    all_metrics : list of per-subject metric dicts.
    all_trues   : accumulated ground-truth labels (0-indexed).
    all_preds   : accumulated predictions (0-indexed).
    """
    logo = LeaveOneGroupOut()
    all_metrics, all_trues, all_preds = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        # Shift labels to 0-indexed as required by XGBoost
        y_train = y.iloc[train_idx] - 1
        y_test  = y.iloc[test_idx]  - 1

        # Fit scaler on training data only to prevent data leakage
        scaler     = StandardScaler().fit(X_train)
        X_train_sc = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
        X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=X.columns)

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)

        metrics = compute_metrics(y_test.values, preds)
        subj    = groups.iloc[test_idx[0]]

        print_subject_metrics(subj, metrics)
        all_metrics.append(metrics)
        all_trues.extend(y_test.values)
        all_preds.extend(preds)

    return all_metrics, all_trues, all_preds


def main() -> None:
    print(f"Loading data from {INPUT_CSV} ...")
    data = pd.read_csv(INPUT_CSV)

    feature_cols = [c for c in data.columns if c not in ("subject", "label")]
    X      = data[feature_cols]
    y      = data["label"]
    groups = data["subject"]

    all_metrics, all_trues, all_preds = run_loso(X, y, groups)

    print_aggregate_metrics(all_metrics)
    plot_confusion_matrix(all_trues, all_preds, title="Matriz de Confusão Global – XGBoost (WESAD)")
    plot_per_class_metrics(all_trues, all_preds, title="Métricas por Classe – XGBoost (WESAD)")


if __name__ == "__main__":
    main()
