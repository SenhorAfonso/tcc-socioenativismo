"""
LightGBM training and evaluation on the DREAMER dataset.

Validation strategy: Leave-One-Subject-Out (LOSO).
Input: per-video brain variation statistics (aggregated across channels).

Usage:
    python dreamer/models/lightgbm/main.py

Expected input file (produced by the DREAMER preprocessing pipeline):
    dreamer/output/subject_bandpower_by_video/brain_variation_per_video.csv
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# Allow imports from the dreamer package root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.metrics import (
    compute_metrics,
    print_aggregate_metrics,
    print_subject_metrics,
    save_loso_csv,
)
from models.shared.plots import (
    build_results_df,
    plot_pearson_boxplot,
    plot_pearson_heatmap_per_subject,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR  = Path(__file__).resolve().parents[3]
INPUT_CSV = (
    ROOT_DIR / "dreamer" / "output"
    / "subject_bandpower_by_video"
    / "brain_variation_per_video.csv"
)

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
LGBM_PARAMS: dict = {
    "n_estimators":  400,
    "learning_rate": 0.05,
    "num_leaves":    31,
    "max_depth":     -1,
    "n_jobs":        -1,
    "random_state":  42,
    "verbose":       -1,
}

TARGET_COLS = ["valence", "arousal", "dominance"]
DIM_NAMES   = ["Valência", "Arousal", "Dominância"]


def run_loso(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
) -> tuple[list, list]:
    """
    Run Leave-One-Subject-Out cross-validation with a MultiOutput LightGBM.

    Returns
    -------
    subjects_order : list of subject IDs in evaluation order.
    all_metrics    : list of per-subject metric dicts.
    """
    logo = LeaveOneGroupOut()
    subjects_order, all_metrics = [], []

    for train_idx, test_idx in logo.split(X_all, Y_all, groups):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = Y_all[train_idx], Y_all[test_idx]

        # Fit scaler on training data only to prevent data leakage
        scaler  = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)

        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        X_test_df  = pd.DataFrame(X_test,  columns=feature_cols)

        model = MultiOutputRegressor(lgb.LGBMRegressor(**LGBM_PARAMS))
        model.fit(X_train_df, y_train)
        preds = model.predict(X_test_df)

        metrics = compute_metrics(y_test, preds)
        subj    = groups[test_idx][0]

        print_subject_metrics(subj, metrics)
        subjects_order.append(subj)
        all_metrics.append(metrics)

    return subjects_order, all_metrics


def main() -> None:
    print(f"Loading data from {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    drop_cols   = ["subject_id", "video_id"] + TARGET_COLS
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_all  = df[feature_cols].values
    Y_all  = df[TARGET_COLS].values
    groups = df["subject_id"].values

    subjects_order, all_metrics = run_loso(X_all, Y_all, groups, feature_cols)

    print_aggregate_metrics(all_metrics, DIM_NAMES, with_std=True)

    save_loso_csv(
        subjects_order,
        all_metrics,
        ["Val", "Aro", "Dom"],
        ROOT_DIR / "dreamer" / "output" / "loso_lightgbm.csv",
    )

    results_df = build_results_df(np.array(subjects_order), all_metrics)
    plot_pearson_heatmap_per_subject(
        results_df,
        title="Correlação de Pearson (r) por Sujeito e Dimensão – LightGBM",
    )
    plot_pearson_boxplot(results_df)


if __name__ == "__main__":
    main()
