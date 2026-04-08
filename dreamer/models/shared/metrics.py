"""
Shared evaluation metrics for affect regression models.

All functions operate on numpy arrays of shape (n_samples, n_dimensions)
and return per-dimension results.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


def compute_metrics(
    trues: NDArray,
    preds: NDArray,
) -> dict[str, NDArray]:
    """
    Compute MAE, RMSE, R² and Pearson r for each output dimension.

    Parameters
    ----------
    trues : NDArray, shape (n_samples, n_dims)
        Ground-truth labels.
    preds : NDArray, shape (n_samples, n_dims)
        Model predictions.

    Returns
    -------
    dict with keys 'mae', 'rmse', 'r2', 'pearson', each a 1-D array
    of length n_dims.
    """
    n_dims = trues.shape[1]
    mae  = np.mean(np.abs(preds - trues), axis=0)
    rmse = np.sqrt(mean_squared_error(trues, preds, multioutput="raw_values"))
    r2   = np.array([r2_score(trues[:, i], preds[:, i]) for i in range(n_dims)])
    pear = np.array([pearsonr(trues[:, i], preds[:, i])[0] for i in range(n_dims)])

    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pear}


def print_subject_metrics(subject_id: int, metrics: dict[str, NDArray]) -> None:
    """Print a one-line per-subject metrics summary."""
    print(
        f"Subject {subject_id} → "
        f"MAE: {metrics['mae'].round(3)}, "
        f"RMSE: {metrics['rmse'].round(3)}, "
        f"R²: {metrics['r2'].round(3)}, "
        f"r: {metrics['pearson'].round(3)}"
    )


def print_aggregate_metrics(
    all_metrics: list[dict[str, NDArray]],
    dim_names: list[str],
    with_std: bool = False,
) -> None:
    """
    Print mean (and optionally std) across subjects for each dimension.

    Parameters
    ----------
    all_metrics : list of per-subject metric dicts.
    dim_names : list of dimension labels, e.g. ['Valência', 'Arousal', 'Dominância'].
    with_std : whether to print ± std alongside the mean.
    """
    keys = ["mae", "rmse", "r2", "pearson"]
    labels = {"mae": "MAE", "rmse": "RMSE", "r2": "R²", "pearson": "r"}

    stacked = {k: np.array([m[k] for m in all_metrics]) for k in keys}

    print("\n=== MÉDIAS FINAIS ===")
    for i, dim in enumerate(dim_names):
        print(f"{dim}:")
        for k in keys:
            mean_val = stacked[k][:, i].mean()
            if with_std:
                std_val = stacked[k][:, i].std()
                print(f"  {labels[k]:4s} = {mean_val:.3f} ± {std_val:.3f}")
            else:
                print(f"  {labels[k]:4s} = {mean_val:.3f}")
