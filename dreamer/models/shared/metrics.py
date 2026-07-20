"""
Shared evaluation metrics for affect regression models.

All functions operate on numpy arrays of shape (n_samples, n_dimensions)
and return per-dimension results.
"""

import csv
from pathlib import Path

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
    with_std: bool = True,
) -> None:
    """
    Print mean, std, min and max across subjects for each dimension.

    Parameters
    ----------
    all_metrics : list of per-subject metric dicts.
    dim_names : list of dimension labels, e.g. ['Valência', 'Arousal', 'Dominância'].
    with_std : whether to print std/min/max alongside the mean (default True).
    """
    keys   = ["mae", "rmse", "r2", "pearson"]
    labels = {"mae": "MAE", "rmse": "RMSE", "r2": "R²", "pearson": "r"}

    stacked = {k: np.array([m[k] for m in all_metrics]) for k in keys}

    print("\n=== MÉDIAS FINAIS ===")
    for i, dim in enumerate(dim_names):
        print(f"\n{dim}:")
        for k in keys:
            vals = stacked[k][:, i]
            if with_std:
                print(
                    f"  {labels[k]:4s}: "
                    f"média={vals.mean():.4f} | "
                    f"std={vals.std():.4f} | "
                    f"min={vals.min():.4f} | "
                    f"max={vals.max():.4f} | "
                    f"faixa=[{vals.min():.4f} – {vals.max():.4f}]"
                )
            else:
                print(f"  {labels[k]:4s} = {vals.mean():.4f}")


def save_loso_csv(
    subjects: list,
    all_metrics: list[dict[str, NDArray]],
    dim_names: list[str],
    output_path: str | Path,
) -> None:
    """
    Persist per-subject LOSO metrics to a CSV file and print a summary table.

    One row per subject; columns are grouped by dimension:
        subject,
        MAE_<dim0>, RMSE_<dim0>, R2_<dim0>, r_<dim0>,
        MAE_<dim1>, …

    A footer block with mean/std/min/max is appended as comment lines
    (prefixed with ``#``) so the numeric rows remain machine-readable.

    Parameters
    ----------
    subjects : list
        Subject identifiers in the same order as ``all_metrics``.
    all_metrics : list[dict[str, NDArray]]
        Per-subject metric dicts (keys: mae, rmse, r2, pearson).
    dim_names : list[str]
        Short dimension labels used in column names, e.g. ['Val', 'Aro', 'Dom'].
    output_path : str or Path
        Destination CSV file.  Parent directories are created automatically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys   = ["mae", "rmse", "r2", "pearson"]
    metric_labels = {"mae": "MAE", "rmse": "RMSE", "r2": "R2", "pearson": "r"}

    # Build flat column names: MAE_Val, RMSE_Val, …, MAE_Aro, …
    col_names = []
    for dim in dim_names:
        for k in metric_keys:
            col_names.append(f"{metric_labels[k]}_{dim}")

    # ── Print per-subject table ────────────────────────────────────────────
    col_w = 9
    header_row = f"{'Sujeito':<10}" + "".join(f"{c:>{col_w}}" for c in col_names)
    print("\n" + "=" * len(header_row))
    print(header_row)
    print("-" * len(header_row))
    for subj, m in zip(subjects, all_metrics):
        vals = []
        for i in range(len(dim_names)):
            for k in metric_keys:
                vals.append(m[k][i])
        row = f"{str(subj):<10}" + "".join(f"{v:>{col_w}.4f}" for v in vals)
        print(row)
    print("=" * len(header_row))

    # ── Print aggregate summary ────────────────────────────────────────────
    print("\n── Resumo agregado ──")
    for i, dim in enumerate(dim_names):
        print(f"\n  {dim}:")
        for k in metric_keys:
            vals = np.array([m[k][i] for m in all_metrics])
            print(
                f"    {metric_labels[k]:4s}: "
                f"média={vals.mean():.4f} | "
                f"std={vals.std():.4f} | "
                f"min={vals.min():.4f} | "
                f"max={vals.max():.4f}"
            )

    # ── Write CSV ─────────────────────────────────────────────────────────
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject"] + col_names)

        for subj, m in zip(subjects, all_metrics):
            row_vals = [subj]
            for i in range(len(dim_names)):
                for k in metric_keys:
                    row_vals.append(round(float(m[k][i]), 6))
            writer.writerow(row_vals)

        # Aggregate footer
        writer.writerow([])
        for stat_name, fn in [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
            row_vals = [f"# {stat_name}"]
            for i in range(len(dim_names)):
                for k in metric_keys:
                    vals = np.array([m[k][i] for m in all_metrics])
                    row_vals.append(round(float(fn(vals)), 6))
            writer.writerow(row_vals)

    print(f"\n  CSV salvo em: {output_path}")
