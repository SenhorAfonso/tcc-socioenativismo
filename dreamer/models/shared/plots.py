"""
Shared plotting utilities for affect regression results.

All functions accept a results DataFrame with columns:
    subject, MAE_Val, MAE_Aro, MAE_Dom,
    RMSE_Val, RMSE_Aro, RMSE_Dom,
    R2_Val, R2_Aro, R2_Dom,
    r_Val, r_Aro, r_Dom
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

DIM_NAMES = ["Valência", "Arousal", "Dominância"]


def build_results_df(
    subjects: NDArray,
    all_metrics: list[dict[str, NDArray]],
) -> pd.DataFrame:
    """
    Assemble a per-subject results DataFrame from a list of metric dicts.

    Parameters
    ----------
    subjects : NDArray
        Ordered array of subject identifiers.
    all_metrics : list[dict]
        One dict per subject, as returned by ``compute_metrics``.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for subj, m in zip(subjects, all_metrics):
        rows.append({
            "subject":   subj,
            "MAE_Val":   m["mae"][0],   "MAE_Aro":   m["mae"][1],   "MAE_Dom":   m["mae"][2],
            "RMSE_Val":  m["rmse"][0],  "RMSE_Aro":  m["rmse"][1],  "RMSE_Dom":  m["rmse"][2],
            "R2_Val":    m["r2"][0],    "R2_Aro":    m["r2"][1],    "R2_Dom":    m["r2"][2],
            "r_Val":     m["pearson"][0], "r_Aro":   m["pearson"][1], "r_Dom":   m["pearson"][2],
        })
    return pd.DataFrame(rows)


def plot_pearson_heatmap_per_subject(
    results_df: pd.DataFrame,
    title: str = "Correlação de Pearson (r) por Sujeito e Dimensão",
) -> None:
    """
    Heatmap of Pearson r per subject and affect dimension.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame as returned by ``build_results_df``.
    title : str
        Plot title.
    """
    r_vals = results_df[["r_Val", "r_Aro", "r_Dom"]].copy()
    r_vals.index = results_df["subject"]
    r_vals.columns = DIM_NAMES

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(8, 6))
    sns.heatmap(r_vals, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Dimensão")
    plt.ylabel("Sujeito")
    plt.tight_layout()
    plt.show()


def plot_mean_metrics_heatmap(
    all_metrics: list[dict[str, NDArray]],
    title: str = "Desempenho médio por dimensão (LOSO)",
) -> None:
    """
    Heatmap of mean MAE, RMSE, R² and r across subjects per dimension.

    Parameters
    ----------
    all_metrics : list[dict]
        One dict per subject, as returned by ``compute_metrics``.
    title : str
        Plot title.
    """
    keys   = ["mae", "rmse", "r2", "pearson"]
    labels = {"mae": "MAE", "rmse": "RMSE", "r2": "R²", "pearson": "r"}

    data = {labels[k]: np.mean([m[k] for m in all_metrics], axis=0) for k in keys}
    df_metrics = pd.DataFrame(data, index=DIM_NAMES)

    plt.figure(figsize=(7, 4))
    sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Valor"})
    plt.title(title)
    plt.xlabel("Métrica")
    plt.ylabel("Dimensão")
    plt.tight_layout()
    plt.show()


def plot_pearson_boxplot(
    results_df: pd.DataFrame,
    title: str = "Distribuição da Correlação (r) entre Sujeitos",
) -> None:
    """
    Boxplot + swarmplot of Pearson r distribution across subjects per dimension.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame as returned by ``build_results_df``.
    title : str
        Plot title.
    """
    r_df = results_df.melt(
        id_vars="subject",
        value_vars=["r_Val", "r_Aro", "r_Dom"],
        var_name="Dimensão",
        value_name="r",
    )
    r_df["Dimensão"] = r_df["Dimensão"].str.replace("r_", "", regex=False)

    plt.figure(figsize=(6, 5))
    sns.boxplot(data=r_df, x="Dimensão", y="r", palette="Set3")
    sns.swarmplot(data=r_df, x="Dimensão", y="r", color="k", alpha=0.6)
    plt.title(title)
    plt.ylabel("Correlação de Pearson (r)")
    plt.xlabel("Dimensão")
    plt.tight_layout()
    plt.show()
