"""
Shared classification plots for LOSO evaluation results.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from shared.classification.metrics import get_classification_report_df


def plot_confusion_matrix(
    trues: NDArray,
    preds: NDArray,
    title: str = "Matriz de Confusão Global",
) -> None:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    trues : array-like of int
    preds : array-like of int
    title : str
    """
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_per_class_metrics(
    trues: NDArray,
    preds: NDArray,
    title: str = "Métricas por Classe",
) -> None:
    """
    Plot a heatmap of precision, recall, F1 and global accuracy per class.

    Parameters
    ----------
    trues : array-like of int
    preds : array-like of int
    title : str
    """
    df_classes = get_classification_report_df(trues, preds)

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        df_classes[["precision", "recall", "f1-score", "accuracy"]],
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
    )
    plt.title(title)
    plt.ylabel("Classe")
    plt.xlabel("Métrica")
    plt.tight_layout()
    plt.show()
