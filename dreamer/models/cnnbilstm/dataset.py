"""
Dataset utilities for the CNN-BiLSTM model.

Handles sequence construction from the window-level band power CSV,
the PyTorch Dataset class, and the variable-length collate function.
"""

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

FEATURE_COLS: list[str] = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "beta/alpha",
    "(beta+gamma)/(alpha+theta)",
]

TARGET_COLS: list[str] = ["valence", "arousal", "dominance"]


def build_sequences(
    df: pd.DataFrame,
) -> tuple[list[NDArray], list[NDArray]]:
    """
    Convert the window-level band power DataFrame into per-video sequences.

    Each (subject, video) group is reshaped into a 2-D array of shape
    (T, C * F), where T is the number of windows, C the number of EEG
    channels, and F the number of frequency-band features.

    Parameters
    ----------
    df : pd.DataFrame
        Window-level band power DataFrame produced by the DREAMER pipeline.
        Must contain columns: subject_id, video_id, window_id, channel,
        the band feature columns, and the target columns.

    Returns
    -------
    X_list : list of NDArray, each shape (T, C * F)
    y_list : list of NDArray, each shape (3,) — [valence, arousal, dominance]
    """
    X_list: list[NDArray] = []
    y_list: list[NDArray] = []

    for _, group in df.groupby(["subject_id", "video_id"]):
        group = group.sort_values(["window_id", "channel"])

        n_channels = group["channel"].nunique()
        n_bands    = len(FEATURE_COLS)
        n_windows  = group["window_id"].nunique()

        # Pivot to (T, C * F) via (T, C, F) reshape
        X = (
            group[FEATURE_COLS + ["channel", "window_id"]]
            .pivot(index="window_id", columns="channel", values=FEATURE_COLS)
            .values
        )
        X = X.reshape(n_windows, n_channels, n_bands).reshape(n_windows, n_channels * n_bands)

        y = group[TARGET_COLS].iloc[0].values.astype("float32")

        X_list.append(X.astype("float32"))
        y_list.append(y)

    return X_list, y_list

class DreamerSeqDataset(Dataset):
    """
    PyTorch Dataset wrapping per-video EEG sequences and affect labels.

    Parameters
    ----------
    X_list : list of NDArray, each shape (T, D)
    y_list : list of NDArray, each shape (3,)
    """

    def __init__(self, X_list: list[NDArray], y_list: list[NDArray]) -> None:
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.y = [torch.tensor(y, dtype=torch.float32) for y in y_list]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        seq = self.X[idx]
        return seq, self.y[idx], seq.shape[0]


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate variable-length sequences into a padded batch.

    Sequences are padded to the maximum length in the batch.
    A boolean padding mask is created so the model can ignore padded positions.

    Parameters
    ----------
    batch : list of (sequence, target, seq_len) tuples.

    Returns
    -------
    padded_sequences : Tensor, shape (B, 1, T_max, D)
    targets          : Tensor, shape (B, 3)
    seq_lengths      : Tensor, shape (B,)
    padding_mask     : BoolTensor, shape (B, T_max) — True where padded
    """
    sequences, targets, seq_lengths = zip(*batch)

    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)  # (B, T_max, D)

    padding_mask = torch.zeros(len(seq_lengths), padded.shape[1], dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        padding_mask[i, length:] = True

    return (
        padded.unsqueeze(1),           # (B, 1, T_max, D)
        torch.stack(targets),
        torch.tensor(seq_lengths),
        padding_mask,
    )
