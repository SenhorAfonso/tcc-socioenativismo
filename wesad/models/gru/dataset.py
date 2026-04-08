"""
PyTorch Dataset for the WESAD GRU model.

Converts a flat feature matrix into overlapping sequences using a
sliding window, assigning the label of the last timestep to each window.
"""

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class PhysioDataset(Dataset):
    """
    Sliding-window sequence dataset over physiological signal features.

    Given a flat array of shape (N, F), produces sequences of shape
    (seq_len, F) with stride ``stride``. The target label for each
    sequence is the label at the last timestep of the window.

    Parameters
    ----------
    X : NDArray, shape (N, F)
        Feature matrix, already scaled.
    y : NDArray, shape (N,)
        Integer class labels (0-indexed).
    seq_len : int
        Number of timesteps per sequence.
    stride : int
        Step size between consecutive windows.
    """

    def __init__(
        self,
        X: NDArray,
        y: NDArray,
        seq_len: int = 10,
        stride: int = 5,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride
        self.num_samples = (len(self.X) - self.seq_len) // self.stride + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x_seq = self.X[start : start + self.seq_len]
        y_label = self.y[start + self.seq_len - 1]
        return x_seq, y_label
