"""
CNN-BiLSTM model for continuous affect prediction from EEG band power sequences.

Architecture:
    1. 1-D Temporal CNN — extracts local temporal patterns across windows.
    2. Bidirectional LSTM — captures long-range sequential dependencies.
    3. Mean pooling over time — aggregates the sequence into a fixed-size vector.
    4. Two fully-connected layers — map to the three affect dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTM(nn.Module):
    """
    Convolutional + Bidirectional LSTM network for affect regression.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step (n_channels * n_bands).
    hidden_cnn : int
        Number of output filters in the Conv1d layer.
    hidden_lstm : int
        Number of hidden units per direction in the BiLSTM.
    output_dim : int
        Number of regression targets (default: 3 — valence, arousal, dominance).
    dropout : float
        Dropout probability applied before the final linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_cnn: int = 64,
        hidden_lstm: int = 128,
        output_dim: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Temporal CNN block
        self.conv1   = nn.Conv1d(input_dim, hidden_cnn, kernel_size=3, padding=1)
        self.bn1     = nn.BatchNorm1d(hidden_cnn)
        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_cnn,
            hidden_size=hidden_lstm,
            batch_first=True,
            bidirectional=True,
        )

        # Regression head
        self.fc1 = nn.Linear(hidden_lstm * 2, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, D, T)
            Input batch — features D along the channel dimension, T time steps.
        padding_mask : BoolTensor, shape (B, T)
            True at positions that correspond to padding and should be ignored.

        Returns
        -------
        Tensor, shape (B, output_dim)
        """
        x = F.relu(self.bn1(self.conv1(x)))   # (B, hidden_cnn, T)
        x = x.transpose(1, 2)                  # (B, T, hidden_cnn)

        # Zero out padded positions before feeding into the LSTM
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        out, _ = self.lstm(x)                  # (B, T, hidden_lstm * 2)
        out = out.mean(dim=1)                  # mean pooling over time → (B, hidden_lstm * 2)

        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)
