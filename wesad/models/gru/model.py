"""
Bidirectional GRU encoder for physiological signal classification.

Architecture:
    1. Multi-layer Bidirectional GRU — encodes the input sequence.
    2. Last hidden state extraction — concatenates forward and backward
       states of the final layer into a fixed-size context vector.
    3. Two fully-connected layers — map to class logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysioGRUEncoder(nn.Module):
    """
    Bidirectional GRU encoder for multiclass physiological state classification.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.
    hidden_dim : int
        Number of hidden units per direction in the GRU.
    output_dim : int
        Number of output classes.
    n_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout probability applied between GRU layers (only when n_layers > 1)
        and before the final linear layer.
    bidirectional : bool
        Whether to use a bidirectional GRU.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        context_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc1     = nn.Linear(context_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, T, input_dim)

        Returns
        -------
        Tensor, shape (B, output_dim)
        """
        # h_n shape: (n_layers * n_dirs, B, hidden_dim)
        _, h_n = self.gru(x)

        if self.bidirectional:
            # Concatenate the last forward (index -2) and backward (index -1) hidden states
            context = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            context = h_n[-1]

        out = self.dropout(F.relu(self.fc1(context)))
        return self.fc2(out)
