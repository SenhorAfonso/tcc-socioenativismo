"""
CNN-BiLSTM training and evaluation on the DREAMER dataset.

Validation strategy: Leave-One-Subject-Out (LOSO).

Usage:
    python dreamer/models/cnnbilstm/main.py

Expected input file (produced by the DREAMER preprocessing pipeline):
    dreamer/output/dreamer_bandpower_all.csv
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Allow imports from the dreamer package root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.cnnbilstm.dataset import DreamerSeqDataset, build_sequences, collate_fn
from models.cnnbilstm.model import CNNBiLSTM
from models.shared.metrics import compute_metrics, print_aggregate_metrics, print_subject_metrics
from models.shared.plots import (
    build_results_df,
    plot_mean_metrics_heatmap,
    plot_pearson_heatmap_per_subject,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR  = Path(__file__).resolve().parents[3]
INPUT_CSV = ROOT_DIR / "dreamer" / "output" / "dreamer_bandpower_all.csv"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED        = 42
BATCH_SIZE  = 8
EPOCHS      = 15
LR          = 1e-3
WEIGHT_DECAY = 1e-4

DIM_NAMES = ["Valência", "Arousal", "Dominância"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: CNNBiLSTM,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> None:
    model.train()
    for xb, yb, _, mask in loader:
        xb = xb.squeeze(1).transpose(1, 2).to(device)  # (B, D, T)
        yb, mask = yb.to(device), mask.to(device)
        optimizer.zero_grad()
        loss_fn(model(xb, mask), yb).backward()
        optimizer.step()


def evaluate(
    model: CNNBiLSTM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb, _, mask in loader:
            xb  = xb.squeeze(1).transpose(1, 2).to(device)
            mask = mask.to(device)
            preds.append(model(xb, mask).cpu().numpy())
            trues.append(yb.numpy())
    return np.vstack(preds), np.vstack(trues)


def run_loso(
    df: pd.DataFrame,
    X_list: list,
    y_list: list,
    device: torch.device,
) -> tuple[list, list]:
    """
    Run Leave-One-Subject-Out cross-validation.

    Returns
    -------
    subjects_order : list of subject IDs in evaluation order.
    all_metrics    : list of per-subject metric dicts.
    """
    # Build a mapping from sequence index → subject_id
    seq_subjects = [
        s for (s, _) in df.groupby(["subject_id", "video_id"]).groups.keys()
    ]
    unique_subjects = df["subject_id"].unique()

    subjects_order, all_metrics = [], []

    for test_subj in unique_subjects:
        train_idx = [i for i, s in enumerate(seq_subjects) if s != test_subj]
        test_idx  = [i for i, s in enumerate(seq_subjects) if s == test_subj]

        train_ds = DreamerSeqDataset([X_list[i] for i in train_idx], [y_list[i] for i in train_idx])
        test_ds  = DreamerSeqDataset([X_list[i] for i in test_idx],  [y_list[i] for i in test_idx])

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
        test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model    = CNNBiLSTM(input_dim=X_list[train_idx[0]].shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn  = nn.L1Loss()

        for _ in range(EPOCHS):
            train_epoch(model, train_dl, optimizer, loss_fn, device)

        preds, trues = evaluate(model, test_dl, device)
        metrics = compute_metrics(trues, preds)

        print_subject_metrics(test_subj, metrics)
        subjects_order.append(test_subj)
        all_metrics.append(metrics)

    return subjects_order, all_metrics


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading data from {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    print("Building sequences ...")
    X_list, y_list = build_sequences(df)

    subjects_order, all_metrics = run_loso(df, X_list, y_list, device)

    print_aggregate_metrics(all_metrics, DIM_NAMES, with_std=False)

    results_df = build_results_df(np.array(subjects_order), all_metrics)
    plot_pearson_heatmap_per_subject(
        results_df,
        title="Correlação de Pearson (r) por Sujeito e Dimensão – CNN-BiLSTM",
    )
    plot_mean_metrics_heatmap(all_metrics, title="Desempenho médio por dimensão (LOSO) – CNN-BiLSTM")


if __name__ == "__main__":
    main()
