"""
GRU training and evaluation on the WESAD dataset.

Validation strategy: Leave-One-Subject-Out (LOSO).

Usage:
    python wesad/models/gru/main.py

Expected input file (produced by the WESAD preprocessing pipeline):
    wesad/output/final_wesad.csv
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from shared.classification.metrics import (
    compute_metrics,
    print_aggregate_metrics,
    print_subject_metrics,
    save_loso_csv,
)
from shared.classification.plots import plot_confusion_matrix, plot_per_class_metrics
from wesad.models.gru.dataset import PhysioDataset
from wesad.models.gru.model import PhysioGRUEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_CSV = ROOT_DIR / "wesad" / "output" / "final_wesad.csv"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS     = 25
BATCH_SIZE = 128
SEQ_LEN    = 10
STRIDE     = 5
LR         = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2
SEED         = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: PhysioGRUEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: PhysioGRUEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run inference and return (preds, trues) as flat lists."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            preds.extend(torch.argmax(model(xb), dim=1).cpu().numpy())
            trues.extend(yb.numpy())
    return preds, trues


def run_loso(
    data: pd.DataFrame,
    features: list[str],
    subjects: np.ndarray,
    num_features: int,
    num_classes: int,
    device: torch.device,
) -> tuple[list, list[dict], list[int], list[int]]:
    """
    Run Leave-One-Subject-Out cross-validation.

    Returns
    -------
    subjects_order : list of subject IDs in evaluation order.
    all_metrics    : list of per-subject metric dicts.
    all_trues      : accumulated ground-truth labels across all folds.
    all_preds      : accumulated predictions across all folds.
    """
    subjects_order: list = []
    all_metrics: list[dict] = []
    all_trues:   list[int]  = []
    all_preds:   list[int]  = []

    for test_subject in tqdm(subjects, desc="LOSO"):
        train_df = data[data["subject"] != test_subject]
        test_df  = data[data["subject"] == test_subject]

        # Labels are 1-indexed in the CSV; shift to 0-indexed for CrossEntropyLoss
        X_train = train_df[features].values
        y_train = train_df["label"].values - 1
        X_test  = test_df[features].values
        y_test  = test_df["label"].values - 1

        # Fit scaler on training data only to prevent data leakage
        scaler  = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)

        train_dl = DataLoader(
            PhysioDataset(X_train, y_train, SEQ_LEN, STRIDE),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
        )
        test_dl = DataLoader(
            PhysioDataset(X_test, y_test, SEQ_LEN, STRIDE),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
        )

        model     = PhysioGRUEncoder(input_dim=num_features, output_dim=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_dl, optimizer, criterion, device)
            print(f"  [S{test_subject}] Epoch {epoch + 1}/{EPOCHS} | Loss {loss:.4f}")

        preds, trues = evaluate(model, test_dl, device)
        metrics = compute_metrics(trues, preds)

        print_subject_metrics(test_subject, metrics)
        subjects_order.append(test_subject)
        all_metrics.append(metrics)
        all_trues.extend(trues)
        all_preds.extend(preds)

    return subjects_order, all_metrics, all_trues, all_preds


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading data from {INPUT_CSV} ...")
    data = pd.read_csv(INPUT_CSV)

    features    = [c for c in data.columns if c not in ("subject", "label")]
    subjects    = data["subject"].unique()
    num_features = len(features)
    num_classes  = data["label"].nunique()

    subjects_order, all_metrics, all_trues, all_preds = run_loso(
        data, features, subjects, num_features, num_classes, device
    )

    print_aggregate_metrics(all_metrics)
    save_loso_csv(subjects_order, all_metrics, ROOT_DIR / "wesad" / "output" / "loso_gru.csv")
    plot_confusion_matrix(all_trues, all_preds, title="Matriz de Confusão Global – GRU (WESAD)")
    plot_per_class_metrics(all_trues, all_preds, title="Métricas por Classe – GRU (WESAD)")


if __name__ == "__main__":
    main()
