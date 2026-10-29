# -*- coding: utf-8 -*-
"""
Fold-wise encoder retraining version of transfer_learning.py.
Trains PhysioGRUEncoder (15 folds, 15 epochs/fold) and CNNBiLSTM (23 folds, 15 epochs/fold)
strictly out-of-fold before running the 23-fold HybridFusion LOSO evaluation.
Outputs are written to a separate directory so original results are untouched.
"""
import time
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

TCC_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EPOCHS_BASE = 15
BATCH_SIZE = 128
SEQ_LEN = 10
STRIDE = 5
LR = 1e-3
NUM_WORKERS = 0


class PhysioGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=4, n_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0, bidirectional=bidirectional,
        )
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(self.hidden_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        context = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) if self.bidirectional else h_n[-1, :, :]
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        return self.fc2(x)

    def forward_features(self, x):
        _, h_n = self.gru(x)
        return torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) if self.bidirectional else h_n[-1, :, :]


class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_cnn=64, hidden_lstm=128, output_dim=3, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_cnn, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_cnn)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=hidden_cnn, hidden_size=hidden_lstm, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_lstm * 2, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x, padding_mask):
        x = self.relu(self.bn1(self.conv1(x))).transpose(1, 2)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(out))))

    def forward_features(self, x, padding_mask):
        x = self.relu(self.bn1(self.conv1(x))).transpose(1, 2)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.)
        out, _ = self.lstm(x)
        return out.mean(dim=1)


class PhysioDataset(Dataset):
    def __init__(self, X, y, seq_len=10, stride=5):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride
        self.num_samples = (len(self.X) - self.seq_len) // self.stride + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        return self.X[start_idx : start_idx + self.seq_len], self.y[start_idx + self.seq_len - 1]


class DreamerSeqDataset(Dataset):
    def __init__(self, X_list, y_list):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.y = [torch.tensor(y, dtype=torch.float32) for y in y_list]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        seq = self.X[i]
        return seq.unsqueeze(0), self.y[i], seq.shape[0]


def collate_fn(batch):
    sequences, targets, seq_lengths = zip(*batch)
    padded_sequences = pad_sequence([seq.squeeze(0) for seq in sequences], batch_first=True, padding_value=0.0)
    padding_mask = torch.zeros_like(padded_sequences, dtype=torch.bool)[:, :, 0]
    for i, seq_len in enumerate(seq_lengths):
        padding_mask[i, seq_len:] = True
    return padded_sequences.unsqueeze(1), torch.stack(targets), torch.tensor(seq_lengths), padding_mask


def build_sequences(df, feature_cols):
    X_list, y_list = [], []
    for (subj, vid), g in df.groupby(["subject_id", "video_id"]):
        g = g.sort_values(["window_id", "channel"])
        channels = g["channel"].nunique()
        bands = len(feature_cols)
        T = g["window_id"].nunique()
        X = g[feature_cols + ["channel", "window_id"]].pivot(index="window_id", columns="channel", values=feature_cols).values
        X = X.reshape(T, channels, bands).reshape(T, channels * bands)
        y = g[["valence", "arousal", "dominance"]].iloc[0].values.astype("float32")
        X_list.append(X.astype("float32"))
        y_list.append(y)
    return X_list, y_list


class HybridEmbeddingsDataset(Dataset):
    def __init__(self, z_eeg, z_ecg, y):
        self.z_eeg = z_eeg.clone().detach() if isinstance(z_eeg, torch.Tensor) else torch.tensor(z_eeg, dtype=torch.float32)
        self.z_ecg = z_ecg.clone().detach() if isinstance(z_ecg, torch.Tensor) else torch.tensor(z_ecg, dtype=torch.float32)
        self.y = y.clone().detach() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.z_eeg[idx], torch.zeros(self.z_eeg[idx].shape[0], dtype=torch.bool), self.z_ecg[idx], self.y[idx]


class HybridFusion(nn.Module):
    def __init__(self, input_dim_eeg, input_dim_ecg, hidden_dim=256, output_dim=3, dropout=0.5, lambda_align=0.01):
        super().__init__()
        self.proj_eeg = nn.Linear(input_dim_eeg, hidden_dim)
        self.proj_ecg = nn.Linear(input_dim_ecg, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.lambda_align = lambda_align

    def forward(self, z_eeg, z_ecg, return_loss=False):
        h_eeg = F.relu(self.proj_eeg(z_eeg))
        h_ecg = F.relu(self.proj_ecg(z_ecg))
        out = self.fusion(torch.cat([h_eeg, h_ecg], dim=1))
        if return_loss:
            return out, F.mse_loss(h_eeg, h_ecg)
        return out


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_wesad = pd.read_csv(TCC_DIR / "wesad" / "output" / "final_wesad.csv")
    data_dreamer = pd.read_csv(TCC_DIR / "dreamer" / "output" / "dreamer_bandpower_all.csv")

    # 1. Fold-wise WESAD GRU training & out-of-fold embedding extraction (15 folds)
    features_wesad = [col for col in data_wesad.columns if col not in ["subject", "label"]]
    subjects_wesad = data_wesad["subject"].unique()
    num_features_wesad = len(features_wesad)
    num_classes_wesad = data_wesad["label"].nunique()

    embeddings_ecg = []
    t_gru_start = time.time()
    print(f"\n[1/3] Training WESAD GRU encoder per fold ({len(subjects_wesad)} folds, {EPOCHS_BASE} epochs/fold)...")
    for idx_s, test_subject in enumerate(subjects_wesad):
        train_df = data_wesad[data_wesad.subject != test_subject]
        test_df = data_wesad[data_wesad.subject == test_subject]

        scaler = StandardScaler().fit(train_df[features_wesad].values)
        X_train = scaler.transform(train_df[features_wesad].values)
        y_train = train_df["label"].values - 1
        X_test = scaler.transform(test_df[features_wesad].values)
        y_test = test_df["label"].values - 1

        train_ds = PhysioDataset(X_train, y_train, seq_len=SEQ_LEN, stride=STRIDE)
        test_ds = PhysioDataset(X_test, y_test, seq_len=SEQ_LEN, stride=STRIDE)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model_ecg = PhysioGRUEncoder(input_dim=num_features_wesad, hidden_dim=128, output_dim=num_classes_wesad, bidirectional=True).to(device)
        optimizer = optim.Adam(model_ecg.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model_ecg.train()
        for _ in range(EPOCHS_BASE):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model_ecg(xb), yb)
                loss.backward()
                optimizer.step()

        model_ecg.eval()
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                embeddings_ecg.append(model_ecg.forward_features(xb).cpu())
        print(f"  WESAD GRU fold {idx_s+1:02d}/{len(subjects_wesad)} (Subject {test_subject}) done ({time.time() - t_gru_start:.1f}s elapsed)")
    t_gru = time.time() - t_gru_start

    # 2. Fold-wise DREAMER CNN-BiLSTM training & out-of-fold embedding extraction (23 folds)
    feature_cols_dreamer = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'beta/alpha', '(beta+gamma)/(alpha+theta)']
    subjects_dreamer = data_dreamer["subject_id"].unique()
    X_list_dreamer, y_list_dreamer = build_sequences(data_dreamer, feature_cols_dreamer)
    seq_subjects = [s for (s, _) in data_dreamer.groupby(["subject_id", "video_id"]).groups.keys()]

    embeddings_eeg = []
    labels_eeg = []
    subject_ids_eeg = []

    t_eeg_start = time.time()
    print(f"\n[2/3] Training DREAMER CNN-BiLSTM encoder per fold ({len(subjects_dreamer)} folds, {EPOCHS_BASE} epochs/fold)...")
    for idx_s, test_subj in enumerate(subjects_dreamer):
        train_idx = [i for i, s in enumerate(seq_subjects) if s != test_subj]
        test_idx = [i for i, s in enumerate(seq_subjects) if s == test_subj]

        train_ds = DreamerSeqDataset([X_list_dreamer[i] for i in train_idx], [y_list_dreamer[i] for i in train_idx])
        test_ds = DreamerSeqDataset([X_list_dreamer[i] for i in test_idx], [y_list_dreamer[i] for i in test_idx])
        train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
        test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

        model_eeg = CNNBiLSTM(input_dim=X_list_dreamer[0].shape[1]).to(device)
        optimizer = optim.Adam(model_eeg.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.L1Loss()

        model_eeg.train()
        for _ in range(EPOCHS_BASE):
            for xb, yb, _, padding_mask in train_dl:
                xb = xb.squeeze(1).transpose(1, 2).to(device)
                yb, padding_mask = yb.to(device), padding_mask.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model_eeg(xb, padding_mask), yb)
                loss.backward()
                optimizer.step()

        model_eeg.eval()
        with torch.no_grad():
            for xb, yb, _, padding_mask in test_dl:
                xb = xb.squeeze(1).transpose(1, 2).to(device)
                padding_mask = padding_mask.to(device)
                z_eeg = model_eeg.forward_features(xb, padding_mask)
                embeddings_eeg.append(z_eeg.cpu())
                labels_eeg.append(yb)
                subject_ids_eeg.extend([test_subj] * len(yb))
        print(f"  DREAMER CNN-BiLSTM fold {idx_s+1:02d}/{len(subjects_dreamer)} (Subject {test_subj}) done ({time.time() - t_eeg_start:.1f}s elapsed)")
    t_eeg = time.time() - t_eeg_start

    # 3. Pair embeddings and run 23-fold HybridFusion LOSO
    z_eeg_tensor = torch.cat(embeddings_eeg, dim=0)
    z_ecg_tensor = torch.cat(embeddings_ecg, dim=0)
    y_tensor = torch.cat(labels_eeg, dim=0)
    subject_ids_eeg = np.array(subject_ids_eeg)

    min_len = min(len(z_eeg_tensor), len(z_ecg_tensor))
    np.random.seed(SEED)
    idx_ecg = np.random.choice(len(z_ecg_tensor), size=min_len, replace=False)
    idx_eeg = np.random.choice(len(z_eeg_tensor), size=min_len, replace=False)

    z_ecg_sync = z_ecg_tensor[idx_ecg]
    z_eeg_sync = z_eeg_tensor[idx_eeg]
    y_sync = y_tensor[idx_eeg]
    subj_sync = subject_ids_eeg[idx_eeg]

    unique_subjects = np.unique(subj_sync)
    num_folds = len(unique_subjects)
    EPOCHS_CV = 500
    dim_names = ['Valence', 'Arousal', 'Dominance']
    fold_maes, fold_rmses, fold_r2s, fold_pears = [], [], [], []

    t_fus_start = time.time()
    print(f"\n[3/3] Training HybridFusion LOSO ({num_folds} folds, {EPOCHS_CV} epochs/fold)...")
    for f_idx, test_subj in enumerate(unique_subjects):
        train_idx = (subj_sync != test_subj)
        test_idx = (subj_sync == test_subj)

        train_ds = HybridEmbeddingsDataset(z_eeg_sync[train_idx], z_ecg_sync[train_idx], y_sync[train_idx])
        test_ds = HybridEmbeddingsDataset(z_eeg_sync[test_idx], z_ecg_sync[test_idx], y_sync[test_idx])
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        hybrid_fold = HybridFusion(
            input_dim_eeg=z_eeg_sync.shape[1],
            input_dim_ecg=z_ecg_sync.shape[1],
            hidden_dim=256, output_dim=3, dropout=0.5, lambda_align=0.01
        ).to(device)
        optimizer = optim.Adam(hybrid_fold.parameters(), lr=1e-4, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        hybrid_fold.train()
        for _ in range(EPOCHS_CV):
            for x_eeg, _, x_ecg, y in train_loader:
                x_eeg, x_ecg, y = x_eeg.to(device), x_ecg.to(device), y.to(device)
                pred, loss_align = hybrid_fold(x_eeg, x_ecg, return_loss=True)
                loss_total = loss_fn(pred, y) + hybrid_fold.lambda_align * loss_align
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

        hybrid_fold.eval()
        with torch.no_grad():
            preds, trues = [], []
            for x_eeg, _, x_ecg, y in test_loader:
                x_eeg, x_ecg, y = x_eeg.to(device), x_ecg.to(device), y.to(device)
                preds.append(hybrid_fold(x_eeg, x_ecg).cpu())
                trues.append(y.cpu())

        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()

        mae_f = np.mean(np.abs(preds - trues), axis=0)
        rmse_f = np.sqrt(mean_squared_error(trues, preds, multioutput='raw_values'))
        r2_f = np.array([r2_score(trues[:, i], preds[:, i]) for i in range(trues.shape[1])])
        pear_f = np.array([pearsonr(trues[:, i], preds[:, i])[0] if len(trues) > 1 and np.std(preds[:, i]) > 0 and np.std(trues[:, i]) > 0 else 0.0 for i in range(trues.shape[1])])

        fold_maes.append(mae_f)
        fold_rmses.append(rmse_f)
        fold_r2s.append(r2_f)
        fold_pears.append(pear_f)
    t_fus = time.time() - t_fus_start
    total_time = time.time() - t0

    fold_maes = np.array(fold_maes)
    fold_rmses = np.array(fold_rmses)
    fold_r2s = np.array(fold_r2s)
    fold_pears = np.array(fold_pears)

    fold_records = []
    for f_idx, test_subj in enumerate(unique_subjects):
        for d_idx, dim in enumerate(dim_names):
            fold_records.append({
                'Subject': test_subj,
                'Dimension': dim,
                'MAE': fold_maes[f_idx, d_idx],
                'RMSE': fold_rmses[f_idx, d_idx],
                'R²': fold_r2s[f_idx, d_idx],
                'r': fold_pears[f_idx, d_idx]
            })
    df_folds = pd.DataFrame(fold_records)
    folds_csv = OUT_DIR / "fusion_loso_folds_foldwise.csv"
    df_folds.to_csv(folds_csv, index=False)

    print(f"\nDone! Timing: WESAD GRU (15 folds) = {t_gru:.1f}s | DREAMER CNN-BiLSTM (23 folds) = {t_eeg:.1f}s | Fusion LOSO (23 folds) = {t_fus:.1f}s | Total = {total_time:.1f}s")
    print(f"Saved fold-wise CSV to: {folds_csv}")


if __name__ == "__main__":
    main()
