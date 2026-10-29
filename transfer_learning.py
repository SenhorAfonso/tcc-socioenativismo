# -*- coding: utf-8 -*-
"""
Transfer learning pipeline for continuous affect prediction using multimodal fusion (EEG + ECG).

Expected inputs:
- wesad/output/final_wesad.csv
- dreamer/output/dreamer_bandpower_all.csv

If these files do not exist, the script attempts to run the preprocessing pipelines.
If raw datasets are missing, a descriptive error message is shown.
If pretrained weights are missing, the base models are automatically trained first.
"""

import os
import sys
import random
import subprocess
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Global setup
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
EPOCHS_BASE = 15
BATCH_SIZE = 128
SEQ_LEN = 10
STRIDE = 5
LR = 1e-3
NUM_WORKERS = 0  # Set to 0 for maximum compatibility across OS/Windows environments

# ---------------------------------------------------------------------------
# Base Architectures
# ---------------------------------------------------------------------------

class PhysioGRUEncoder(nn.Module):
    """
    Bidirectional GRU encoder for physiological signals (ECG/WESAD).
    Includes forward_features to extract latent embeddings.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=4,
        n_layers=2,
        dropout=0.3,
        bidirectional=True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(self.hidden_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        if self.bidirectional:
            context = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            context = h_n[-1, :, :]
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward_features(self, x):
        _, h_n = self.gru(x)
        if self.bidirectional:
            context = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            context = h_n[-1, :, :]
        return context


class CNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM architecture for EEG sequences (DREAMER).
    Includes forward_features to extract latent embeddings.
    """
    def __init__(
        self,
        input_dim,
        hidden_cnn=64,
        hidden_lstm=128,
        output_dim=3,
        dropout=0.3
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_cnn,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_cnn)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=hidden_cnn,
            hidden_size=hidden_lstm,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(hidden_lstm * 2, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x, padding_mask):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

    def forward_features(self, x, padding_mask):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return out

# ---------------------------------------------------------------------------
# Datasets & Collators
# ---------------------------------------------------------------------------

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
        x_seq = self.X[start_idx : start_idx + self.seq_len]
        y_seq = self.y[start_idx + self.seq_len - 1]
        return x_seq, y_seq


class DreamerSeqDataset(Dataset):
    def __init__(self, X_list, y_list):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.y = [torch.tensor(y, dtype=torch.float32) for y in y_list]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        seq = self.X[i]
        seq_len = seq.shape[0]
        return seq.unsqueeze(0), self.y[i], seq_len


def collate_fn(batch):
    sequences, targets, seq_lengths = zip(*batch)
    padded_sequences = pad_sequence([seq.squeeze(0) for seq in sequences], batch_first=True, padding_value=0.0)
    padding_mask = torch.zeros_like(padded_sequences, dtype=torch.bool)[:, :, 0]
    for i, seq_len in enumerate(seq_lengths):
        padding_mask[i, seq_len:] = True
    stacked_targets = torch.stack(targets)
    return padded_sequences.unsqueeze(1), stacked_targets, torch.tensor(seq_lengths), padding_mask


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

# ---------------------------------------------------------------------------
# Preprocessing Scripts Handlers (Subprocesses)
# ---------------------------------------------------------------------------

def run_preprocess_wesad():
    print("Starting WESAD preprocessing pipeline in the background...", flush=True)
    print("This process is CPU-intensive and may take a few minutes. Real-time logs are shown below:\n", flush=True)
    result = subprocess.run([sys.executable, "main.py"], cwd=ROOT_DIR / "wesad")
    if result.returncode != 0:
        print("\nError: WESAD preprocessing pipeline failed.", file=sys.stderr, flush=True)
        sys.exit(1)
    print("\nWESAD preprocessing pipeline completed successfully.\n", flush=True)


def run_preprocess_dreamer():
    print("Starting DREAMER preprocessing pipeline in the background...", flush=True)
    print("This process is CPU-intensive and may take a few minutes. Real-time logs are shown below:\n", flush=True)
    result = subprocess.run([sys.executable, "main.py"], cwd=ROOT_DIR / "dreamer")
    if result.returncode != 0:
        print("\nError: DREAMER preprocessing pipeline failed.", file=sys.stderr, flush=True)
        sys.exit(1)
    print("\nDREAMER preprocessing pipeline completed successfully.\n", flush=True)

# ---------------------------------------------------------------------------
# Pre-training Base Models (if weights are missing)
# ---------------------------------------------------------------------------

def train_and_save_ecg_model(data_file, save_path, device):
    print("Training base WESAD GRU model from scratch for transfer learning purposes...")
    data = pd.read_csv(data_file)
    features = [col for col in data.columns if col not in ["subject", "label"]]
    X = data[features].values
    y = data["label"].values - 1

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    dataset = PhysioDataset(X, y, seq_len=SEQ_LEN, stride=STRIDE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = PhysioGRUEncoder(input_dim=len(features), hidden_dim=128, output_dim=data["label"].nunique(), bidirectional=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS_BASE):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  [WESAD Base] Epoch {epoch + 1}/{EPOCHS_BASE} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Base WESAD GRU model saved to {save_path}\n")


def train_and_save_eeg_model(data_file, save_path, device):
    print("Training base DREAMER CNN-BiLSTM model from scratch for transfer learning purposes...")
    df = pd.read_csv(data_file)
    feature_cols = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'beta/alpha', '(beta+gamma)/(alpha+theta)']
    X_list, y_list = build_sequences(df, feature_cols)

    dataset = DreamerSeqDataset(X_list, y_list)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = CNNBiLSTM(input_dim=X_list[0].shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.L1Loss()

    model.train()
    for epoch in range(EPOCHS_BASE):
        total_loss = 0.0
        for xb, yb, _, padding_mask in loader:
            xb = xb.squeeze(1).transpose(1, 2).to(device)
            yb = yb.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            pred = model(xb, padding_mask)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  [DREAMER Base] Epoch {epoch + 1}/{EPOCHS_BASE} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Base DREAMER CNN-BiLSTM model saved to {save_path}\n")

# ---------------------------------------------------------------------------
# Multimodal Fusion & Environment Simulation
# ---------------------------------------------------------------------------

class HybridEmbeddingsDataset(Dataset):
    def __init__(self, z_eeg, z_ecg, y, mask_eeg=None):
        self.z_eeg = z_eeg.clone().detach() if isinstance(z_eeg, torch.Tensor) else torch.tensor(z_eeg, dtype=torch.float32)
        self.z_ecg = z_ecg.clone().detach() if isinstance(z_ecg, torch.Tensor) else torch.tensor(z_ecg, dtype=torch.float32)
        self.y = y.clone().detach() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
        self.mask_eeg = None
        if mask_eeg is not None:
            self.mask_eeg = mask_eeg.clone().detach() if isinstance(mask_eeg, torch.Tensor) else torch.tensor(mask_eeg, dtype=torch.bool)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.mask_eeg is not None:
            return self.z_eeg[idx], self.mask_eeg[idx], self.z_ecg[idx], self.y[idx]
        else:
            return self.z_eeg[idx], torch.zeros(self.z_eeg[idx].shape[0], dtype=torch.bool), self.z_ecg[idx], self.y[idx]


class HybridFusion(nn.Module):
    """
    Multimodal fusion model (EEG + ECG).
    Projects both modalities into a joint latent space, applies consistency
    regularization (lambda_align), and predicts continuous valence, arousal,
    and dominance vectors.
    """
    def __init__(
        self,
        input_dim_eeg: int,
        input_dim_ecg: int,
        hidden_dim: int = 256,
        output_dim: int = 3,
        dropout: float = 0.5,
        lambda_align: float = 0.01
    ):
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

        h = torch.cat([h_eeg, h_ecg], dim=1)
        out = self.fusion(h)

        loss_align = F.mse_loss(h_eeg, h_ecg)

        if return_loss:
            return out, loss_align
        else:
            return out


class SimulatedEnvironment:
    """Heuristic environment that updates emotional state based on actions."""
    def __init__(self):
        self.actions = ["light+", "light-", "sound+", "sound-", "pause", "none"]

    def step(self, state, action):
        val, aro, dom = state
        if action == "light+":
            val += 0.05
            aro += 0.05
        elif action == "light-":
            val += 0.02
            aro -= 0.08
        elif action == "sound+":
            aro += 0.10
        elif action == "sound-":
            aro -= 0.10
            val += 0.03
        elif action == "pause":
            aro -= 0.15
            val += 0.08
        val, aro = np.clip(val, 0, 1), np.clip(aro, 0, 1)
        new_state = np.array([val, aro, dom])
        reward = self.reward(state, new_state)
        return new_state, reward

    def reward(self, old, new):
        # Reward increases for higher valence and lower arousal
        dv, da = new[0] - old[0], old[1] - new[1]
        return dv + da


class SimplePolicy:
    """Action-selection policy based on current estimated state."""
    def __init__(self, actions, randomness=0.1):
        self.actions = actions
        self.randomness = randomness

    def select_action(self, state):
        val, aro, dom = state
        if np.random.rand() < self.randomness:
            return np.random.choice(self.actions)
        if aro > 0.7:
            return "sound-"
        if val < 0.4:
            return "light+"
        if aro < 0.3:
            return "sound+"
        return "none"


def simulate_interaction(model, env, policy, n_steps=10, noise_std=0.02):
    log = []
    model.eval()
    with torch.no_grad():
        # Start state is slightly stressed (low valence, high arousal)
        state = np.array([
            np.random.uniform(0.2, 0.4),
            np.random.uniform(0.7, 0.9),
            np.random.uniform(0.4, 0.6)
        ])

        for step in range(n_steps):
            state += np.random.normal(0, noise_std, size=state.shape)
            state = np.clip(state, 0, 1)

            action = policy.select_action(state)
            new_state, reward = env.step(state, action)

            log.append((step, state.copy(), action, new_state.copy(), reward))
            state = new_state
    return log

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths Setup
    data_file_wesad = ROOT_DIR / "wesad" / "output" / "final_wesad.csv"
    data_file_dreamer = ROOT_DIR / "dreamer" / "output" / "dreamer_bandpower_all.csv"

    # 1. Dataset Preprocessing Validation
    if not data_file_wesad.exists():
        print(f"WESAD file not found at: {data_file_wesad}")
        zip_wesad = ROOT_DIR / "datasets" / "WESAD.zip"
        if not zip_wesad.exists():
            print("ERROR: WESAD.zip not found in datasets/ directory. Please place the corresponding file there.")
            sys.exit(1)
        run_preprocess_wesad()

    if not data_file_dreamer.exists():
        print(f"DREAMER file not found at: {data_file_dreamer}")
        zip_dreamer = ROOT_DIR / "datasets" / "DREAMER.zip"
        if not zip_dreamer.exists():
            print("ERROR: DREAMER.zip not found in datasets/ directory. Please place the corresponding file there.")
            sys.exit(1)
        run_preprocess_dreamer()

    # 2. Weights Pre-training Check
    # NOTE (methodological): the frozen base encoders used below are trained ONCE on all subjects of
    # their datasets (train_and_save_*_model); they are not retrained within each fold. The LOSO loop
    # in step 7 therefore applies to the fusion network only, and the fusion results are not strictly
    # subject-independent at the encoder level (see README, "Notes on the fusion experiment").
    ecg_weights_path = ROOT_DIR / "wesag_model_gru.pth"
    # Fallback to alternative names if exists
    if not ecg_weights_path.exists() and (ROOT_DIR / "wesad_model_gru.pth").exists():
        ecg_weights_path = ROOT_DIR / "wesad_model_gru.pth"

    eeg_weights_path = ROOT_DIR / "dreamer_model_cnnbilstm.pth"

    if not ecg_weights_path.exists():
        train_and_save_ecg_model(data_file_wesad, ecg_weights_path, device)

    if not eeg_weights_path.exists():
        train_and_save_eeg_model(data_file_dreamer, eeg_weights_path, device)

    # 3. Load Preprocessed Data
    print("Loading preprocessed datasets...")
    data_wesad = pd.read_csv(data_file_wesad)
    data_dreamer = pd.read_csv(data_file_dreamer)

    # 4. Initialize & Load ECG Encoder
    features_wesad = [col for col in data_wesad.columns if col not in ["subject", "label"]]
    subjects_wesad = data_wesad["subject"].unique()
    num_features_wesad = len(features_wesad)
    num_classes_wesad = data_wesad["label"].nunique()

    # Scaling WESAD
    scaler = StandardScaler()
    data_wesad[features_wesad] = scaler.fit_transform(data_wesad[features_wesad])

    model_ecg = PhysioGRUEncoder(
        input_dim=num_features_wesad,
        hidden_dim=128,
        output_dim=num_classes_wesad,
        bidirectional=True
    ).to(device)
    model_ecg.load_state_dict(torch.load(ecg_weights_path, map_location=device, weights_only=True))

    # Extract ECG features
    embeddings_ecg = []
    labels_ecg = []
    
    for test_subject in tqdm(subjects_wesad, desc="Extracting ECG embeddings (WESAD)"):
        # We slice on each subject and run inference
        test_df = data_wesad[data_wesad.subject == test_subject]
        X_test = test_df[features_wesad].values
        y_test = test_df["label"].values - 1

        test_ds = PhysioDataset(X_test, y_test, seq_len=SEQ_LEN, stride=STRIDE)
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        model_ecg.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                z_ecg = model_ecg.forward_features(xb)
                embeddings_ecg.append(z_ecg.cpu())
                labels_ecg.append(yb)

    # 5. Initialize & Load EEG Encoder
    feature_cols_dreamer = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'beta/alpha', '(beta+gamma)/(alpha+theta)']
    subjects_dreamer = data_dreamer["subject_id"].unique()
    X_list_dreamer, y_list_dreamer = build_sequences(data_dreamer, feature_cols_dreamer)

    model_eeg = CNNBiLSTM(input_dim=X_list_dreamer[0].shape[1]).to(device)
    model_eeg.load_state_dict(torch.load(eeg_weights_path, map_location=device, weights_only=True))

    embeddings_eeg = []
    labels_eeg = []
    subject_ids_eeg = []

    for test_subj in tqdm(subjects_dreamer, desc="Extracting EEG embeddings (DREAMER)"):
        # extract for the subject
        test_idx = [i for i, (s, v) in enumerate(data_dreamer.groupby(["subject_id", "video_id"]).groups.keys()) if s == test_subj]
        Xte = [X_list_dreamer[i] for i in test_idx]
        yte = [y_list_dreamer[i] for i in test_idx]

        test_ds = DreamerSeqDataset(Xte, yte)
        test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

        model_eeg.eval()
        with torch.no_grad():
            for xb, yb, _, padding_mask in test_dl:
                xb = xb.squeeze(1).transpose(1, 2).to(device)
                padding_mask = padding_mask.to(device)
                z_eeg = model_eeg.forward_features(xb, padding_mask)
                embeddings_eeg.append(z_eeg.cpu())
                labels_eeg.append(yb)
                subject_ids_eeg.extend([test_subj] * len(yb))

    # 6. Concat and Align Lengths
    z_eeg_tensor = torch.cat(embeddings_eeg, dim=0)
    z_ecg_tensor = torch.cat(embeddings_ecg, dim=0)
    y_tensor     = torch.cat(labels_eeg, dim=0)
    subject_ids_eeg = np.array(subject_ids_eeg)

    print(f"EEG extracted: {len(z_eeg_tensor)} | ECG extracted: {len(z_ecg_tensor)}")

    min_len = min(len(z_eeg_tensor), len(z_ecg_tensor))
    print(f"Randomly pairing to common length of: {min_len} samples...")

    # Synced pairing
    idx_ecg = np.random.choice(len(z_ecg_tensor), size=min_len, replace=False)
    idx_eeg = np.random.choice(len(z_eeg_tensor), size=min_len, replace=False)

    z_ecg_sync = z_ecg_tensor[idx_ecg]
    z_eeg_sync = z_eeg_tensor[idx_eeg]
    y_sync     = y_tensor[idx_eeg]
    subj_sync  = subject_ids_eeg[idx_eeg]

    # 7. Leave-One-Subject-Out (LOSO) Cross-Validation
    unique_subjects = np.unique(subj_sync)
    num_folds = len(unique_subjects)
    print(f"Starting Leave-One-Subject-Out (LOSO) Cross-Validation with {num_folds} folds...")

    EPOCHS_CV = 500
    dim_names = ['Valence', 'Arousal', 'Dominance']
    
    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    fold_pears = []

    for f_idx, test_subj in enumerate(unique_subjects):
        train_idx = (subj_sync != test_subj)
        test_idx  = (subj_sync == test_subj)

        z_eeg_train, z_ecg_train, y_train = z_eeg_sync[train_idx], z_ecg_sync[train_idx], y_sync[train_idx]
        z_eeg_test, z_ecg_test, y_test = z_eeg_sync[test_idx], z_ecg_sync[test_idx], y_sync[test_idx]

        train_ds = HybridEmbeddingsDataset(z_eeg_train, z_ecg_train, y_train)
        test_ds  = HybridEmbeddingsDataset(z_eeg_test, z_ecg_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

        # Initialize fold model
        hybrid_fold = HybridFusion(
            input_dim_eeg=z_eeg_sync.shape[1],
            input_dim_ecg=z_ecg_sync.shape[1],
            hidden_dim=256,
            output_dim=3,
            dropout=0.5,
            lambda_align=0.01
        ).to(device)

        optimizer = optim.Adam(hybrid_fold.parameters(), lr=1e-4, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        hybrid_fold.train()
        fold_loss = 0.0
        for epoch in range(EPOCHS_CV):
            total_loss = 0.0
            for x_eeg, _, x_ecg, y in train_loader:
                x_eeg, x_ecg, y = x_eeg.to(device), x_ecg.to(device), y.to(device)

                pred, loss_align = hybrid_fold(x_eeg, x_ecg, return_loss=True)
                loss_main = loss_fn(pred, y)
                loss_total = loss_main + hybrid_fold.lambda_align * loss_align

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                total_loss += loss_total.item()
            fold_loss = total_loss / len(train_loader)

        # Evaluate fold model
        hybrid_fold.eval()
        with torch.no_grad():
            preds, trues = [], []
            for x_eeg, _, x_ecg, y in test_loader:
                x_eeg, x_ecg, y = x_eeg.to(device), x_ecg.to(device), y.to(device)
                out = hybrid_fold(x_eeg, x_ecg)
                preds.append(out.cpu())
                trues.append(y.cpu())

        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()

        # Calculate metrics for this subject
        mae_f = np.mean(np.abs(preds - trues), axis=0)
        rmse_f = np.sqrt(mean_squared_error(trues, preds, multioutput='raw_values'))
        r2_f = np.array([r2_score(trues[:, i], preds[:, i]) for i in range(trues.shape[1])])
        pear_f = np.array([pearsonr(trues[:, i], preds[:, i])[0] if len(trues) > 1 and np.std(preds[:, i]) > 0 and np.std(trues[:, i]) > 0 else 0.0 for i in range(trues.shape[1])])

        fold_maes.append(mae_f)
        fold_rmses.append(rmse_f)
        fold_r2s.append(r2_f)
        fold_pears.append(pear_f)

        print(f"  [Fold {f_idx+1:02d}/{num_folds}] Test Subject {test_subj:02d} | samples: {test_idx.sum()} | Last Loss: {fold_loss:.4f} | Mean MAE: {np.mean(mae_f):.3f}")

    # Convert to arrays for statistical aggregation
    fold_maes = np.array(fold_maes)
    fold_rmses = np.array(fold_rmses)
    fold_r2s = np.array(fold_r2s)
    fold_pears = np.array(fold_pears)

    # 8. Statistical Aggregation (Mean, Std, Min, Max, Range)
    mean_mae = np.mean(fold_maes, axis=0)
    std_mae  = np.std(fold_maes, axis=0)
    min_mae  = np.min(fold_maes, axis=0)
    max_mae  = np.max(fold_maes, axis=0)

    mean_rmse = np.mean(fold_rmses, axis=0)
    std_rmse  = np.std(fold_rmses, axis=0)
    min_rmse  = np.min(fold_rmses, axis=0)
    max_rmse  = np.max(fold_rmses, axis=0)

    mean_r2 = np.mean(fold_r2s, axis=0)
    std_r2  = np.std(fold_r2s, axis=0)
    min_r2  = np.min(fold_r2s, axis=0)
    max_r2  = np.max(fold_r2s, axis=0)

    mean_pear = np.mean(fold_pears, axis=0)
    std_pear  = np.std(fold_pears, axis=0)
    min_pear  = np.min(fold_pears, axis=0)
    max_pear  = np.max(fold_pears, axis=0)

    print("==================================================")
    print("Leave-One-Subject-Out (LOSO) Cross-Validation Results:")
    print("==================================================")
    for i, dim in enumerate(dim_names):
        print(f"\nDimension: {dim}")
        print(f"  MAE  | Mean: {mean_mae[i]:.4f} | Std: {std_mae[i]:.4f} | Range: [{min_mae[i]:.4f} – {max_mae[i]:.4f}]")
        print(f"  RMSE | Mean: {mean_rmse[i]:.4f} | Std: {std_rmse[i]:.4f} | Range: [{min_rmse[i]:.4f} – {max_rmse[i]:.4f}]")
        print(f"  R²   | Mean: {mean_r2[i]:.4f} | Std: {std_r2[i]:.4f} | Range: [{min_r2[i]:.4f} – {max_r2[i]:.4f}]")
        print(f"  r    | Mean: {mean_pear[i]:.4f} | Std: {std_pear[i]:.4f} | Range: [{min_pear[i]:.4f} – {max_pear[i]:.4f}]")
    print("==================================================")

    # 9. Output Generation & Metrics Saving
    output_dir = ROOT_DIR / "transfer_learning_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary metrics to CSV
    df_metrics = pd.DataFrame({
        'MAE': mean_mae,
        'RMSE': mean_rmse,
        'R²': mean_r2,
        'r': mean_pear
    }, index=dim_names)
    df_metrics.to_csv(output_dir / "transfer_learning_metrics.csv")

    # Save detailed fold-by-fold metrics
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
    df_folds.to_csv(output_dir / "transfer_learning_loso_folds.csv", index=False)

    # Generate Mean Metrics Heatmap
    plt.figure(figsize=(7, 4))
    sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap='YlGnBu', cbar_kws={'label': 'Value'})
    plt.title("Mean Performance per Dimension (LOSO)")
    plt.xlabel("Metric")
    plt.ylabel("Dimension")
    plt.tight_layout()
    plt.savefig(output_dir / "transfer_learning_metrics.png")
    plt.close()

    # 10. Train Final Model on 100% of Data for Simulation
    print("\nTraining final Multimodal Fusion model on 100% of paired data for simulation...")
    final_hybrid = HybridFusion(
        input_dim_eeg=z_eeg_sync.shape[1],
        input_dim_ecg=z_ecg_sync.shape[1],
        hidden_dim=256,
        output_dim=3,
        dropout=0.5,
        lambda_align=0.01
    ).to(device)

    optimizer = optim.Adam(final_hybrid.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    EPOCHS_FINAL = 500
    train_ds_all = HybridEmbeddingsDataset(z_eeg_sync, z_ecg_sync, y_sync)
    train_loader_all = DataLoader(train_ds_all, batch_size=64, shuffle=True)

    final_hybrid.train()
    for epoch in range(EPOCHS_FINAL):
        total_loss = 0
        for x_eeg, _, x_ecg, y in train_loader_all:
            x_eeg, x_ecg, y = x_eeg.to(device), x_ecg.to(device), y.to(device)

            pred, loss_align = final_hybrid(x_eeg, x_ecg, return_loss=True)
            loss_main = loss_fn(pred, y)
            loss_total = loss_main + final_hybrid.lambda_align * loss_align

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            total_loss += loss_total.item()
        if epoch % 100 == 0 or epoch == EPOCHS_FINAL - 1:
            print(f"Final Model Train Epoch {epoch}/{EPOCHS_FINAL} | Loss: {total_loss / len(train_loader_all):.4f}")

    # 11. Simulation of Socio-enactive Environment
    print("\nStarting simulation of socio-enactive environment...")
    env = SimulatedEnvironment()
    policy = SimplePolicy(env.actions)

    log = simulate_interaction(final_hybrid, env, policy, n_steps=30)  # 30 iterations, matching the manuscript

    # Write simulation log
    with open(output_dir / "simulation_log.txt", "w", encoding="utf-8") as f:
        for step, s, a, s2, r in log:
            f.write(f"Step {step:02d} | State: {s.round(2)} | Action: {a:6s} | "
                    f"New: {s2.round(2)} | Reward: {r:+.3f}\n")

    # Plot simulation states
    states = np.array([s for _, s, _, _, _ in log])
    plt.figure(figsize=(10, 5))
    plt.plot(states[:, 0], label='Valence', color='blue', linewidth=2, marker='o', markersize=4)
    plt.plot(states[:, 1], label='Arousal', color='red', linewidth=2, marker='s', markersize=4)
    plt.plot(states[:, 2], label='Dominance', color='green', linewidth=2, marker='^', markersize=4)  # now plotted, matching the manuscript
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylim(0, 1)
    plt.title("Evolution of Simulated Affective States over 30 Enactive Iterations")
    plt.xlabel("Interaction Steps")
    plt.ylabel("State Level")
    plt.tight_layout()
    plt.savefig(output_dir / "transfer_learning_simulation.png")
    plt.close()

    print(f"Results and plots successfully saved in: {output_dir}")


if __name__ == "__main__":
    main()
