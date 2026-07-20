import sys
from pathlib import Path
import csv

# Paths setup
ROOT_DIR = Path(__file__).resolve().parent
cnnbilstm_path = ROOT_DIR / "dreamer" / "output" / "loso_cnnbilstm.csv"
fusion_path = ROOT_DIR / "transfer_learning_output" / "transfer_learning_loso_folds.csv"

print("==================================================")
print("DREAMER Table 4 Generator: CNN-BiLSTM vs. Fusion")
print("==================================================\n")

missing_any = False

if not cnnbilstm_path.exists():
    print(f"[-] CNN-BiLSTM results file not found at '{cnnbilstm_path.relative_to(ROOT_DIR)}'.")
    print("    Please run the CNN-BiLSTM training script first:")
    print("    python dreamer/models/cnnbilstm/main.py")
    print()
    missing_any = True

if not fusion_path.exists():
    print(f"[-] Multimodal Fusion results file not found at '{fusion_path.relative_to(ROOT_DIR)}'.")
    print("    Please run the transfer learning script first:")
    print("    python transfer_learning.py")
    print()
    missing_any = True

if missing_any:
    print("--------------------------------------------------")
    print("NOTE: Please run the missing scripts above first to generate the necessary folds data.")
    print("--------------------------------------------------")
    sys.exit(1)

# Helper to compute mean and population standard deviation
def get_stats(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mean_val = sum(vals) / n
    var_val = sum((x - mean_val) ** 2 for x in vals) / n
    std_val = var_val ** 0.5
    return mean_val, std_val

# 1. Parse CNN-BiLSTM (EEG Only) CSV
cnnbilstm_data = {
    'Valence': {'MAE': [], 'RMSE': [], 'R2': [], 'r': []},
    'Arousal': {'MAE': [], 'RMSE': [], 'r': [], 'R2': []},
    'Dominance': {'MAE': [], 'RMSE': [], 'r': [], 'R2': []}
}

dim_map = {
    'Val': 'Valence',
    'Aro': 'Arousal',
    'Dom': 'Dominance'
}

with cnnbilstm_path.open("r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if not row or row[0].strip().startswith("#"):
            continue
        for col_idx, col_name in enumerate(header):
            if col_name == "subject":
                continue
            parts = col_name.split("_")
            if len(parts) == 2:
                met, dim_abbrev = parts
                dim = dim_map.get(dim_abbrev)
                # Map standard metrics names
                if met == 'R2':
                    met_key = 'R2'
                elif met == 'r':
                    met_key = 'r'
                else:
                    met_key = met
                if dim and met_key in cnnbilstm_data[dim]:
                    cnnbilstm_data[dim][met_key].append(float(row[col_idx]))

# 2. Parse Fusion (EEG + ECG) CSV
fusion_data = {
    'Valence': {'MAE': [], 'RMSE': [], 'R2': [], 'r': []},
    'Arousal': {'MAE': [], 'RMSE': [], 'R2': [], 'r': []},
    'Dominance': {'MAE': [], 'RMSE': [], 'R2': [], 'r': []}
}

with fusion_path.open("r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    
    sub_idx = header.index("Subject")
    dim_idx = header.index("Dimension")
    mae_idx = header.index("MAE")
    rmse_idx = header.index("RMSE")
    r2_idx = header.index("R²")
    r_idx = header.index("r")
    
    for row in reader:
        if not row:
            continue
        dim = row[dim_idx]
        if dim in fusion_data:
            fusion_data[dim]['MAE'].append(float(row[mae_idx]))
            fusion_data[dim]['RMSE'].append(float(row[rmse_idx]))
            fusion_data[dim]['R2'].append(float(row[r2_idx]))
            fusion_data[dim]['r'].append(float(row[r_idx]))

# 3. Print markdown and LaTeX tables
dimensions = ['Valence', 'Arousal', 'Dominance']
metrics = ['MAE', 'RMSE', 'R2', 'r']
metric_labels = {'MAE': 'MAE', 'RMSE': 'RMSE', 'R2': 'R²', 'r': 'r'}

print("==========================================================================================")
print("Table 4: Comparative Performance (LOSO Cross-Validation) — CNN-BiLSTM vs. Multimodal Fusion")
print("==========================================================================================")
print(f"{'Dimension':<12} | {'Metric':<6} | {'CNN-BiLSTM (EEG Only)':<28} | {'Multimodal Fusion (EEG + ECG)':<28}")
print("-" * 88)

for dim in dimensions:
    for idx, met in enumerate(metrics):
        # Stats for CNN-BiLSTM
        c_mean, c_std = get_stats(cnnbilstm_data[dim][met])
        # Stats for Fusion
        f_mean, f_std = get_stats(fusion_data[dim][met])
        
        c_str = f"{c_mean:.4f} ± {c_std:.4f}"
        f_str = f"{f_mean:.4f} ± {f_std:.4f}"
        
        dim_str = dim if idx == 0 else ""
        print(f"{dim_str:<12} | {metric_labels[met]:<6} | {c_str:<28} | {f_str:<28}")
    print("-" * 88)

print("==========================================================================================")
print("\nMarkdown Table format (perfect for direct copy-paste to your report/manuscript):\n")

print("| Dimension | Metric | CNN-BiLSTM (EEG Only) | Multimodal Fusion (EEG + ECG) |")
print("| :--- | :--- | :--- | :--- |")
for dim in dimensions:
    for met in metrics:
        c_mean, c_std = get_stats(cnnbilstm_data[dim][met])
        f_mean, f_std = get_stats(fusion_data[dim][met])
        print(f"| {dim} | {metric_labels[met]} | {c_mean:.4f} ± {c_std:.4f} | {f_mean:.4f} ± {f_std:.4f} |")
