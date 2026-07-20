import sys
from pathlib import Path
import csv

# Paths setup
ROOT_DIR = Path(__file__).resolve().parent
output_dir = ROOT_DIR / "wesad" / "output"

models = {
    "GRU": output_dir / "loso_gru.csv",
    "LightGBM": output_dir / "loso_lightgbm.csv",
    "XGBoost": output_dir / "loso_xgboost.csv"
}

print("==================================================")
print("WESAD Classification Models LOSO Metrics Extractor")
print("==================================================\n")

missing_any = False

for name, path in models.items():
    if not path.exists():
        print(f"[-] {name} results file not found at '{path.relative_to(ROOT_DIR)}'.")
        print("    Please run the training script in your active python environment to generate it:")
        if name == "GRU":
            print("    python wesad/models/gru/main.py")
        elif name == "LightGBM":
            print("    python wesad/models/lightgbm/main.py")
        elif name == "XGBoost":
            print("    python wesad/models/xgboost/main.py")
        print()
        missing_any = True
        continue
    
    # Load accuracy column using built-in csv module (no pandas/numpy needed)
    accs = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        acc_idx = header.index("accuracy")
        
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue  # Skip blank lines and footer comments
            accs.append(float(row[acc_idx]))
            
    if not accs:
        print(f"[-] {name} CSV file is empty or invalid.\n")
        continue

    # Pure Python statistics (ddof=0 for population std, ddof=1 for sample std)
    n = len(accs)
    mean_acc = sum(accs) / n
    
    var_pop = sum((x - mean_acc) ** 2 for x in accs) / n
    std_acc_pop = var_pop ** 0.5
    
    if n > 1:
        var_sample = sum((x - mean_acc) ** 2 for x in accs) / (n - 1)
        std_acc_sample = var_sample ** 0.5
    else:
        std_acc_sample = 0.0
        
    min_acc = min(accs)
    max_acc = max(accs)
    
    print(f"[+] {name} (from {path.name}):")
    print(f"    Format (Mean ± Std):  {mean_acc:.4f} ± {std_acc_pop:.4f}  (ddof=0, population std)")
    print(f"    Format (Mean ± Std):  {mean_acc:.4f} ± {std_acc_sample:.4f}  (ddof=1, sample std)")
    print(f"    Min-Max Range      :  [{min_acc:.4f} – {max_acc:.4f}]")
    print()

if missing_any:
    print("--------------------------------------------------")
    print("NOTE: Please run the missing training scripts above first.")
    print("--------------------------------------------------")
else:
    print("--------------------------------------------------")
    print("All models successfully processed.")
    print("--------------------------------------------------")
