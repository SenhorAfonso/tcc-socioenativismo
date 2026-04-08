"""
Configuration constants for the WESAD dataset preprocessing pipeline.

WESAD (Wearable Stress and Affect Detection) dataset:
- Chest sensor: RespiBAN (700 Hz)
- Wrist sensor: Empatica E4 (varies per signal)

Label mapping:
    0 = Not defined / transient
    1 = Baseline
    2 = Stress
    3 = Amusement
    4 = Meditation
    5, 6, 7 = Additional conditions (excluded)

Expected directory layout:
    datasets/
    └── WESAD.zip   ← raw dataset archive
    wesad/
    └── output/     ← generated CSVs (created automatically)
"""

from pathlib import Path

# Root of the repository (one level above this file)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Path to the WESAD zip archive
WESAD_ZIP: Path = ROOT_DIR / "datasets" / "WESAD.zip"

# Directory where the zip is extracted (temporary, inside datasets/)
WESAD_EXTRACTED_DIR: Path = ROOT_DIR / "datasets" / "WESAD"

# Output directory for cleaned CSVs
OUTPUT_DIR: Path = ROOT_DIR / "wesad" / "output"

# Final merged CSV filename
FINAL_CSV_NAME: str = "final_wesad.csv"

# Subject IDs available in the dataset (S1 and S12 are missing)
SUBJECTS: list[str] = [
    "S2", "S3", "S4", "S5", "S6", "S7",
    "S8", "S9", "S10", "S11", "S13",
    "S14", "S15", "S16", "S17",
]

# Labels to exclude (undefined, transient, or out-of-scope conditions)
INVALID_LABELS: set[int] = {0, 5, 6, 7}

# Window size in samples at the reference frequency (700 Hz).
# 700 samples/s * 10 s = 7000 samples per window.
WINDOW_SECONDS: int = 10

# Sampling rates (Hz) per sensor and channel
SAMPLING_RATES: dict[str, dict[str, int]] = {
    "chest": {
        "ACC":  700,
        "ECG":  700,
        "EMG":  700,
        "EDA":  700,
        "Temp": 700,
        "Resp": 700,
    },
    "wrist": {
        "ACC":  32,
        "BVP":  64,
        "EDA":  4,
        "TEMP": 4,
    },
}

# Samples per window for each channel, derived from sampling rate * window duration
WINDOW_SIZES: dict[str, dict[str, int]] = {
    location: {
        channel: rate * WINDOW_SECONDS
        for channel, rate in channels.items()
    }
    for location, channels in SAMPLING_RATES.items()
}

# Reference window size (chest ECG at 700 Hz)
REFERENCE_WINDOW_SIZE: int = WINDOW_SIZES["chest"]["ECG"]

# ACC columns that contain 3-axis arrays and need to be decomposed into x/y/z
ACC_AGGREGATE_COLUMNS: list[str] = [
    "CHEST_ACC_sum", "CHEST_ACC_std", "CHEST_ACC_min", "CHEST_ACC_max",
    "WRIST_ACC_sum", "WRIST_ACC_std", "WRIST_ACC_min", "WRIST_ACC_max",
]
