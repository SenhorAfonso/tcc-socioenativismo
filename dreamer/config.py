"""
Configuration constants for the DREAMER dataset preprocessing pipeline.

DREAMER (A Database for Emotion Recognition through EEG and ECG Signals
from Wireless Low-cost Off-the-Shelf Devices):
- EEG device: Emotiv EPOC (14 channels, 128 Hz)
- ECG device: Shimmer2 (256 Hz) — not used in this pipeline

Affect labels (per video stimulus, self-reported):
    ScoreValence   : 1-5
    ScoreArousal   : 1-5
    ScoreDominance : 1-5

Expected directory layout:
    datasets/
    └── DREAMER.zip   ← raw dataset archive
    dreamer/
    └── output/       ← generated CSVs (created automatically)
"""

from pathlib import Path

# Root of the repository (one level above this file)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Path to the DREAMER zip archive
DREAMER_ZIP: Path = ROOT_DIR / "datasets" / "DREAMER.zip"

# Directory where the zip is extracted
DREAMER_EXTRACTED_DIR: Path = ROOT_DIR / "datasets" / "DREAMER"

# Path to the .mat file inside the extracted archive
DREAMER_MAT: Path = DREAMER_EXTRACTED_DIR / "DREAMER.mat"

# Output directories
OUTPUT_DIR: Path = ROOT_DIR / "dreamer" / "output"
OUTPUT_BY_VIDEO_DIR: Path = OUTPUT_DIR / "subject_bandpower_by_video"
OUTPUT_BY_SUBJECT_DIR: Path = OUTPUT_DIR / "agg_bandpower_by_subject"

# Final merged CSV filename
FINAL_CSV_NAME: str = "dreamer_bandpower_all.csv"
BRAIN_VARIATION_CSV_NAME: str = "brain_variation_per_video.csv"

# EEG sampling frequency (Hz)
EEG_FS: int = 128

# Sliding window parameters
WINDOW_SAMPLES: int = EEG_FS * 10   # 10-second windows → 1280 samples
WINDOW_OVERLAP: float = 0.5         # 50% overlap

# EEG channel index-to-name mapping (Emotiv EPOC layout)
CHANNEL_MAP: dict[int, str] = {
    0:  "AF3",
    1:  "F7",
    2:  "F3",
    3:  "FC5",
    4:  "T7",
    5:  "P7",
    6:  "O1",
    7:  "O2",
    8:  "P8",
    9:  "T8",
    10: "FC6",
    11: "F4",
    12: "F8",
    13: "AF4",
}

CHANNELS: list[str] = list(CHANNEL_MAP.values())

# EEG frequency bands (Hz)
FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
