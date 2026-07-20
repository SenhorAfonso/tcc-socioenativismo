"""
Per-subject processing pipeline for the DREAMER dataset.

Responsibilities:
- Extract DREAMER.zip (if not already done).
- Load the DREAMER.mat file.
- For each subject, extract window-level band power per video and save CSVs.
- Aggregate per-video and per-subject CSVs.
- Merge all subjects into a final dataset.
- Compute brain variation statistics per video across all subjects.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from config import (
    BRAIN_VARIATION_CSV_NAME,
    DREAMER_EXTRACTED_DIR,
    DREAMER_MAT,
    DREAMER_ZIP,
    EEG_FS,
    FINAL_CSV_NAME,
    OUTPUT_BY_SUBJECT_DIR,
    OUTPUT_BY_VIDEO_DIR,
    OUTPUT_DIR,
    WINDOW_OVERLAP,
    WINDOW_SAMPLES,
)
from features import (
    extract_window_bandpower,
    get_variation_per_channel,
    get_variation_per_video,
    sliding_windows,
)


def extract_dataset() -> None:
    """
    Extract ``DREAMER.zip`` into the datasets directory if not already done.

    Raises
    ------
    FileNotFoundError
        If ``DREAMER.zip`` is not found at the path defined in ``config.DREAMER_ZIP``.
    """
    if DREAMER_EXTRACTED_DIR.exists():
        return

    if not DREAMER_ZIP.exists():
        raise FileNotFoundError(
            f"DREAMER archive not found at '{DREAMER_ZIP}'. "
            "Please place DREAMER.zip inside the datasets/ directory."
        )

    print(f"Extracting {DREAMER_ZIP} ...")
    DREAMER_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DREAMER_ZIP, "r") as zf:
        zf.extractall(DREAMER_EXTRACTED_DIR)
    print("Extraction complete.")


def load_dataset() -> list:
    """
    Load the DREAMER.mat file and return the list of subject objects.

    Returns
    -------
    list
        List of MATLAB struct objects, one per subject.
    """
    raw = loadmat(str(DREAMER_MAT), squeeze_me=True, struct_as_record=False)
    return raw["DREAMER"].Data


def process_subject_videos(subject_id: int, subjects: list) -> None:
    """
    Extract window-level band power for every video of a single subject
    and save one CSV per video.

    Parameters
    ----------
    subject_id : int
        Zero-based subject index.
    subjects : list
        Full list of subject objects loaded from the .mat file.
    """
    subject = subjects[subject_id]
    subject_eeg = subject.EEG.stimuli

    out_dir = OUTPUT_BY_VIDEO_DIR / f"subject_{subject_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for video_id in range(subject_eeg.shape[0]):
        video_signal = subject_eeg[video_id]
        windows = sliding_windows(video_signal, WINDOW_SAMPLES, WINDOW_OVERLAP)

        print(
            f"  [S{subject_id}] Video {video_id}: "
            f"{windows.shape[0]} windows × {windows.shape[1]} samples × {windows.shape[2]} channels"
        )

        window_dfs = []
        for win_idx, window in enumerate(windows):
            df = extract_window_bandpower(window, EEG_FS)
            df["window_id"]  = win_idx
            df["video_id"]   = video_id
            df["subject_id"] = subject_id
            df["arousal"]    = subject.ScoreArousal[video_id]
            df["dominance"]  = subject.ScoreDominance[video_id]
            df["valence"]    = subject.ScoreValence[video_id]
            window_dfs.append(df)

        video_df = pd.concat(window_dfs, ignore_index=True)
        video_df.to_csv(out_dir / f"sub_{subject_id}_bandpower_video_{video_id}.csv", index=False)


def aggregate_subject_bandpower(subject_id: int) -> None:
    """
    Concatenate all per-video CSVs for a subject into a single aggregated CSV.

    Parameters
    ----------
    subject_id : int
        Zero-based subject index.
    """
    subject_dir = OUTPUT_BY_VIDEO_DIR / f"subject_{subject_id}"
    csv_files = sorted(subject_dir.glob("*.csv"))

    combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    OUTPUT_BY_SUBJECT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_BY_SUBJECT_DIR / f"sub_{subject_id}_agg_bandpower.csv", index=False)


def aggregate_all_subjects() -> pd.DataFrame:
    """
    Merge all per-subject aggregated CSVs into a single final DataFrame.

    Returns
    -------
    pd.DataFrame
        Combined dataset for all subjects.
    """
    csv_files = sorted(OUTPUT_BY_SUBJECT_DIR.glob("*.csv"))
    combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_DIR / FINAL_CSV_NAME, index=False)
    return combined


def compute_brain_variation() -> pd.DataFrame:
    """
    Compute per-video brain variation statistics across all subjects and videos.

    Reads each per-video CSV, computes channel-level and video-level aggregations,
    and saves the result to ``brain_variation_per_video.csv``.

    Returns
    -------
    pd.DataFrame
        Brain variation DataFrame with one row per (subject, video) pair.
    """
    video_variations = []

    for subject_dir in sorted(OUTPUT_BY_VIDEO_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        print(f"  Reading {subject_dir.name} ...")
        for csv_file in sorted(subject_dir.glob("*.csv")):
            data = pd.read_csv(csv_file)
            variation_channel = get_variation_per_channel(data)
            variation_video   = get_variation_per_video(variation_channel)
            video_variations.append(variation_video)

    result = pd.concat(video_variations, ignore_index=True)

    out_path = OUTPUT_BY_VIDEO_DIR / BRAIN_VARIATION_CSV_NAME
    result.to_csv(out_path, index=False)
    print(f"  Brain variation saved to {out_path}")
    return result
