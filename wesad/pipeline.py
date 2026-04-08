"""
Per-subject processing pipeline for the WESAD dataset.

Responsibilities:
- Extract the WESAD zip archive (if not already extracted).
- Load the raw .pkl file for a given subject.
- Extract windowed features via the feature module.
- Post-process the resulting DataFrame (flatten arrays, decompose ACC axes).
- Persist the cleaned DataFrame as a CSV file.
"""

import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from config import OUTPUT_DIR, WESAD_EXTRACTED_DIR, WESAD_ZIP
from features import decompose_acc_columns, extract_window_features, flatten_scalar_arrays


def extract_dataset() -> None:
    """
    Extract ``WESAD.zip`` into the datasets directory if not already done.

    Raises
    ------
    FileNotFoundError
        If ``WESAD.zip`` is not found at the path defined in ``config.WESAD_ZIP``.
    """
    if WESAD_EXTRACTED_DIR.exists():
        return

    if not WESAD_ZIP.exists():
        raise FileNotFoundError(
            f"WESAD archive not found at '{WESAD_ZIP}'. "
            "Please place WESAD.zip inside the datasets/ directory."
        )

    print(f"Extracting {WESAD_ZIP} ...")
    with zipfile.ZipFile(WESAD_ZIP, "r") as zf:
        zf.extractall(WESAD_ZIP.parent)
    print("Extraction complete.")


def _load_subject_pkl(subject_id: str) -> dict:
    """
    Load the raw pickle file for a subject from its dedicated directory.

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. 'S2'.

    Returns
    -------
    dict
        Raw data dictionary with keys 'subject', 'signal', and 'label'.

    Raises
    ------
    FileNotFoundError
        If the .pkl file does not exist at the expected path.
    """
    pkl_path = WESAD_EXTRACTED_DIR / subject_id / f"{subject_id}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def process_subject(subject_id: str) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline for a single subject.

    Steps:
    1. Load raw .pkl data.
    2. Extract statistical features from valid, pure windows.
    3. Flatten single-element array cells to scalars.
    4. Decompose 3-axis ACC columns into x/y/z.

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. 'S2'.

    Returns
    -------
    pd.DataFrame
        Cleaned feature DataFrame for the subject.
    """
    print(f"[{subject_id}] Loading raw data...")
    raw_data = _load_subject_pkl(subject_id)

    chest_signals: dict[str, NDArray] = raw_data["signal"]["chest"]
    wrist_signals: dict[str, NDArray] = raw_data["signal"]["wrist"]
    labels: NDArray = raw_data["label"]

    print(f"[{subject_id}] Extracting windowed features...")
    rows = extract_window_features(subject_id, chest_signals, wrist_signals, labels)

    df = pd.DataFrame(rows)
    df = flatten_scalar_arrays(df)
    df = decompose_acc_columns(df)

    print(f"[{subject_id}] Done — {len(df)} valid windows extracted.")
    return df


def save_subject_csv(df: pd.DataFrame, subject_id: str) -> Path:
    """
    Persist a subject's feature DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Processed feature DataFrame.
    subject_id : str
        Used to name the output file, e.g. 'S2.csv'.

    Returns
    -------
    Path
        Absolute path to the saved CSV file.
    """
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{subject_id}.csv"
    df.to_csv(output_path, index=False)
    print(f"[{subject_id}] Saved to {output_path}")
    return output_path


def merge_subject_csvs(subject_ids: list[str]) -> pd.DataFrame:
    """
    Concatenate all per-subject CSVs into a single DataFrame.

    Parameters
    ----------
    subject_ids : list[str]
        Ordered list of subject identifiers to merge.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing data from all subjects.
    """
    output_dir = OUTPUT_DIR
    frames = [pd.read_csv(output_dir / f"{sid}.csv") for sid in subject_ids]
    return pd.concat(frames, ignore_index=True)
