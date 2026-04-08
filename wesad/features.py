"""
Feature extraction utilities for the WESAD preprocessing pipeline.

Each valid window produces statistical aggregates (sum, std, min, max)
for every sensor channel. ACC channels are stored as 3-axis arrays and
later decomposed into individual x/y/z columns.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from config import (
    ACC_AGGREGATE_COLUMNS,
    INVALID_LABELS,
    REFERENCE_WINDOW_SIZE,
    WINDOW_SIZES,
)


def _extract_channel_features(
    signal: NDArray,
    prefix: str,
    channel: str,
) -> dict[str, NDArray | float]:
    """
    Compute statistical aggregates for a single signal window.

    Parameters
    ----------
    signal : NDArray
        1-D or 2-D array slice for the channel window.
    prefix : str
        Sensor location prefix, e.g. 'CHEST' or 'WRIST'.
    channel : str
        Channel name, e.g. 'ECG', 'BVP'.

    Returns
    -------
    dict
        Keys follow the pattern ``{prefix}_{channel}_{stat}``.
    """
    key = f"{prefix}_{channel}"
    return {
        f"{key}_sum": np.sum(signal, axis=0),
        f"{key}_std": np.std(signal, axis=0),
        f"{key}_min": np.min(signal, axis=0),
        f"{key}_max": np.max(signal, axis=0),
    }


def extract_window_features(
    subject_id: str,
    chest_signals: dict[str, NDArray],
    wrist_signals: dict[str, NDArray],
    labels: NDArray,
) -> list[dict]:
    """
    Slide a non-overlapping window over the recording and extract features
    from every pure, valid window.

    A window is considered valid when:
    - All labels within it belong to the same class.
    - That class is not in ``INVALID_LABELS``.

    Parameters
    ----------
    subject_id : str
        Identifier for the subject (e.g. 'S2').
    chest_signals : dict
        Dictionary mapping channel names to their raw signal arrays (chest).
    wrist_signals : dict
        Dictionary mapping channel names to their raw signal arrays (wrist).
    labels : NDArray
        1-D array of per-sample labels at the reference frequency (700 Hz).

    Returns
    -------
    list[dict]
        Each element is a flat dictionary representing one feature row,
        ready to be passed to ``pd.DataFrame``.
    """
    rows: list[dict] = []
    pointer: int = 0
    n_samples: int = len(labels)

    while pointer + REFERENCE_WINDOW_SIZE <= n_samples:
        ref_start = pointer
        ref_end = pointer + REFERENCE_WINDOW_SIZE

        window_labels = labels[ref_start:ref_end]
        candidate_label = window_labels[0]

        # Skip windows with mixed or invalid labels
        if candidate_label in INVALID_LABELS or not np.all(window_labels == candidate_label):
            pointer += REFERENCE_WINDOW_SIZE
            continue

        row: dict = {"subject": subject_id, "label": int(candidate_label)}

        # Chest channels
        for channel, window_size in WINDOW_SIZES["chest"].items():
            ratio = window_size / REFERENCE_WINDOW_SIZE
            start = int(ref_start * ratio)
            end = int(ref_end * ratio)
            signal_slice = np.array(chest_signals[channel][start:end])
            row.update(_extract_channel_features(signal_slice, "CHEST", channel))

        # Wrist channels
        for channel, window_size in WINDOW_SIZES["wrist"].items():
            ratio = window_size / REFERENCE_WINDOW_SIZE
            start = int(ref_start * ratio)
            end = int(ref_end * ratio)
            signal_slice = np.array(wrist_signals[channel][start:end])
            row.update(_extract_channel_features(signal_slice, "WRIST", channel))

        rows.append(row)
        pointer += REFERENCE_WINDOW_SIZE

    return rows


def flatten_scalar_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert single-element numpy arrays stored in DataFrame cells to scalars.

    Some aggregation functions (e.g. np.sum on a 1-D array) return a scalar,
    but others may return a length-1 ndarray. This ensures uniform dtypes.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    return df.map(
        lambda x: x[0] if isinstance(x, np.ndarray) and x.ndim == 1 and len(x) == 1 else x
    )


def decompose_acc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand 3-axis ACC aggregate columns into separate x, y, z columns.

    Each ACC aggregate column contains a length-3 array [x, y, z].
    This function replaces each such column with three scalar columns
    suffixed ``_x``, ``_y``, and ``_z``.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    for column in ACC_AGGREGATE_COLUMNS:
        if column not in df.columns:
            continue
        axis_df = pd.DataFrame(
            df[column].tolist(),
            columns=[f"{column}_x", f"{column}_y", f"{column}_z"],
            index=df.index,
        )
        df = pd.concat([df.drop(columns=[column]), axis_df], axis=1)
    return df
