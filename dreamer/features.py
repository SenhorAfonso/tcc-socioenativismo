"""
Feature extraction utilities for the DREAMER preprocessing pipeline.

Pipeline per subject:
1. Slide windows over each video's EEG signal.
2. For each window, compute band power (delta, theta, alpha, beta, gamma)
   and derived ratios per channel.
3. Aggregate window-level features into per-channel and per-video statistics.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import welch

from config import (
    CHANNEL_MAP,
    CHANNELS,
    EEG_FS,
    FREQUENCY_BANDS,
    WINDOW_OVERLAP,
    WINDOW_SAMPLES,
)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def sliding_windows(signal: NDArray, window_size: int, overlap: float) -> NDArray:
    """
    Generate a view of overlapping windows over a 2-D signal using stride tricks.

    Parameters
    ----------
    signal : NDArray, shape (n_samples, n_channels)
        Raw EEG signal for a single video stimulus.
    window_size : int
        Number of samples per window.
    overlap : float
        Fractional overlap between consecutive windows (0 ≤ overlap < 1).

    Returns
    -------
    NDArray, shape (n_windows, window_size, n_channels)
    """
    step = int(window_size * (1 - overlap))
    n_windows = (signal.shape[0] - window_size) // step + 1
    shape = (n_windows, window_size, signal.shape[1])
    strides = (signal.strides[0] * step, signal.strides[0], signal.strides[1])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)


# ---------------------------------------------------------------------------
# Band power
# ---------------------------------------------------------------------------

def bandpower(signal: NDArray, fs: int, band: tuple[float, float]) -> float:
    """
    Estimate the power of a signal within a frequency band using Welch's method.

    Parameters
    ----------
    signal : NDArray, shape (n_samples,)
        Single-channel EEG signal.
    fs : int
        Sampling frequency in Hz.
    band : tuple[float, float]
        (fmin, fmax) frequency range in Hz.

    Returns
    -------
    float
        Estimated band power (area under the PSD curve within the band).
    """
    fmin, fmax = band
    freqs, psd = welch(signal, fs=fs, nperseg=fs * 2)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_window_bandpower(window: NDArray, fs: int = EEG_FS) -> pd.DataFrame:
    """
    Compute band power features for every channel in a single EEG window.

    In addition to the five canonical bands, two spectral ratios are computed:
    - beta/alpha
    - (beta + gamma) / (alpha + theta)

    Parameters
    ----------
    window : NDArray, shape (n_samples, n_channels)
        EEG window.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    pd.DataFrame
        One row per channel with columns for each band and ratio,
        plus a 'channel' column with the channel name.
    """
    rows = []
    for ch_idx in range(window.shape[1]):
        ch_signal = window[:, ch_idx]
        bp = {name: bandpower(ch_signal, fs, band) for name, band in FREQUENCY_BANDS.items()}

        bp["beta/alpha"] = bp["beta"] / bp["alpha"] if bp["alpha"] != 0 else 0.0
        bp["(beta+gamma)/(alpha+theta)"] = (
            (bp["beta"] + bp["gamma"]) / (bp["alpha"] + bp["theta"])
            if (bp["alpha"] + bp["theta"]) != 0 else 0.0
        )

        bp["channel"] = CHANNEL_MAP[ch_idx]
        rows.append(bp)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def get_variation_per_channel(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate window-level band power into per-channel statistics for one video.

    For each channel, computes the mean and standard deviation of every band
    and ratio across all windows.

    Parameters
    ----------
    data : pd.DataFrame
        Window-level band power DataFrame for a single (subject, video) pair.
        Must contain columns: channel, subject_id, video_id, valence, arousal,
        dominance, and one column per band/ratio.

    Returns
    -------
    pd.DataFrame
        One row per channel with mean/std statistics.
    """
    band_columns = list(FREQUENCY_BANDS.keys()) + ["beta/alpha", "(beta+gamma)/(alpha+theta)"]
    rows = []

    for channel in CHANNELS:
        channel_data = data[data["channel"] == channel]
        row: dict = {
            "channel":    channel,
            "subject_id": data["subject_id"].iloc[0],
            "video_id":   data["video_id"].iloc[0],
            "valence":    data["valence"].iloc[0],
            "arousal":    data["arousal"].iloc[0],
            "dominance":  data["dominance"].iloc[0],
        }
        for band in band_columns:
            row[f"{band}_mean"] = channel_data[band].mean()
            row[f"{band}_std"]  = channel_data[band].std()

        rows.append(row)

    return pd.DataFrame(rows)


def get_variation_per_video(data: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-channel statistics into a single row representing one video.

    Computes the mean of each channel-level statistic across all channels.

    Parameters
    ----------
    data : pd.DataFrame
        Per-channel statistics DataFrame returned by ``get_variation_per_channel``.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with averaged statistics across channels.
    """
    band_columns = list(FREQUENCY_BANDS.keys()) + ["beta/alpha", "(beta+gamma)/(alpha+theta)"]
    stat_columns = [f"{band}_{stat}" for band in band_columns for stat in ("mean", "std")]

    row: dict = {
        "subject_id": data["subject_id"].iloc[0],
        "video_id":   data["video_id"].iloc[0],
        "valence":    data["valence"].iloc[0],
        "arousal":    data["arousal"].iloc[0],
        "dominance":  data["dominance"].iloc[0],
    }
    for col in stat_columns:
        row[f"{col}_mean"] = data[col].mean()

    return pd.DataFrame([row])
