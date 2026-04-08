# Physiological Signal Processing for Affective Computing

This repository contains the full data preprocessing and machine learning pipeline developed for a study on automatic emotion and stress recognition from physiological signals. Two publicly available datasets are used: **WESAD** (stress classification from wearable sensors) and **DREAMER** (continuous affect prediction from EEG).

All models are evaluated under a **Leave-One-Subject-Out (LOSO)** cross-validation scheme, which simulates a subject-independent scenario — the most realistic and challenging setting for affective computing research.

---

## Repository Structure

```
.
├── datasets/                        # Raw dataset archives (not versioned)
│   ├── WESAD.zip
│   └── DREAMER.zip
│
├── shared/                          # Cross-dataset utilities
│   └── classification/
│       ├── metrics.py               # Accuracy, F1, precision, recall
│       └── plots.py                 # Confusion matrix, per-class heatmap
│
├── wesad/                           # WESAD pipeline
│   ├── config.py                    # Paths, sampling rates, window sizes
│   ├── features.py                  # Window extraction, statistical aggregation
│   ├── pipeline.py                  # Subject loading, processing, CSV export
│   ├── main.py                      # Entry point
│   └── models/
│       ├── gru/
│       │   ├── dataset.py           # Sliding-window PhysioDataset
│       │   ├── model.py             # PhysioGRUEncoder architecture
│       │   └── main.py              # LOSO training and evaluation
│       ├── lightgbm/
│       │   └── main.py
│       └── xgboost/
│           └── main.py
│
└── dreamer/                         # DREAMER pipeline
    ├── config.py                    # Paths, EEG parameters, band definitions
    ├── features.py                  # Band power extraction, channel aggregation
    ├── pipeline.py                  # Subject loading, processing, CSV export
    ├── main.py                      # Entry point
    └── models/
        ├── shared/
        │   ├── metrics.py           # MAE, RMSE, R², Pearson r
        │   └── plots.py             # Pearson heatmaps, distribution boxplots
        ├── cnnbilstm/
        │   ├── dataset.py           # Sequence builder, Dataset, collate_fn
        │   ├── model.py             # CNNBiLSTM architecture
        │   └── main.py              # LOSO training and evaluation
        └── lightgbm/
            └── main.py
```

---

## Datasets

### WESAD — Wearable Stress and Affect Detection

A multimodal dataset for stress and affect detection collected from 15 subjects using two wearable devices: a chest-worn RespiBAN (700 Hz) and a wrist-worn Empatica E4.

- **Article:** [Schmidt et al., 2018 — ACM ICMI](https://dl.acm.org/doi/10.1145/3242969.3242985)
- **Download:** [University of Siegen — USI Data](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)

Signals: ACC, ECG, EMG, EDA, Temperature, Respiration (chest); ACC, BVP, EDA, Temperature (wrist).

| Label | Condition |
|-------|-----------|
| 1 | Baseline |
| 2 | Stress |
| 3 | Amusement |
| 4 | Meditation |
| 0, 5, 6, 7 | Excluded (transient / undefined) |

---

### DREAMER — EEG and ECG Affect Recognition

A dataset for continuous emotion recognition collected from 23 subjects using an Emotiv EPOC EEG headset (14 channels, 128 Hz) while participants watched 18 video clips. Affect is self-reported on three dimensions using the Self-Assessment Manikin (SAM) scale (1–5).

- **Article:** [Katsigiannis & Ramzan, 2018 — IEEE TNSRE](https://ieeexplore.ieee.org/document/7887697)
- **Download:** [Zenodo — Record 546113](https://zenodo.org/records/546113)

> **Access note:** DREAMER files are restricted on Zenodo. To request access, submit your name, e-mail, institution, position, and reason for requesting the database on the Zenodo record page.

EEG channels (Emotiv EPOC layout): AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4.

Affect dimensions: Valence, Arousal, Dominance.

---

## Requirements

```bash
pip install numpy pandas scipy neurokit2 lightgbm xgboost scikit-learn torch seaborn matplotlib tqdm
```

| Package | Used for |
|---------|----------|
| `torch` | GRU and CNN-BiLSTM models |
| `lightgbm` | LightGBM classifiers and regressors |
| `xgboost` | XGBoost classifier (WESAD) |
| `neurokit2` | DREAMER preprocessing pipeline |
| `scikit-learn` | LOSO splits, scalers, metrics |
| `scipy` | Band power estimation (Welch's method), Pearson r |

---

## Setup

1. Clone this repository.
2. Place the dataset archives inside `datasets/`:

```
datasets/
├── WESAD.zip
└── DREAMER.zip
```

The preprocessing pipelines extract the archives automatically on first run.

---

## Preprocessing

Run each pipeline once to generate the cleaned CSV files required by the models.

```bash
python wesad/main.py
python dreamer/main.py
```

### WESAD preprocessing

1. Loads each subject's `.pkl` file.
2. Slides a non-overlapping 10-second window (7000 samples at 700 Hz) over the recording.
3. Discards windows with mixed or invalid labels.
4. Computes statistical aggregates (std, min, max) per channel per window.
5. Decomposes 3-axis ACC arrays into x/y/z columns.
6. Outputs `wesad/output/final_wesad.csv`.

### DREAMER preprocessing

1. Loads `DREAMER.mat`.
2. For each subject and video stimulus, slides a 10-second window (1280 samples, 50% overlap) over the EEG signal.
3. Estimates band power (delta, theta, alpha, beta, gamma) and spectral ratios (beta/alpha, (beta+gamma)/(alpha+theta)) per channel per window using Welch's method.
4. Aggregates window-level features into per-channel and per-video statistics.
5. Outputs `dreamer/output/dreamer_bandpower_all.csv` and `brain_variation_per_video.csv`.

---

## Models

All models use LOSO cross-validation: in each fold, one subject is held out as the test set and the model is trained from scratch on the remaining subjects. Features are standardized per fold using the training set statistics to prevent data leakage.

---

### WESAD — Stress and Affect Classification

Task: classify each physiological window into one of four states (Baseline, Stress, Amusement, Meditation).

Reported metrics: Accuracy, macro F1, macro Precision, macro Recall.

#### GRU (Bidirectional)

A multi-layer bidirectional GRU that processes sequences of physiological feature windows. The last hidden states of the forward and backward passes are concatenated into a context vector and passed through two fully-connected layers for classification. The sliding-window approach (window=10, stride=5) introduces temporal context that flat models cannot capture.

```bash
python wesad/models/gru/main.py
```

#### LightGBM

A gradient boosting classifier trained on flat per-window feature vectors. `sum` features are excluded based on empirical results showing improved performance without them. Class weights are balanced to account for the uneven distribution of affective states.

```bash
python wesad/models/lightgbm/main.py
```

#### XGBoost

Same setup as LightGBM. Labels are shifted to 0-indexed as required by XGBoost's internal representation.

```bash
python wesad/models/xgboost/main.py
```

---

### DREAMER — Continuous Affect Prediction

Task: predict valence, arousal and dominance scores (1–5) from EEG band power features.

Reported metrics per dimension: MAE, RMSE, R², Pearson r.

#### CNN-BiLSTM

A hybrid architecture that treats each video's EEG recording as a temporal sequence of band power windows. A 1-D temporal CNN extracts local patterns across windows, a Bidirectional LSTM captures long-range sequential dependencies, and mean pooling over time produces a fixed-size representation for the regression head. Variable-length sequences across videos are handled via padding and masking.

```bash
python dreamer/models/cnnbilstm/main.py
```

#### LightGBM

A `MultiOutputRegressor` wrapping a `LGBMRegressor` trained on per-video brain variation statistics (mean and std of each band per channel, aggregated across all windows of a video). Provides a strong non-sequential baseline for comparison with the CNN-BiLSTM.

```bash
python dreamer/models/lightgbm/main.py
```

---

## Output Files

**WESAD** — `wesad/output/`

| File | Description |
|------|-------------|
| `{subject_id}.csv` | Per-subject feature CSV |
| `final_wesad.csv` | All subjects merged |

**DREAMER** — `dreamer/output/`

| File | Description |
|------|-------------|
| `subject_bandpower_by_video/subject_{id}/sub_{id}_bandpower_video_{v}.csv` | Window-level band power per subject per video |
| `agg_bandpower_by_subject/sub_{id}_agg_bandpower.csv` | All videos aggregated per subject |
| `dreamer_bandpower_all.csv` | All subjects merged |
| `subject_bandpower_by_video/brain_variation_per_video.csv` | Per-video statistics across all subjects |
