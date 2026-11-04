# Physiological Signal Processing for Affective Computing

This repository contains the full data preprocessing and machine learning pipeline developed for a study on automatic emotion and stress recognition from physiological signals. It is the companion code of the manuscript *"Enactive Knowledge Creation in Physiological Computing: A Proof-of-Concept Architecture"* (submitted to *Cognitive Computation*). Two benchmark datasets are used: **WESAD** (stress classification from wearable sensors; public download) and **DREAMER** (continuous affect prediction from EEG; available on request, see [Data access](#data-access)).

The unimodal WESAD and DREAMER models are evaluated under a **Leave-One-Subject-Out (LOSO)** cross-validation scheme. The main cross-dataset fusion experiment is not strictly subject-independent at the encoder level: its base encoders are trained once on all subjects, and LOSO applies only to the fusion network. A stricter fold-wise encoder-retraining variant is also provided, but it is not fully nested (see [Notes on the fusion experiment](#notes-on-the-fusion-experiment)).

---

## Repository Structure

```
.
├── datasets/                        # Raw dataset archives (not versioned)
│   ├── WESAD.zip
│   └── DREAMER.zip
│
├── results/                         # Curated outputs accompanying the manuscript
├── transfer_learning.py             # Cross-dataset fusion feasibility test + rule-based simulation
├── extract_loso_metrics.py          # Summarizes WESAD per-subject LOSO CSVs
├── extract_table4.py                # Summarizes DREAMER/fusion per-subject CSVs
├── requirements.txt
├── LICENSE
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

- **Article:** [Katsigiannis & Ramzan, 2018 — IEEE J Biomed Health Inform 22(1):98-107](https://ieeexplore.ieee.org/document/7887697)
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
4. Computes statistical aggregates (sum, std, min, max) per channel per window (56 features per window after splitting the accelerometer axes).
5. Decomposes 3-axis ACC arrays into x/y/z columns.
6. Outputs `wesad/output/final_wesad.csv`.

### DREAMER preprocessing

1. Loads `DREAMER.mat`.
2. For each subject and video stimulus, slides a 10-second window (1280 samples, 50% overlap) over the EEG signal.
3. Estimates band power (delta, theta, alpha, beta, gamma) and spectral ratios (beta/alpha, (beta+gamma)/(alpha+theta)) per channel per window using Welch's method.
4. Aggregates window-level features into per-channel and per-video statistics.
5. Outputs `dreamer/output/dreamer_bandpower_all.csv` and `brain_variation_per_video.csv`.

---

## Multimodal Fusion & Modality Alignment

To ensure a robust, realistic, and reproducible framework for affective computing, the modality alignment and fusion strategies are implemented as follows:

### 1. WESAD — Symmetrical Feature-Level Early Fusion
* **Strategy**: Direct deterministic temporal matching (early fusion / feature-level fusion).
* **Implementation Details**: Sensor signals from chest (CHEST) and wrist (WRIST) are sliced in parallel using a unified reference 10-second window at 700 Hz. The signal slices are deterministic and aligned to the exact same physical timestamps of recording by calculating the sample rates ratio:
  $$\text{start} = \left\lfloor \text{ref\_start} \times \frac{\text{window\_size}}{\text{ref\_window\_size}} \right\rfloor$$
  $$\text{end} = \left\lfloor \text{ref\_end} \times \frac{\text{window\_size}}{\text{ref\_window\_size}} \right\rfloor$$
* **Evidence in Code**: Implemented in [wesad/features.py](wesad/features.py#L101-L116) within `extract_window_features`, where chest and wrist signals are extracted synchronously and stored in the same feature row dict.
* **Scope of this statement**: it concerns the WESAD pipeline and the WESAD/DREAMER unimodal models only, whose chest and wrist features are aligned synchronously. The separate cross-dataset fusion experiment (`transfer_learning.py`) does use **random cross-dataset pairing** of EEG (DREAMER) and peripheral-signal (WESAD) embeddings; it is a feasibility test of the fusion mechanism, not a synchronized multimodal baseline (see [Notes on the fusion experiment](#notes-on-the-fusion-experiment)).

### 2. DREAMER — Unimodal Analysis
* **Strategy**: Single-modality (EEG-only) features.
* **Implementation Details**: The DREAMER ECG (Shimmer2) raw modality was discarded during pipeline engineering to focus exclusively on EEG spectral band powers and brain variation statistics.
* **Evidence in Code**: Declared in [dreamer/config.py](dreamer/config.py#L7):
  ```python
  # - ECG device: Shimmer2 (256 Hz) — not used in this pipeline
  ```

---

## Reproducibility & Determinism

To guarantee 100% reproducibility of deep learning experiments:
* **Global Seed**: `SEED = 42` is defined across PyTorch, NumPy, and Python built-in random generators.
* **CuDNN Determinism**: CUDA operations are forced to run deterministically:
  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
* **Evaluation Loop**: In each fold of Leave-One-Subject-Out (LOSO) cross-validation, models are trained and evaluated exactly **once** using this determinism seed.
* **Auto-logging**: Metric evaluation files are automatically computed and outputted per-subject in CSV format.

---

## Models

The unimodal models use LOSO cross-validation: in each fold, one subject is held out as the test set and the model is trained on the remaining subjects. Preprocessing and normalization are model-specific; per-fold standardization is not applied indiscriminately to every model or variant. The main fusion experiment is an exception to fold-wise encoder training: its encoders are trained once on all subjects, and only the fusion network is evaluated with LOSO. The fold-wise variant retrains the encoders within each fold but is not fully nested.

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

A gradient boosting classifier trained on flat per-window feature vectors. The per-channel `sum` features (14 of the 56) are excluded for LightGBM based on empirical results showing improved performance without them; XGBoost and the GRU use all 56 features. Class weights are balanced to account for the uneven distribution of affective states.

```bash
python wesad/models/lightgbm/main.py
```

#### XGBoost

Same LOSO protocol as LightGBM, but using all 56 features (no `sum` exclusion). Labels are shifted to 0-indexed as required by XGBoost's internal representation.

```bash
python wesad/models/xgboost/main.py
```

---

### DREAMER — Continuous Affect Prediction

Task: predict valence, arousal and dominance scores (1–5) from EEG band power features.

Reported metrics per dimension: MAE, RMSE, R², Pearson r.

#### CNN-BiLSTM

A hybrid architecture that treats each video's EEG recording as a temporal sequence of band power windows. A 1-D temporal CNN extracts local patterns across windows, and a Bidirectional LSTM captures long-range sequential dependencies. Padding masks zero padded input positions before the recurrent layer; the implementation then applies unmasked temporal mean pooling over the full padded sequence length. The pooled representation is passed to the regression head.

```bash
python dreamer/models/cnnbilstm/main.py
```

#### LightGBM

A `MultiOutputRegressor` wrapping a `LGBMRegressor` trained on per-video brain variation statistics (mean and std of each band per channel, aggregated across all windows of a video). Provides a strong non-sequential baseline for comparison with the CNN-BiLSTM.

```bash
python dreamer/models/lightgbm/main.py
```

---

### Multimodal Fusion (Transfer Learning)

Task: predict valence, arousal, and dominance scores from a combination of EEG (DREAMER) and ECG (WESAD) pre-trained feature extractors.

The pipeline:
1. Validates that the preprocessed WESAD and DREAMER data exist.
2. Checks for pre-trained ECG and EEG feature extractors (`wesag_model_gru.pth` and `dreamer_model_cnnbilstm.pth`). If missing, it trains them on the respective source datasets first.
3. Extracts latent embeddings (penultimate layer) of the frozen encoders for all subjects of both datasets.
4. Pairs the embeddings **randomly across datasets** (EEG embeddings from DREAMER, peripheral-signal embeddings from WESAD; no within-subject or temporal correspondence).
5. Trains a `HybridFusion` network to project both modalities into a joint latent space (256-d), using an MSE alignment term with `lambda_align = 0.01`.
6. Evaluates the fusion network with leave-one-subject-out folds over the paired embeddings (grouped by the DREAMER subject) on valence, arousal, and dominance, and writes per-fold metrics to `transfer_learning_loso_folds.csv`.
7. Trains a final fusion network on 100% of the paired data and runs a 30-iteration **rule-based** simulation (see below).

#### Notes on the fusion experiment

* **Feasibility test only.** Random cross-dataset pairing means the experiment tests whether the fusion network can be trained and evaluated end to end; it does not establish whether genuine within-subject multimodal fusion would help. DREAMER contains ECG, but its ECG signals are not used in this repository (`dreamer/config.py`), so a synchronized within-dataset EEG–ECG baseline was **not** evaluated.
* **Encoders are trained once, on all subjects.** `train_and_save_ecg_model` and `train_and_save_eeg_model` (15 epochs each) fit the base encoders on all subjects, and the LOSO loop of step 6 applies to the fusion network only. The held-out subject's data (and, for the EEG encoder, ratings) were therefore seen by the frozen encoders, so the fusion results are not strictly subject-independent at the encoder level and are not strictly comparable with the fold-wise unimodal results. The fold-wise variant in [`results/foldwise_encoders/`](results/foldwise_encoders/) retrains both encoders within each fold and takes the held-out subject's embeddings from an encoder that did not see that subject. It is stricter, but not fully nested: the embeddings used to train the fusion network on other subjects came from fold encoders that had seen the held-out subject. The encoder weights (`*.pth`) are not versioned.
* **Simulation.** The environment (`SimulatedEnvironment`) and `SimplePolicy` are a predefined rule-based controller with stochastic exploration: `arousal > 0.7` -> `sound-`; `valence < 0.4` -> `light+`; `arousal < 0.3` -> `sound+`; otherwise no action; with probability 0.1 a uniformly random action is taken. The thresholds are hard-coded and the policy does not learn. The simulated state is passed directly to the policy and the fusion network is not queried during the simulation (no knowledge-base module is implemented). The log used in the manuscript (30 iterations) is in [`results/simulation_log.txt`](results/simulation_log.txt).

```bash
python transfer_learning.py
```

---

## Output Files

Executing preprocessing and model scripts automatically populates the corresponding `output/` folders.

### Preprocessed Datasets

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

### Leave-One-Subject-Out (LOSO) CSV Logs

To support research transparency, subject-level validation metrics (including accuracy, macro F1, macro precision, and macro recall for WESAD, and MAE, RMSE, R², and Pearson r for DREAMER) are exported to CSV files upon model completion:

* **WESAD**:
  * `wesad/output/loso_gru.csv`: Per-subject results for GRU.
  * `wesad/output/loso_lightgbm.csv`: Per-subject results for LightGBM.
  * `wesad/output/loso_xgboost.csv`: Per-subject results for XGBoost.
* **DREAMER**:
  * `dreamer/output/loso_cnnbilstm.csv`: Per-subject results for CNN-BiLSTM.
  * `dreamer/output/loso_lightgbm.csv`: Per-subject results for LightGBM.

### Multimodal Fusion Logs & Visualizations

Running `transfer_learning.py` generates the following files in `transfer_learning_output/`:
* `transfer_learning_metrics.csv`: Performance metrics (MAE, RMSE, R², Pearson correlation) per dimension.
* `transfer_learning_loso_folds.csv`: Per-subject (fold-level) metrics of the fusion network.
* `transfer_learning_metrics.png`: Visual heatmap of the performance metrics.
* `simulation_log.txt`: Detailed log of the socio-enactive simulated environment interaction.
* `transfer_learning_simulation.png`: Time-series plot of Valence, Arousal and Dominance during the 30-iteration simulation.

---

## Data access

* **WESAD** is publicly available: <https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html>.
* **DREAMER** is distributed on request by its maintainers (Zenodo record 546113, <https://zenodo.org/records/546113>); the raw files are not redistributed here.
* Place the archives in `datasets/` (see [Setup](#setup)). Raw data, intermediate CSV files and model weights are not versioned.

## Reproducing the manuscript results

The curated per-subject CSVs required for statistical verification are already included in [`results/`](results/), so the descriptive and paired analyses can be reproduced without retraining the deep models.

1. Install the dependencies (`pip install -r requirements.txt`; versions used in the reported experiments are pinned: Python 3.12, PyTorch 2.8.0, scikit-learn 1.6.1, NumPy 2.0.2, Pandas 2.2.2).
2. To regenerate the experimental logs, run the preprocessing: `python wesad/main.py` and `python dreamer/main.py`.
3. Run the unimodal LOSO experiments (WESAD: GRU, LightGBM, XGBoost; DREAMER: CNN-BiLSTM, LightGBM), then `python extract_loso_metrics.py` and `python extract_table4.py` (DREAMER unimodal vs. fusion; corresponds to Table 4 of the manuscript) to summarize the per-subject CSV logs.
4. Run the main fusion feasibility experiment and simulation: `python transfer_learning.py`. In this variant, LOSO applies only to the fusion network because the encoders were trained once on all subjects.
5. To regenerate the fold-wise fusion results, run `python results/foldwise_encoders/run_foldwise_fusion.py`. This requires the preprocessed datasets and retrains the encoders within each fold; the variant is stricter but not fully nested.
6. Run `python analysis/paired_stats.py` to recompute the descriptive and paired analyses directly from the curated CSVs in `results/`, including the fold-wise variant. To verify the outputs without overwriting the curated files, use `python analysis/paired_stats.py . /tmp/paired-stats-check`.

All stochastic components use `SEED = 42`; each fold is run once. Summary statistics in the manuscript use the sample standard deviation (denominator *n* − 1) across held-out subjects. The reported 95% confidence intervals are computed from the per-subject values with Student's *t* distribution. The manuscript also reports paired mean differences, paired confidence intervals, two-sided Wilcoxon signed-rank tests with Holm adjustment, and bootstrap robustness checks; `analysis/paired_stats.py` reproduces these analyses from the per-subject logs.

## Results folder

[`results/`](results/) holds the curated outputs that accompany the manuscript. See [`results/README.md`](results/README.md) for what is included and how to regenerate the remaining files.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).
