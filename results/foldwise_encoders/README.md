# Fold-wise Encoder Retraining — Multimodal Fusion (`transfer_learning.py`)

In the original [`transfer_learning.py`](../../transfer_learning.py), the frozen base encoders (`PhysioGRUEncoder` on WESAD and `CNNBiLSTM` on DREAMER) are trained once on all subjects (15 epochs each), and the 23-fold LOSO loop applies only to the `HybridFusion` network.

This folder contains the stricter fold-wise evaluation where both base encoders are retrained from scratch inside each LOSO fold (15 folds for the WESAD GRU encoder with per-fold `StandardScaler`, and 23 folds for the DREAMER CNN-BiLSTM encoder, 15 epochs per fold, `SEED = 42`) before extracting the test subject's embeddings and running the 23-fold `HybridFusion` LOSO evaluation. The held-out subject's embeddings come from an encoder that did not see that subject. The variant is not fully nested, however: the embeddings used to train the fusion network on the other subjects came from fold encoders that had seen the held-out subject.

## Execution Time (CPU — 13th Gen Intel Core i7-13650HX)
- **WESAD GRU encoder (15 folds, 15 epochs/fold):** `91.2 s`
- **DREAMER CNN-BiLSTM encoder (23 folds, 15 epochs/fold):** `325.3 s`
- **HybridFusion network LOSO (23 folds, 500 epochs/fold):** `380.4 s`
- **Total runtime:** `798.9 s` (~13.3 minutes)

## Files
- [`fusion_loso_folds_foldwise.csv`](fusion_loso_folds_foldwise.csv): Per-subject (fold-level) metrics (`Subject, Dimension, MAE, RMSE, R², r`) with fold-wise retrained encoders.
- [`paired_stats_dreamer_foldwise_descriptives.csv`](paired_stats_dreamer_foldwise_descriptives.csv): Recomputed per-model descriptive statistics (`CNN-BiLSTM` vs. `Fusion_Foldwise`).
- [`paired_stats_dreamer_foldwise.csv`](paired_stats_dreamer_foldwise.csv): Paired comparisons (`Fusion_Foldwise - CNN-BiLSTM`) with 95% Student's *t* CI, 95% percentile bootstrap CI, two-sided Wilcoxon signed-rank *p*, Holm-adjusted *p*, and paired Cohen's $d_z$.
- [`run_foldwise_fusion.py`](run_foldwise_fusion.py): Script used to run the fold-wise encoder retraining and LOSO fusion evaluation.

## Reproduction

The published per-subject CSV is sufficient to recompute the descriptive and paired statistics without retraining. From the repository root, run `python analysis/paired_stats.py`; the routine processes the main and fold-wise fusion variants together. To regenerate the fold-wise experimental log itself, run `python results/foldwise_encoders/run_foldwise_fusion.py` with the preprocessed datasets available. This retrains the models and is not required for the statistical verification.

## Terminal Output

```text
=== DREAMER (FOLD-WISE ENCODERS): per-model descriptives ===
       metric           model  n   mean  sd_population  sd_sample  ci95_low  ci95_high    min    max
  MAE_Valence      CNN-BiLSTM 23  1.172          0.189      0.193     1.088      1.255  0.751  1.572
  MAE_Valence Fusion_Foldwise 23  1.382          0.191      0.196     1.297      1.466  1.071  1.775
  MAE_Arousal      CNN-BiLSTM 23  0.948          0.151      0.154     0.881      1.015  0.675  1.336
  MAE_Arousal Fusion_Foldwise 23  1.083          0.217      0.222     0.987      1.179  0.697  1.608
MAE_Dominance      CNN-BiLSTM 23  0.931          0.233      0.238     0.828      1.034  0.525  1.580
MAE_Dominance Fusion_Foldwise 23  1.132          0.226      0.231     1.032      1.232  0.620  1.736
   R2_Valence      CNN-BiLSTM 23 -0.159          0.189      0.193    -0.242     -0.075 -0.843  0.029
   R2_Valence Fusion_Foldwise 23 -0.701          0.385      0.393    -0.871     -0.531 -1.630 -0.117
   R2_Arousal      CNN-BiLSTM 23 -0.370          0.564      0.577    -0.620     -0.121 -2.150  0.134
   R2_Arousal Fusion_Foldwise 23 -0.817          0.993      1.016    -1.256     -0.377 -4.365  0.088
 R2_Dominance      CNN-BiLSTM 23 -0.319          0.515      0.527    -0.547     -0.091 -1.755  0.248
 R2_Dominance Fusion_Foldwise 23 -0.988          0.961      0.983    -1.413     -0.563 -3.453  0.114

=== DREAMER (FOLD-WISE ENCODERS): paired comparisons ===
       metric                   comparison  n_pairs  mean_diff  t_ci95_low  t_ci95_high  boot_ci95_low  boot_ci95_high  wilcoxon_p  cohen_dz  wilcoxon_p_holm
  MAE_Valence Fusion_Foldwise - CNN-BiLSTM       23     0.2099      0.1247       0.2951         0.1312          0.2887      0.0001    1.0651           0.0002
  MAE_Arousal Fusion_Foldwise - CNN-BiLSTM       23     0.1352      0.0413       0.2290         0.0463          0.2212      0.0101    0.6230           0.0101
MAE_Dominance Fusion_Foldwise - CNN-BiLSTM       23     0.2013      0.1066       0.2960         0.1141          0.2886      0.0006    0.9192           0.0011
   R2_Valence Fusion_Foldwise - CNN-BiLSTM       23    -0.5419     -0.7361      -0.3477        -0.7196         -0.3619      0.0000   -1.2067           0.0001
   R2_Arousal Fusion_Foldwise - CNN-BiLSTM       23    -0.4464     -0.7999      -0.0929        -0.7799         -0.1286      0.0043   -0.5461           0.0043
 R2_Dominance Fusion_Foldwise - CNN-BiLSTM       23    -0.6688     -1.0216      -0.3160        -0.9988         -0.3548      0.0004   -0.8197           0.0008
```
