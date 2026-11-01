# Results accompanying the manuscript

| File | Description |
|------|-------------|
| `simulation_log.txt` | 30-iteration log of the rule-based simulation (Section 4.3, Figure 4 of the manuscript): 23 iterations without action, 3 `sound-` and 4 `light+` interventions. |

`simulation_log.txt` is reproduced exactly, without any data or trained model, by `python analysis/run_simulation_seed42.py` (the simulation uses only NumPy and does not query the fusion network; the script seeds NumPy with 42 and runs `simulate_interaction` from `transfer_learning.py`). In an end-to-end run of `python transfer_learning.py` the global NumPy generator has already been advanced by the random pairing step, so that run draws a different trajectory of the same kind.

## Per-fold files

The per-subject (fold-level) logs behind the mean ± sample SD values in Tables 2 and 4 are versioned in this folder. They support direct verification of the summary statistics and paired analyses without retraining the models.

| Curated file | Generated originally by |
|---|---|
| `wesad_loso_gru.csv` | `wesad/models/gru/main.py` |
| `wesad_loso_lightgbm.csv` | `wesad/models/lightgbm/main.py` |
| `wesad_loso_xgboost.csv` | `wesad/models/xgboost/main.py` |
| `dreamer_loso_cnnbilstm.csv` | `dreamer/models/cnnbilstm/main.py` |
| `fusion_loso_folds.csv` | `transfer_learning.py` (main variant; LOSO applies only to the fusion network) |
| `foldwise_encoders/fusion_loso_folds_foldwise.csv` | `foldwise_encoders/run_foldwise_fusion.py` (fold-wise encoder-retraining variant; not fully nested) |

From the repository root, run:

```bash
python analysis/paired_stats.py
```

This reads the curated inputs directly and writes descriptive and paired outputs for WESAD, the main DREAMER fusion variant, and the fold-wise fusion variant. To verify the analyses without overwriting the curated outputs, specify a separate output directory:

```bash
python analysis/paired_stats.py . /tmp/paired-stats-check
```

The `.gitignore` of this repository ignores the model `output/` folders but not `results/`.
