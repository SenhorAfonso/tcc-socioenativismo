"""Paired between-model comparisons from the published per-subject LOSO logs.

The script reads the curated CSV files directly from ``results/`` (rows beginning
with '#' are ignored); no copies to the model output directories are required:
  WESAD   : results/wesad_loso_{gru,lightgbm,xgboost}.csv
            (columns: subject, accuracy, f1, ...)
  DREAMER : results/dreamer_loso_cnnbilstm.csv
            (columns: subject, MAE_<dim>, R2_<dim>, ...)
  Fusion  : results/fusion_loso_folds.csv
  Fold-wise fusion: results/foldwise_encoders/fusion_loso_folds_foldwise.csv
            (long format: Subject, Dimension, MAE, RMSE, R², r)

For every specified comparison it reports, over the paired subjects:
  mean difference, 95% CI (Student's t on the per-subject differences),
  95% percentile-bootstrap CI (10,000 resamples), two-sided Wilcoxon
  signed-rank p, Holm-adjusted p within each family, and Cohen's dz.
It also recomputes each model's mean, population and sample SD, and 95% t-CI
from the raw per-subject values, so they can be checked against the manuscript.

Usage (from the repository root):
    python analysis/paired_stats.py
    python analysis/paired_stats.py . /temporary/output/directory

The second form writes the same six CSV outputs to a separate directory for
verification without overwriting the curated outputs. No data are simulated or
altered; the script fails if a file is missing or subjects cannot be paired.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(42)
N_BOOT = 10000


def read_csv(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"missing file: {path}")
    return pd.read_csv(path, comment="#")


def t_ci(x, level=0.95):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * se
    return m - h, m + h


def boot_ci(d, level=0.95):
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return lo, hi


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def describe(name, x):
    x = np.asarray(x, float)
    lo, hi = t_ci(x)
    return dict(model=name, n=len(x), mean=x.mean(), sd_population=x.std(ddof=0), sd_sample=x.std(ddof=1),
                ci95_low=lo, ci95_high=hi, min=x.min(), max=x.max())


def compare(a_name, a, b_name, b, metric):
    """a, b: pandas Series indexed by subject. Difference = a - b."""
    common = a.index.intersection(b.index)
    if len(common) != len(a) or len(common) != len(b):
        sys.exit(f"subjects cannot be paired for {a_name} vs {b_name} ({len(a)} vs {len(b)} vs {len(common)} common)")
    d = (a.loc[common] - b.loc[common]).to_numpy(float)
    lo, hi = t_ci(d)
    blo, bhi = boot_ci(d)
    nz = d[d != 0]
    p = stats.wilcoxon(d).pvalue if len(nz) > 0 else 1.0
    sd = d.std(ddof=1)
    return dict(metric=metric, comparison=f"{a_name} - {b_name}", n_pairs=len(d), mean_diff=d.mean(),
                t_ci95_low=lo, t_ci95_high=hi, boot_ci95_low=blo, boot_ci95_high=bhi,
                wilcoxon_p=p, cohen_dz=(d.mean() / sd if sd > 0 else np.nan))


def wesad(root):
    files = {"GRU": "wesad_loso_gru.csv", "LightGBM": "wesad_loso_lightgbm.csv", "XGBoost": "wesad_loso_xgboost.csv"}
    data = {m: read_csv(root / "results" / f) for m, f in files.items()}
    desc, comps = [], []
    for metric in ("accuracy", "f1"):
        series = {m: df.set_index(df["subject"].astype(str))[metric] for m, df in data.items()}
        desc += [dict(metric=metric, **describe(m, s.to_numpy())) for m, s in series.items()]
        rows = [compare(a, series[a], b, series[b], metric) for a, b in combinations(series, 2)]
        adj = holm([r["wilcoxon_p"] for r in rows])
        for r, q in zip(rows, adj):
            r["wilcoxon_p_holm"] = q
        comps += rows
    return pd.DataFrame(desc), pd.DataFrame(comps)


def dreamer(root, fusion_path=None, fusion_name="Fusion"):
    uni = read_csv(root / "results" / "dreamer_loso_cnnbilstm.csv")
    if fusion_path is None:
        fusion_path = root / "results" / "fusion_loso_folds.csv"
    fus = read_csv(fusion_path)
    uni = uni.set_index(uni["subject"].astype(str))
    dims = {"Valence": "Val", "Arousal": "Aro", "Dominance": "Dom"}
    desc, comps = [], []
    for metric, ucol, fcol in (("MAE", "MAE_{}", "MAE"), ("R2", "R2_{}", "R²")):
        rows = []
        for dim, short in dims.items():
            u = uni[ucol.format(short)]
            f = fus[fus["Dimension"] == dim].set_index(fus[fus["Dimension"] == dim]["Subject"].astype(str))[fcol]
            desc += [dict(metric=f"{metric}_{dim}", **describe("CNN-BiLSTM", u.to_numpy())),
                     dict(metric=f"{metric}_{dim}", **describe(fusion_name, f.to_numpy()))]
            rows.append(compare(fusion_name, f, "CNN-BiLSTM", u, f"{metric}_{dim}"))
        adj = holm([r["wilcoxon_p"] for r in rows])
        for r, q in zip(rows, adj):
            r["wilcoxon_p_holm"] = q
        comps += rows
    return pd.DataFrame(desc), pd.DataFrame(comps)


def dreamer_foldwise(root):
    return dreamer(
        root,
        fusion_path=root / "results" / "foldwise_encoders" / "fusion_loso_folds_foldwise.csv",
        fusion_name="Fusion_Foldwise",
    )


def write_analysis(root, out, name, fn, foldwise=False):
    d, c = fn(root)
    target = out / "foldwise_encoders" if foldwise else out
    target.mkdir(parents=True, exist_ok=True)
    d.to_csv(target / f"paired_stats_{name}_descriptives.csv", index=False)
    c.to_csv(target / f"paired_stats_{name}.csv", index=False)
    print(f"\n=== {name.upper()}: per-model descriptives (recomputed from per-subject values) ===")
    print(d.round(3).to_string(index=False))
    print(f"\n=== {name.upper()}: paired comparisons ===")
    print(c.round(4).to_string(index=False))


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    write_analysis(root, out, "wesad", wesad)
    write_analysis(root, out, "dreamer", dreamer)
    write_analysis(root, out, "dreamer_foldwise", dreamer_foldwise, foldwise=True)
