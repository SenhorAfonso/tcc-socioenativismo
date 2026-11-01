"""Data-free reproduction of the 30-iteration rule-based simulation (Section 4.3, Figure 4).

The simulation uses only NumPy: the state is passed directly to the policy and the trained fusion network is not
queried. This script takes SimulatedEnvironment, SimplePolicy and simulate_interaction verbatim from
transfer_learning.py (by parsing the file, so neither PyTorch nor the datasets are needed), seeds NumPy with 42
and writes the log to the path given as argument (default: results/simulation_log_standalone.txt).

Usage (from the repository root):
    python analysis/run_simulation_seed42.py
    python analysis/run_simulation_seed42.py --n-runs 1000     # distribution over seeds 0..999 (robustness summary)

Note: in the end-to-end run of transfer_learning.py the global NumPy generator has already been advanced by the
random pairing step before the simulation starts, so that run draws a different trajectory of the same kind.
"""
import argparse, collections, contextlib, sys, types
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "transfer_learning.py").read_text(encoding="utf-8")


def grab(kind, name):
    lines = SRC[SRC.index(f"{kind} {name}"):].split("\n")
    out = [lines[0]]
    for ln in lines[1:]:
        if ln and not ln.startswith((" ", "\t", "#")):
            break
        out.append(ln)
    return "\n".join(out)


ns = {"np": np, "torch": types.SimpleNamespace(no_grad=contextlib.nullcontext)}
for kind, name in (("class", "SimulatedEnvironment"), ("class", "SimplePolicy"), ("def", "simulate_interaction")):
    exec(grab(kind, name), ns)


class _NoModel:          # simulate_interaction() only calls model.eval(); the model is never used
    def eval(self):
        pass


def run(seed, n_steps=30):
    np.random.seed(seed)
    env = ns["SimulatedEnvironment"]()
    return ns["simulate_interaction"](_NoModel(), env, ns["SimplePolicy"](env.actions), n_steps=n_steps)


def fmt(log):
    return "".join(f"Step {st:02d} | State: {s.round(2)} | Action: {a:6s} | New: {s2.round(2)} | Reward: {r:+.3f}\n"
                   for st, s, a, s2, r in log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(ROOT / "results" / "simulation_log_standalone.txt"))
    ap.add_argument("--n-runs", type=int, default=0)
    a = ap.parse_args()
    log = run(42)
    Path(a.out).write_text(fmt(log), encoding="utf-8")
    print("seed 42 actions:", dict(collections.Counter(x[2] for x in log)), "->", a.out)
    ref = ROOT / "results" / "simulation_log.txt"
    if ref.exists():
        print("identical to results/simulation_log.txt:", ref.read_text(encoding="utf-8") == fmt(log))
    if a.n_runs:
        runs = [run(s) for s in range(a.n_runs)]
        cnt = collections.Counter(x[2] for r in runs for x in r); tot = sum(cnt.values())
        none = np.array([sum(x[2] == "none" for x in r) for r in runs])
        s0 = np.array([r[0][1] for r in runs]); sN = np.array([r[-1][3] for r in runs])
        print(f"{a.n_runs} runs x 30 steps; action shares:", {str(k): round(v / tot, 3) for k, v in sorted(cnt.items())})
        print(f"no-action steps per run: mean {none.mean():.1f}, 5-95%: {np.percentile(none, [5, 95]).tolist()}")
        print("mean initial state", s0.mean(0).round(2), "| mean final state", sN.mean(0).round(2))
        print(f"arousal lower at the end than at the start in {100 * np.mean(sN[:, 1] < s0[:, 1]):.1f}% of runs; "
              f"valence higher in {100 * np.mean(sN[:, 0] > s0[:, 0]):.1f}%")
