"""Track B experiments (Person B).

    python experiments/run_track_b.py                # everything (~10 min)
    python experiments/run_track_b.py b1 b3          # only some experiments
    python experiments/run_track_b.py --quick        # fast smoke test

b1  paper-style NN, 1..10 neurons, 4 feature sets, both subjects, protocol P1 (kfold)
b2  the same under P2 (leave-one-day-out)
b3  Random Forest / Extra Trees / Gradient Boosting, kfold + lodo
b4  extended MLP (2 hidden layers) vs paper-style NN, kfold + lodo

Results go to results/track_b/*.csv and figures to results/figures/*.png.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from evaluation import evaluate  # noqa: E402  (Person A's harness)
from models_nn import extended_mlp_factory, paper_nn_factory  # noqa: E402
from models_trees import TREE_MODELS, tree_factory  # noqa: E402
import analysis  # noqa: E402

OUT = ROOT / "results" / "track_b"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["S1", "S2"]
FEATURE_SETS = ["all", "step+emg", "step", "emg"]


def summary_dict(res):
    """Turn whatever evaluate() put in res.summary into a flat dict of floats."""
    s = res.summary
    if isinstance(s, pd.DataFrame):
        s = s.iloc[0] if len(s) == 1 else s.mean(numeric_only=True)
    s = dict(s)
    out = {}
    for k, v in s.items():
        try:
            out[str(k).lower()] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def run_one(rows, experiment, model_name, make_model, subject, fs, protocol, **extra):
    t0 = time.time()
    res = evaluate(make_model, subject=subject, feature_set=fs, protocol=protocol)
    row = {"experiment": experiment, "model": model_name, "subject": subject,
           "feature_set": fs, "protocol": protocol, **extra, **summary_dict(res),
           "seconds": round(time.time() - t0, 1)}
    rows.append(row)
    print(f"  {experiment} {model_name:<18} {subject} {fs:<9} {protocol:<6} "
          f"{extra}  mse={row.get('mse', float('nan')):.4f}")


def save(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)
    print(f"  -> saved {OUT / name}")


def nn_sweep(experiment, protocol, neurons, feature_sets, n_seeds):
    rows = []
    for subject in SUBJECTS:
        for fs in feature_sets:
            for n in neurons:
                run_one(rows, experiment, "paper_nn", paper_nn_factory(n, n_seeds=n_seeds),
                        subject, fs, protocol, n_hidden=n, n_seeds=n_seeds)
    return rows


def b1(quick):
    neurons = [1, 3, 5] if quick else list(range(1, 11))
    fsets = ["all", "emg"] if quick else FEATURE_SETS
    save(nn_sweep("b1", "kfold", neurons, fsets, 3 if quick else 10), "b1_nn_sweep_kfold.csv")


def b2(quick):
    neurons = [1, 3, 5] if quick else list(range(1, 11))
    fsets = ["all", "emg"] if quick else FEATURE_SETS
    save(nn_sweep("b2", "lodo", neurons, fsets, 3 if quick else 10), "b2_nn_sweep_lodo.csv")


def b3(quick):
    rows = []
    fsets = ["all", "emg"] if quick else FEATURE_SETS
    for protocol in ["kfold", "lodo"]:
        for subject in SUBJECTS:
            for fs in fsets:
                for name in TREE_MODELS:
                    kw = {"n_estimators": 50} if quick else {}
                    run_one(rows, "b3", name, tree_factory(name, **kw), subject, fs, protocol)
    save(rows, "b3_trees.csv")


def b4(quick):
    rows = []
    fsets = ["all", "emg"] if quick else FEATURE_SETS
    seeds = 3 if quick else 10
    for protocol in ["kfold", "lodo"]:
        for subject in SUBJECTS:
            for fs in fsets:
                run_one(rows, "b4", "extended_mlp", extended_mlp_factory(n_seeds=seeds),
                        subject, fs, protocol, n_seeds=seeds)
    save(rows, "b4_extended_mlp.csv")


def report():
    """Rebuild figures and tables from whatever CSVs exist."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    files = sorted(OUT.glob("b[0-9]_*.csv"))
    if not files:
        return
    df = pd.concat([analysis.load_results(f) for f in files], ignore_index=True)
    for protocol in ["kfold", "lodo"]:
        if ((df["protocol"] == protocol) & (df["model"] == "paper_nn")).any():
            analysis.plot_neuron_sweep(df, protocol, FIG / f"b_neuron_sweep_{protocol}.png")
            t = analysis.best_nn_vs_paper(df, protocol)
            t.to_csv(OUT / f"table1_vs_paper_{protocol}.csv", index=False)
            print(f"\nOur best NN vs paper ({protocol}):\n{t.to_string(index=False)}")
        if (df["protocol"] == protocol).any():
            lb = analysis.leaderboard(df, protocol)
            lb.to_csv(OUT / f"leaderboard_{protocol}.csv")
            print(f"\nLeaderboard ({protocol}, best MSE per model):\n{lb}")


EXPERIMENTS = {"b1": b1, "b2": b2, "b3": b3, "b4": b4}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    todo = args or list(EXPERIMENTS)
    for name in todo:
        if name not in EXPERIMENTS:
            sys.exit(f"unknown experiment {name!r}; choose from {list(EXPERIMENTS)}")
    for name in todo:
        print(f"\n== {name} {'(quick)' if quick else ''} ==")
        EXPERIMENTS[name](quick)
    report()