"""Track B experiment b6b: fixing the "tuning made Subject 2 worse" problem.

Compares, for our seed-averaged network, on UNSEEN days:
    default      untuned network (5 tanh neurons), leave-one-day-out
    wide         b6 setting: wide search space, best single trial       (nested LODO)
    small        smaller regularised space, best single trial           (nested LODO)
    small_top5   smaller regularised space, average of the best 5 trials (nested LODO)

    python experiments/run_tuning_b2.py           # 20 trials per outer fold (~10 min)
    python experiments/run_tuning_b2.py --quick   # 5 trials

Output: results/track_b/b6b_tuned_nn_robust.csv
"""
import sys
import time
import warnings
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from evaluation import evaluate  # noqa: E402
from models_nn import SeedAveragedMLP, paper_nn_factory  # noqa: E402
from tuning import OptunaTuned  # noqa: E402
from tuning_b import TopKOptunaTuned  # noqa: E402

OUT = ROOT / "results" / "track_b"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
N_TRIALS = 5 if QUICK else 20


def space_wide(t):
    return {"neurons": t.suggest_int("neurons", 1, 10),
            "alpha": t.suggest_float("alpha", 0.1, 100.0, log=True),
            "activation": t.suggest_categorical("activation", ["tanh", "relu"])}


def space_small(t):  # fewer free choices, always some regularisation
    return {"neurons": t.suggest_int("neurons", 1, 5),
            "alpha": t.suggest_float("alpha", 1.0, 100.0, log=True),
            "activation": t.suggest_categorical("activation", ["tanh"])}


def build(p):
    return make_pipeline(StandardScaler(), SeedAveragedMLP(
        hidden_layer_sizes=(p["neurons"],), alpha=p["alpha"],
        activation=p["activation"], n_seeds=5))


VARIANTS = {
    "wide": partial(OptunaTuned, build, space_wide, N_TRIALS),
    "small": partial(OptunaTuned, build, space_small, N_TRIALS),
    "small_top5": partial(TopKOptunaTuned, build, space_small, N_TRIALS, 0, 5),
}

rows = []
for subject in ["S1", "S2"]:
    for fs in ["all", "step+emg"]:
        t0 = time.time()
        default = float(evaluate(paper_nn_factory(5, n_seeds=5), subject=subject, feature_set=fs,
                                 protocol="lodo").summary["mse"])
        row = {"experiment": "b6b", "subject": subject, "feature_set": fs, "default_lodo_mse": default,
               "n_trials": N_TRIALS}
        for name, make in VARIANTS.items():
            mse = float(evaluate(make, subject=subject, feature_set=fs, protocol="nested").summary["mse"])
            row[f"{name}_mse"] = mse
            row[f"{name}_vs_default_pct"] = 100 * (1 - mse / default)
        row["seconds"] = round(time.time() - t0)
        rows.append(row)
        print(f"  {subject} {fs:<9} default={default:.4f} | " + " | ".join(
            f"{n}={row[n + '_mse']:.4f} ({row[n + '_vs_default_pct']:+.0f}%)" for n in VARIANTS))
pd.DataFrame(rows).to_csv(OUT / "b6b_tuned_nn_robust.csv", index=False)
print(f"-> saved {OUT / 'b6b_tuned_nn_robust.csv'}")