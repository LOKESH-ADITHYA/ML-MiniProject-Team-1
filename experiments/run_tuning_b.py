"""Track B experiment b6: tune OUR seed-averaged tanh/relu network with Optuna (nested LODO).

Reuses the shared tuner (src/tuning.py, OptunaTuned) with a search space that matches
models_nn.SeedAveragedMLP: neurons, L2 strength (alpha) and activation.

    python experiments/run_tuning_b.py            # 30 trials per outer fold (~10 min)
    python experiments/run_tuning_b.py --quick    # 6 trials

Output: results/track_b/b6_tuned_nn.csv   (default network under LODO vs tuned under nested LODO)
"""
import sys
import time
import warnings
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

OUT = ROOT / "results" / "track_b"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
N_TRIALS = 6 if QUICK else 30


def space(t):
    return {"neurons": t.suggest_int("neurons", 1, 10),
            "alpha": t.suggest_float("alpha", 0.1, 100.0, log=True),
            "activation": t.suggest_categorical("activation", ["tanh", "relu"])}


def build(p):
    return make_pipeline(StandardScaler(), SeedAveragedMLP(
        hidden_layer_sizes=(p["neurons"],), alpha=p["alpha"],
        activation=p["activation"], n_seeds=5))


def tuned_factory():
    return OptunaTuned(build, space, n_trials=N_TRIALS)


rows = []
for subject in ["S1", "S2"]:
    for fs in ["all", "step+emg"]:
        t0 = time.time()
        base = evaluate(paper_nn_factory(5, n_seeds=5), subject=subject, feature_set=fs, protocol="lodo")
        tuned = evaluate(tuned_factory, subject=subject, feature_set=fs, protocol="nested")
        b, t = base.summary, tuned.summary
        row = {"experiment": "b6", "model": "paper_nn_tuned", "subject": subject, "feature_set": fs,
               "default_lodo_mse": float(b["mse"]), "tuned_nested_mse": float(t["mse"]),
               "improvement_pct": 100 * (1 - float(t["mse"]) / float(b["mse"])),
               "n_trials": N_TRIALS, "seconds": round(time.time() - t0)}
        for k in ("r2", "skill", "spearman_day", "top5_in_top10"):
            if k in t:
                row["tuned_" + k] = float(t[k])
        rows.append(row)
        print(f"  b6 {subject} {fs:<9} default={row['default_lodo_mse']:.4f} "
              f"tuned={row['tuned_nested_mse']:.4f} ({row['improvement_pct']:+.1f}%)")
pd.DataFrame(rows).to_csv(OUT / "b6_tuned_nn.csv", index=False)
print(f"-> saved {OUT / 'b6_tuned_nn.csv'}")