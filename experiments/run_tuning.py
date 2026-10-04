"""Optuna tuning experiment (E3, Track B models). Results -> results/tuning/, figure -> results/figures/.

    python experiments/run_tuning.py              # 30 trials per outer fold (~25 min)
    python experiments/run_tuning.py --quick      # 8 trials, for a fast check
    python experiments/run_tuning.py gb rf        # only some models

For each model and subject: untuned defaults under LODO vs Optuna-tuned under nested LODO.
"""
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import SUBJECTS
from evaluation import NOISE_FLOOR, evaluate
from tuning import BUILDERS, DEFAULTS, optuna_model

RES = ROOT / "results" / "tuning"
FIG = ROOT / "results" / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
N_TRIALS = 8 if QUICK else 30
FEATURES = "all"

NAMES = {"nn_paper": "NN (1 hidden, paper-style)", "nn_extended": "MLP (1-3 layers)",
         "rf": "Random Forest", "et": "Extra Trees", "gb": "Gradient Boosting"}


def main():
    chosen = [a for a in sys.argv[1:] if a in BUILDERS] or list(BUILDERS)
    rows, params = [], []
    for name in chosen:
        for s in SUBJECTS:
            t = time.time()
            base = evaluate(lambda: BUILDERS[name](DEFAULTS[name]), s, FEATURES, "lodo").summary
            tuned = evaluate(optuna_model(name, n_trials=N_TRIALS), s, FEATURES, "nested")
            ts = tuned.summary
            rows.append({"model": NAMES[name], "key": name, "subject": s, "features": FEATURES,
                         "default_lodo_mse": base["mse"], "tuned_nested_mse": ts["mse"],
                         "improvement_pct": 100 * (1 - ts["mse"] / base["mse"]),
                         "tuned_r2": ts["r2"], "tuned_skill": ts["skill"],
                         "spearman_day": ts["spearman_day"], "top5_in_top10": ts["top5_in_top10"],
                         "n_trials": N_TRIALS})
            for d, p in zip(tuned.folds["test_day"], tuned.extra["best_params"]):
                params.append({"model": NAMES[name], "subject": s, "test_day": d, **p})
            print(f"  {name:12s} {s}  default={base['mse']:.4f}  tuned={ts['mse']:.4f}  ({time.time() - t:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "optuna_leaderboard.csv", index=False)
    pd.DataFrame(params).to_csv(RES / "optuna_best_params.csv", index=False)
    print(df[["model", "subject", "default_lodo_mse", "tuned_nested_mse", "improvement_pct", "spearman_day"]]
          .round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, s in zip(axes, SUBJECTS):
        d = df[df.subject == s]
        y = np.arange(len(d))
        ax.barh(y - 0.2, d.default_lodo_mse, 0.4, color="#adb5bd", label="default settings")
        ax.barh(y + 0.2, d.tuned_nested_mse, 0.4, color="#2a6fdb", label=f"Optuna-tuned ({N_TRIALS} trials, nested)")
        ax.set_yticks(y, d.model)
        ax.axvline(NOISE_FLOOR, ls="--", c="k", lw=1)
        ax.set_title(f"Subject {s[-1]}: unseen-day MSE")
        ax.set_xlabel("MSE")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e3_optuna_tuning.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
