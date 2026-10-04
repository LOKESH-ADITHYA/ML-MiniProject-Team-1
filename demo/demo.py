"""Live demo: predict an unseen day's metabolic cost from muscle + step data.

    python demo/demo.py                              # Subject 1, day 5 held out
    python demo/demo.py --subject S2 --day 3 --features step+emg

The models are trained on the OTHER four days only, then asked to predict the
held-out day (this is the honest leave-one-day-out test).  Takes about 10 seconds.
Outputs a results table, results/figures/demo_prediction.png, and (if present)
the cached leaderboard from the full experiments.
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402

from ensemble import make_ridge, voting_factory  # noqa: E402
from interpret import NOISE_FLOOR, ranking_metrics  # noqa: E402
from loader_b import load_subject_b  # noqa: E402
from models_nn import paper_nn_factory  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subject", default="S1", choices=["S1", "S2"])
ap.add_argument("--day", type=int, default=5, help="held-out day, 1-5")
ap.add_argument("--features", default="all", choices=["all", "step+emg", "step", "emg"])
args = ap.parse_args()

X, y, day = load_subject_b(args.subject, args.features)
test = day == args.day - 1
if not test.any():
    sys.exit("no rows for that day")
print(f"Subject {args.subject[-1]} | features: {args.features} ({X.shape[1]}) | "
      f"train rows: {int((~test).sum())} (4 days) | unseen day {args.day}: {int(test.sum())} blocks\n")

models = {"mean predictor": DummyRegressor, "ridge": make_ridge,
          "tanh network": paper_nn_factory(5, n_seeds=5), "ensemble (average)": voting_factory()}
preds, rows = {}, []
for name, make in models.items():
    p = make().fit(X.values[~test], y[~test]).predict(X.values[test])
    preds[name] = p
    mse = float(np.mean((p - y[test]) ** 2))
    rk = ranking_metrics(y[test], p)
    rows.append({"model": name, "MSE": round(mse, 4), "RMSE": round(mse ** 0.5, 3),
                 "within-day rank corr": round(rk["spearman"], 2) if name != "mean predictor" else "-",
                 "best-5 found in top-10": rk["best5_in_pred_top10"] if name != "mean predictor" else "-"})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n(noise floor MSE = {NOISE_FLOOR};  rank corr 1.0 = perfect ordering of the 36 control settings)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(y[test], "k-o", ms=3, label="actual")
for name in ["ridge", "tanh network", "ensemble (average)"]:
    axes[0].plot(preds[name], "--", label=name)
axes[0].set_xlabel("control setting (2-minute block) within the day")
axes[0].set_ylabel("normalised metabolic cost")
axes[0].set_title(f"Subject {args.subject[-1]}, unseen day {args.day}")
axes[0].legend(fontsize=7)
axes[1].scatter(y[test], preds["ensemble (average)"], s=14)
lim = [min(y[test].min(), preds["ensemble (average)"].min()), max(y[test].max(), preds["ensemble (average)"].max())]
axes[1].plot(lim, lim, "k:", lw=1)
axes[1].set_xlabel("actual")
axes[1].set_ylabel("predicted (ensemble)")
fig.tight_layout()
out = ROOT / "results" / "figures" / "demo_prediction.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nplot saved to {out}")

lb = ROOT / "results" / "track_b" / "leaderboard_lodo.csv"
if lb.exists():
    print("\nCached results from the full experiments (best MSE per model, leave-one-day-out):")
    print(pd.read_csv(lb).to_string(index=False))