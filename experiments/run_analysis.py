"""Track B experiment b7: interpretation and diagnostics on unseen days.

    python experiments/run_analysis.py            # ~8 min
    python experiments/run_analysis.py --quick


"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from functools import partial  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402

import interpret as ip  # noqa: E402
from ensemble import make_ridge, voting_factory  # noqa: E402
from loader_b import load_subject_b  # noqa: E402
from models_nn import paper_nn_factory  # noqa: E402
from models_trees import make_extra_trees  # noqa: E402

OUT = ROOT / "results" / "track_b"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
SEEDS = 3 if QUICK else 10
REPEATS = 5 if QUICK else 20

NN = paper_nn_factory(5, n_seeds=SEEDS)
ET = partial(make_extra_trees, n_estimators=100 if QUICK else 300)
MODELS = {"mean predictor": DummyRegressor, "ridge": make_ridge, "tanh net (5)": NN,
          "extra trees": ET, "ensemble (average)": voting_factory()}


def importance():
    for model_name, make in [("nn", NN), ("et", ET)]:
        tables = {}
        for s in ["S1", "S2"]:
            X, y, day = load_subject_b(s, "all")
            t = ip.permutation_importance_lodo(make, X, y, day, n_repeats=REPEATS)
            t.to_csv(OUT / f"b7_perm_importance_{s}_{model_name}.csv", index=False)
            tables[f"Subject {s[-1]} - {'tanh net' if model_name == 'nn' else 'extra trees'}"] = t
            print(f"  importance {s} {model_name}: top = {list(t['feature'][:4])}")
        ip.plot_importance(tables, FIG / f"b7_perm_importance_{model_name}.png")


def learning_curves():
    rows = []
    for s in ["S1", "S2"]:
        X, y, day = load_subject_b(s, "all")
        for name in ["mean predictor", "ridge", "tanh net (5)", "extra trees"]:
            lc = ip.learning_curve_days(MODELS[name], X, y, day)
            lc["subject"], lc["model"] = s, name
            rows.append(lc)
            print(f"  learning curve {s} {name}: {lc['mse'].round(4).tolist()}")
    df = pd.concat(rows)
    df.to_csv(OUT / "b7_learning_curve.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, s in zip(axes, ["S1", "S2"]):
        for name, g in df[df.subject == s].groupby("model"):
            ax.plot(g["n_train_days"], g["mse"], "o-", label=name)
        ax.axhline(ip.NOISE_FLOOR, color="k", ls=":", lw=1, label="noise floor")
        ax.set_title(f"Subject {s[-1]}: error on an unseen day")
        ax.set_xlabel("number of training days")
        ax.set_xticks([1, 2, 3, 4])
    axes[0].set_ylabel("MSE")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "b7_learning_curve.png", dpi=150)
    plt.close(fig)


def per_day():
    rows = []
    for s in ["S1", "S2"]:
        X, y, day = load_subject_b(s, "all")
        for name, make in MODELS.items():
            d = ip.per_day_mse(make, X, y, day)
            d["subject"], d["model"] = s, name
            rows.append(d)
    df = pd.concat(rows)
    df.to_csv(OUT / "b7_per_day_mse.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, s in zip(axes, ["S1", "S2"]):
        p = df[df.subject == s].pivot(index="test_day", columns="model", values="mse")
        p.plot.bar(ax=ax, rot=0, width=0.8)
        ax.set_title(f"Subject {s[-1]}: MSE on each held-out day")
        ax.set_xlabel("held-out day")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(FIG / "b7_per_day_mse.png", dpi=150)
    plt.close(fig)
    print(df.pivot_table(index=["subject", "test_day"], columns="model", values="mse").round(4))


def shap_plots():
    for s in ["S1", "S2"]:
        X, y, _ = load_subject_b(s, "all")
        model = ET().fit(X.values, y)
        ok = ip.shap_tree_summary(model, X, FIG / f"b7_shap_{s}.png",
                                  f"Subject {s[-1]}: SHAP (extra trees, all days)")
        print(f"  shap {s}: {'saved' if ok else 'skipped (shap not available)'}")


def noise_floor_table():
    files = sorted(OUT.glob("b[1-6]_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["protocol"].isin(["kfold", "lodo"]) & df["mse"].notna()]
    best = (df.loc[df.groupby(["subject", "protocol"])["mse"].idxmin()]
            [["subject", "protocol", "model", "feature_set", "mse"]].copy())
    best["noise_floor"] = ip.NOISE_FLOOR
    best["times_noise_floor"] = (best["mse"] / ip.NOISE_FLOOR).round(1)
    best.to_csv(OUT / "b7_noise_floor_table.csv", index=False)
    print(best.round(4).to_string(index=False))


if __name__ == "__main__":
    for step in (importance, learning_curves, per_day, shap_plots, noise_floor_table):
        print(f"\n== {step.__name__} ==")
        step()