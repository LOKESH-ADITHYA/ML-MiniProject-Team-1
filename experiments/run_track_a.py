"""Track A experiments. Writes CSVs to results/track_a/ and figures to results/figures/.

    python experiments/run_track_a.py            # everything (~10 min on a laptop)
    python experiments/run_track_a.py e1 e5      # only some experiments
    python experiments/run_track_a.py --quick    # 1 repeat for k-fold, fewer k values

E1a  Lasso half of Table I (P1, 10-fold and 5-fold as in the authors' code) vs paper
E2   classical models: paper-style k-fold (P1) vs leave-one-day-out (P2)
E3a  classical models tuned with nested LODO (P3) + ranking metric, and forward-chaining (P5)
E5   feature reduction: PCA / forward stepwise vs PLS, MI, RFE, stability selection, ARD
E6   feature-engineering ablation
E7a  cross-subject test (P4) and time-only baseline
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import RidgeCV

from data import FEATURE_SETS, SUBJECTS, load_subject
from evaluation import NOISE_FLOOR, cross_subject, evaluate, run_grid
from features import (SELECTION_METHODS, day_center, engineer, make_selector, selected_features,
                      with_interactions)
from models_classical import MODELS, TUNED_MODELS
from data import get_xy

RES = ROOT / "results" / "track_a"
FIG = ROOT / "results" / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
REPEATS = 1 if QUICK else 5
COL = {"S1": "#2a6fdb", "S2": "#d9480f"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

PAPER_TABLE1 = {  # Table I, Krimsky & Ng: constant + linear-regression (Lasso) columns
    "S1": {"Mean": 0.0657, "all": 0.0089, "step+emg": 0.0089, "step": 0.0200, "emg": 0.0158},
    "S2": {"Mean": 0.0465, "all": 0.0176, "step+emg": 0.0168, "step": 0.0217, "emg": 0.0203},
}


def flat(p):
    """Pivot table -> flat columns like S1_lodo."""
    p.columns = ["_".join(map(str, c)) if isinstance(c, tuple) else c for c in p.columns]
    return p.reset_index()


def save(df, name):
    df.to_csv(RES / f"{name}.csv", index=False)
    print(df.round(4).to_string(index=False), "\n")


# ------------------------------------------------------------------------------ E1a
def e1():
    rows = []
    for s in SUBJECTS:
        for k in (10, 5):
            m = evaluate(MODELS["Mean"], s, "all", "kfold", n_splits=k, n_repeats=REPEATS).summary
            rows.append({"subject": s, "folds": k, "features": "Mean", "mse": m["mse"],
                         "mse_std": m["mse_std"], "paper": PAPER_TABLE1[s]["Mean"]})
            for fs in FEATURE_SETS:
                r = evaluate(MODELS["Lasso"], s, fs, "kfold", n_splits=k, n_repeats=REPEATS).summary
                rows.append({"subject": s, "folds": k, "features": fs, "mse": r["mse"],
                             "mse_std": r["mse_std"], "paper": PAPER_TABLE1[s][fs]})
    df = pd.DataFrame(rows)
    save(df, "e1_table1_lasso")

    d = df[df.folds == 10]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, s in zip(axes, SUBJECTS):
        x = d[d.subject == s]
        i = np.arange(len(x))
        ax.bar(i - 0.2, x.mse, 0.4, yerr=x.mse_std, color=COL[s], label="ours (10-fold x5)")
        ax.bar(i + 0.2, x.paper, 0.4, color="#adb5bd", label="paper")
        ax.set_xticks(i, x.features)
        ax.axhline(NOISE_FLOOR, ls="--", c="k", lw=1)
        ax.set_title(f"Subject {s[-1]}: Lasso vs paper Table I")
        ax.set_ylabel("MSE")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e1_table1_lasso.png")
    plt.close(fig)


# ------------------------------------------------------------------------------ E2
def e2():
    df = run_grid(MODELS, feature_sets=("all", "step+emg"), protocols=("kfold", "lodo"), n_repeats=REPEATS)
    save(df[["model", "subject", "features", "protocol", "mse", "mse_std", "r2", "mean_mse", "skill"]], "e2_kfold_vs_lodo")

    d = df[df.features == "all"].pivot_table(index="model", columns=["subject", "protocol"], values="mse")
    d = d.loc[list(MODELS)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, s in zip(axes, SUBJECTS):
        y = np.arange(len(d))
        ax.barh(y - 0.2, d[(s, "kfold")], 0.4, color="#2a6fdb", label="random 10-fold (P1)")
        ax.barh(y + 0.2, d[(s, "lodo")], 0.4, color="#f59f00", label="leave-one-day-out (P2)")
        ax.set_yticks(y, d.index)
        ax.axvline(NOISE_FLOOR, ls="--", c="k", lw=1)
        ax.set_title(f"Subject {s[-1]} (all features)")
        ax.set_xlabel("MSE")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e2_kfold_vs_lodo.png")
    plt.close(fig)


# ------------------------------------------------------------------------------ E3a
def e3():
    rows, params = [], []
    for name, make in TUNED_MODELS.items():
        for s in SUBJECTS:
            for fs in ("all", "step+emg"):
                r = evaluate(make, s, fs, "nested", name=name)
                rows.append(r.summary)
                for f, p in zip(r.folds.get("test_day", []), r.extra["best_params"]):
                    params.append({"model": name, "subject": s, "features": fs, "test_day": f, "params": str(p)})
        print(f"  nested {name} done")
    df = pd.DataFrame(rows)
    save(df[["model", "subject", "features", "mse", "mse_std", "r2", "skill", "spearman_day", "top5_in_top10"]],
         "e3_nested_lodo_leaderboard")
    pd.DataFrame(params).to_csv(RES / "e3_nested_best_params.csv", index=False)

    fwd = run_grid({k: MODELS[k] for k in ["Mean", "Lasso", "Ridge", "BayesianRidge", "PLS"]},
                   feature_sets=("all",), protocols=("forward",))
    save(fwd[["model", "subject", "mse", "skill", "spearman_day"]], "e3_forward_chaining")


# ------------------------------------------------------------------------------ E5
def e5():
    ks = [3, 5, 10, 15, 20] if QUICK else [2, 3, 5, 8, 10, 12, 15, 20, 24]
    rows = []
    for s in SUBJECTS:
        for prot in ("lodo", "kfold"):
            reps = {"n_repeats": 2 if not QUICK else 1} if prot == "kfold" else {}
            base = evaluate(make_selector("none"), s, "step+emg", prot, **reps).summary
            rows.append({"subject": s, "protocol": prot, "method": "none (all 25)", "k": 25, "mse": base["mse"]})
            for m in SELECTION_METHODS:
                for k in ks:
                    if m == "pls" and k > 15:
                        continue
                    r = evaluate(make_selector(m, k), s, "step+emg", prot, **reps).summary
                    rows.append({"subject": s, "protocol": prot, "method": m, "k": k, "mse": r["mse"]})
            for t in (0.5, 0.6, 0.7, 0.8, 0.9):
                r = evaluate(make_selector("stability", threshold=t), s, "step+emg", prot, **reps).summary
                rows.append({"subject": s, "protocol": prot, "method": "stability", "k": t, "mse": r["mse"]})
            r = evaluate(make_selector("ard"), s, "step+emg", prot, **reps).summary
            rows.append({"subject": s, "protocol": prot, "method": "ard", "k": np.nan, "mse": r["mse"]})
            print(f"  selection {s} {prot} done")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e5_feature_selection_curves.csv", index=False)

    best = df.loc[df.groupby(["subject", "protocol", "method"]).mse.idxmin()]
    save(flat(best.pivot_table(index="method", columns=["subject", "protocol"], values="mse")),
         "e5_feature_selection_best")

    picked = []
    for s in SUBJECTS:
        X, y, _ = get_xy(load_subject(s), "step+emg")
        for m in ["forward", "mi", "rfe", "stability", "ard"]:
            picked.append({"subject": s, "method": m, "k": 10 if m in SELECTION_METHODS else None,
                           "features": ", ".join(selected_features(m, X, y, k=10) or [])})
    save(pd.DataFrame(picked), "e5_selected_features")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, s in zip(axes, SUBJECTS):
        d = df[(df.subject == s) & (df.protocol == "lodo") & df.method.isin(SELECTION_METHODS)]
        for m, g in d.groupby("method"):
            ax.plot(g.k, g.mse, "o-", ms=3, label=m)
        ax.axhline(df[(df.subject == s) & (df.protocol == "lodo") & (df.method == "none (all 25)")].mse.iloc[0],
                   c="k", lw=1, ls=":", label="all 25 features")
        ax.set_title(f"Subject {s[-1]}: LODO MSE vs features kept")
        ax.set_xlabel("k (features / components)")
        ax.set_ylabel("MSE (Ridge on selected)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "e5_feature_selection.png")
    plt.close(fig)


# ------------------------------------------------------------------------------ E6
def e6():
    variants = {
        "base (step+emg)": (),
        "+symmetry": ("symmetry",),
        "+emg_bilateral": ("emg_bilateral",),
        "+emg_summary": ("emg_summary",),
        "+all three": ("symmetry", "emg_bilateral", "emg_summary"),
    }
    models = {"Ridge": MODELS["Ridge"], "Lasso": MODELS["Lasso"], "BayesianRidge": MODELS["BayesianRidge"]}
    rows = []
    for s in SUBJECTS:
        raw = load_subject(s)
        for vname, groups in variants.items():
            df, cols = engineer(raw, groups)
            for mname, make in models.items():
                for prot in ("lodo", "kfold"):
                    r = evaluate(make, s, cols, prot, df=df, n_repeats=REPEATS).summary
                    rows.append({"subject": s, "variant": vname, "n_features": len(cols), "model": mname,
                                 "protocol": prot, "mse": r["mse"]})
        for k in (3, 5, 8):  # 5(d) pairwise interactions among the top-k features (picked per fold)
            for prot in ("lodo", "kfold"):
                r = evaluate(with_interactions(k), s, "step+emg", prot, n_repeats=REPEATS).summary
                rows.append({"subject": s, "variant": f"+interactions (top {k})", "n_features": 25 + k * (k - 1) // 2,
                             "model": "Ridge", "protocol": prot, "mse": r["mse"]})
        for mname, make in models.items():  # per-day centring of features (label-free)
            for prot in ("lodo", "kfold"):
                r = evaluate(make, s, "step+emg", prot, transform=day_center, n_repeats=REPEATS).summary
                rows.append({"subject": s, "variant": "day-centred features", "n_features": 25, "model": mname,
                             "protocol": prot, "mse": r["mse"]})
    df = pd.DataFrame(rows)
    save(flat(df.pivot_table(index=["variant", "model"], columns=["subject", "protocol"], values="mse")),
         "e6_feature_engineering_ablation")
    df.to_csv(RES / "e6_feature_engineering_long.csv", index=False)

    d = df[(df.model == "Ridge") & (df.protocol == "lodo")].pivot(index="variant", columns="subject", values="mse")
    d = d.loc[list(variants) + [f"+interactions (top {k})" for k in (3, 5, 8)] + ["day-centred features"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(d))
    for j, s in enumerate(SUBJECTS):
        ax.barh(y + (j - 0.5) * 0.4, d[s], 0.4, color=COL[s], label=f"Subject {s[-1]}")
    ax.set_yticks(y, d.index)
    ax.invert_yaxis()
    ax.set_xlabel("LODO MSE (Ridge)")
    ax.set_title("Feature engineering ablation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e6_feature_engineering.png")
    plt.close(fig)


# ------------------------------------------------------------------------------ E7a
def time_only(col):
    """Predict metabolic cost from position in time only (no sensors)."""
    return lambda: make_pipeline(StandardScaler(), SplineTransformer(n_knots=6, degree=3),
                                 RidgeCV(alphas=np.logspace(-3, 3, 13)))


def e7():
    rows = []
    for tr, te in [("S1", "S2"), ("S2", "S1")]:
        for fs in ("step+emg", "emg", "step"):
            for name in ["Mean", "Lasso", "Ridge", "BayesianRidge", "PLS", "SVR", "GP"]:
                r = cross_subject(MODELS[name], fs, tr, te)
                rows.append({"model": name, **r})
    df = pd.DataFrame(rows)
    save(df[["train", "test", "features", "model", "mse", "mean_mse", "skill", "spearman_day"]], "e7_cross_subject")

    rows = []
    for s in SUBJECTS:
        for feat, label in [(["row_in_day"], "position within day"), (["row"], "global time index")]:
            for prot in ("kfold", "lodo"):
                r = evaluate(time_only(feat), s, feat, prot, n_repeats=REPEATS).summary
                rows.append({"subject": s, "time_feature": label, "protocol": prot, "mse": r["mse"],
                             "mean_mse": r["mean_mse"], "skill": r["skill"]})
        for prot in ("kfold", "lodo"):
            r = evaluate(MODELS["Ridge"], s, "all", prot, n_repeats=REPEATS).summary
            rows.append({"subject": s, "time_feature": "(reference) Ridge on all sensors", "protocol": prot,
                         "mse": r["mse"], "mean_mse": r["mean_mse"], "skill": r["skill"]})
    tb = pd.DataFrame(rows)
    save(tb, "e7_time_only_baseline")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, s in zip(axes, SUBJECTS):
        d = tb[tb.subject == s].pivot(index="time_feature", columns="protocol", values="mse")
        y = np.arange(len(d))
        ax.barh(y - 0.2, d["kfold"], 0.4, color="#2a6fdb", label="random 10-fold")
        ax.barh(y + 0.2, d["lodo"], 0.4, color="#f59f00", label="leave-one-day-out")
        ax.set_yticks(y, d.index)
        ax.set_title(f"Subject {s[-1]}: how far does time alone get?")
        ax.set_xlabel("MSE")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e7_time_only_baseline.png")
    plt.close(fig)


EXPERIMENTS = {"e1": e1, "e2": e2, "e3": e3, "e5": e5, "e6": e6, "e7": e7}

if __name__ == "__main__":
    chosen = [a for a in sys.argv[1:] if a in EXPERIMENTS] or list(EXPERIMENTS)
    for name in chosen:
        t = time.time()
        print(f"===== {name.upper()} =====")
        EXPERIMENTS[name]()
        print(f"[{name} took {time.time() - t:.0f}s]\n")
