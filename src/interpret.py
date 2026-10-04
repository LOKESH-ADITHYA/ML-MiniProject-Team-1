"""Track B: interpretation and diagnostics (all evaluated on UNSEEN days).

    permutation_importance_lodo   which features matter (EMG channel ranking)
    learning_curve_days           error vs number of training days
    per_day_mse                   error on each held-out day
    ranking_metrics               within-day ranking quality (HILO-relevant)
    shap_tree_summary             optional SHAP bar chart for a tree model
"""
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from loader_b import feature_group

GROUP_COLOR = {"step": "tab:green", "emg": "tab:orange", "ctrl": "tab:purple"}
NOISE_FLOOR = 0.0025


def _mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def permutation_importance_lodo(make_model, X, y, day, n_repeats=20, seed=0):
    """Fit on 4 days, shuffle each feature on the 5th day, average the MSE increase."""
    names = list(X.columns)
    Xv = X.values
    per_day = []
    for d in np.unique(day):
        tr, te = day != d, day == d
        model = make_model().fit(Xv[tr], y[tr])
        r = permutation_importance(model, Xv[te], y[te], scoring="neg_mean_squared_error",
                                   n_repeats=n_repeats, random_state=seed)
        per_day.append(r.importances_mean)
    per_day = np.array(per_day)
    return (pd.DataFrame({"feature": names, "importance": per_day.mean(0),
                          "std_over_days": per_day.std(0),
                          "group": [feature_group(n) for n in names]})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def plot_importance(tables, out_path, top=12):
    """tables: {panel title: DataFrame from permutation_importance_lodo}."""
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (title, t) in zip(axes, tables.items()):
        t = t.head(top).iloc[::-1]
        ax.barh(t["feature"], t["importance"], xerr=t["std_over_days"],
                color=[GROUP_COLOR[g] for g in t["group"]], alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("MSE increase when shuffled (unseen day)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
    axes[-1].legend(handles, GROUP_COLOR.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def learning_curve_days(make_model, X, y, day):
    """Mean unseen-day MSE as a function of how many training days are used."""
    Xv = X.values
    days = np.unique(day)
    rows = []
    for test_day in days:
        others = [d for d in days if d != test_day]
        te = day == test_day
        for k in range(1, len(others) + 1):
            for combo in itertools.combinations(others, k):
                tr = np.isin(day, combo)
                m = make_model().fit(Xv[tr], y[tr])
                rows.append({"test_day": int(test_day), "n_train_days": k,
                             "mse": _mse(m.predict(Xv[te]), y[te])})
    return pd.DataFrame(rows).groupby("n_train_days")["mse"].mean().reset_index()


def per_day_mse(make_model, X, y, day):
    """Leave-one-day-out MSE for each held-out day."""
    Xv = X.values
    rows = []
    for d in np.unique(day):
        tr, te = day != d, day == d
        m = make_model().fit(Xv[tr], y[tr])
        rows.append({"test_day": int(d) + 1, "mse": _mse(m.predict(Xv[te]), y[te])})
    return pd.DataFrame(rows)


def ranking_metrics(y_true, y_pred, k_best=5, k_pick=10):
    """Within-day ranking: Spearman rho, and how many of the k truly cheapest
    blocks are among the k_pick predicted-cheapest ones (lower cost = better)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rho = float(spearmanr(y_true, y_pred)[0])
    best = set(np.argsort(y_true)[:k_best])
    picked = set(np.argsort(y_pred)[:k_pick])
    return {"spearman": rho, "best5_in_pred_top10": len(best & picked) / k_best}


def shap_tree_summary(model, X, out_path, title=""):
    """Optional SHAP bar chart for a fitted tree model.  Returns False if shap is unavailable."""
    try:
        import shap
    except Exception:  # noqa: BLE001
        return False
    sv = shap.TreeExplainer(model).shap_values(X.values)
    imp = np.abs(sv).mean(0)
    t = pd.DataFrame({"feature": X.columns, "mean_abs_shap": imp}).sort_values(
        "mean_abs_shap", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.barh(t["feature"], t["mean_abs_shap"],
            color=[GROUP_COLOR[feature_group(f)] for f in t["feature"]])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True