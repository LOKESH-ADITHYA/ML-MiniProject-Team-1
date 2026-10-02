"""Evaluation harness (Track A builds, Track B uses).

One entry point:

    from evaluation import evaluate
    res = evaluate(make_model, subject="S1", feature_set="all", protocol="lodo")
    res.summary      # dict: mse, rmse, mae, r2, skill, mean_mse, spearman_day, top5_in_top10
    res.folds        # DataFrame, one row per outer fold
    res.pred         # out-of-fold predictions aligned with res.y

`make_model` is a zero-argument callable that returns a FRESH sklearn-style estimator
(fit/predict). Any preprocessing (scaling, feature selection, PCA) must be inside it,
e.g. a Pipeline, so it is fit on training folds only.

Protocols
  P1 "kfold"    paper-style shuffled K-fold (default 10 folds x 5 repeats; k=5 matches their code)
  P2 "lodo"     leave-one-day-out: train on 4 days, test on the unseen 5th
  P3 "nested"   LODO outer loop; the estimator tunes itself on the 4 training days.
                Use `tuned(estimator, grid)` from this module, or any estimator with
                `needs_groups = True` whose fit(X, y, groups=...) does its own inner LODO
                (e.g. Track B's Optuna tuner).
  P4 "cross"    train on one subject, test on the other (use cross_subject()).
  P5 "forward"  forward-chaining: train on days 1..k, test on day k+1 (k = 2..4).

Metrics are computed on pooled out-of-fold predictions. The mean predictor is re-scored on
the SAME folds (`mean_mse`), because under LODO it is far worse than the variance.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut

from data import SUBJECTS, get_xy, load_subject

NOISE_FLOOR = 0.0025  # min achievable MSE from metabolic measurement noise (paper)
PROTOCOLS = ("kfold", "lodo", "nested", "forward")


# ------------------------------------------------------------------ fold generators
def make_folds(days, protocol, n_splits=10, n_repeats=5, seed=0):
    """Yield (repeat, fold, train_idx, test_idx)."""
    idx = np.arange(len(days))
    if protocol == "kfold":
        for r in range(n_repeats):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            for f, (tr, te) in enumerate(kf.split(idx)):
                yield r, f, tr, te
    elif protocol in ("lodo", "nested"):
        for f, (tr, te) in enumerate(LeaveOneGroupOut().split(idx, groups=days)):
            yield 0, f, tr, te
    elif protocol == "forward":
        for k in range(2, days.max()):
            yield 0, k, idx[days <= k], idx[days == k + 1]
    else:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")


# ------------------------------------------------------------------ metrics
def regression_metrics(y, p):
    mse = float(np.mean((y - p) ** 2))
    return {"mse": mse, "rmse": mse ** 0.5, "mae": float(np.mean(np.abs(y - p))),
            "r2": 1 - mse / float(np.var(y))}


def ranking_metrics(y, p, days):
    """HILO usefulness: does the model rank the 36 settings of a day correctly?

    spearman_day   mean Spearman correlation between predicted and true cost within each day
    top5_in_top10  fraction of the truly cheapest 5 settings that land in the predicted cheapest 10
    """
    rhos, hits = [], []
    for d in np.unique(days):
        m = days == d
        if m.sum() < 10 or np.ptp(p[m]) == 0:  # constant predictions cannot rank
            continue
        rhos.append(spearmanr(y[m], p[m]).statistic)
        true5 = set(np.argsort(y[m])[:5])
        pred10 = set(np.argsort(p[m])[:10])
        hits.append(len(true5 & pred10) / 5)
    return {"spearman_day": float(np.nanmean(rhos)) if rhos else np.nan,
            "top5_in_top10": float(np.mean(hits)) if hits else np.nan}


# ------------------------------------------------------------------ nested tuning helper
class tuned(BaseEstimator, RegressorMixin):
    """Grid search with an inner leave-one-day-out loop (protocol P3).

    tuned(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1, 10]})
    """
    needs_groups = True

    def __init__(self, estimator, param_grid, scoring="neg_mean_squared_error"):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring

    def fit(self, X, y, groups=None):
        if groups is None or len(np.unique(groups)) < 2:  # fall back to 5-fold when no day info
            cv = KFold(5, shuffle=True, random_state=0)
        else:
            cv = LeaveOneGroupOut()
        self.search_ = GridSearchCV(clone(self.estimator), self.param_grid, cv=cv,
                                    scoring=self.scoring, n_jobs=-1)
        self.search_.fit(X, y, groups=groups)
        self.best_params_ = self.search_.best_params_
        self.best_estimator_ = self.search_.best_estimator_
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)


def _fit(model, X, y, groups):
    if getattr(model, "needs_groups", False):
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


# ------------------------------------------------------------------ main entry point
@dataclass
class EvalResult:
    summary: dict
    folds: pd.DataFrame
    y: np.ndarray
    pred: np.ndarray
    days: np.ndarray
    extra: dict = field(default_factory=dict)


def evaluate(make_model, subject="S1", feature_set="all", protocol="lodo", df=None,
             transform=None, n_splits=10, n_repeats=5, seed=0, name=None):
    """Score `make_model()` on one subject under one protocol.

    df        optional pre-loaded DataFrame (e.g. with engineered columns already added)
    transform optional label-free function df -> df applied BEFORE splitting
              (only for things like per-day centering, which uses no targets)
    """
    df = load_subject(subject) if df is None else df
    if transform is not None:
        df = transform(df)
    X, y, days = get_xy(df, feature_set)
    X = X.to_numpy(dtype=float)

    preds = np.zeros((n_repeats if protocol == "kfold" else 1, len(y)))
    mask = np.zeros_like(preds, dtype=bool)
    mean_preds = np.zeros_like(preds)
    fold_rows, params = [], []
    for r, f, tr, te in make_folds(days, protocol, n_splits, n_repeats, seed):
        model = _fit(make_model(), X[tr], y[tr], days[tr])
        p = model.predict(X[te])
        preds[r, te], mask[r, te], mean_preds[r, te] = p, True, y[tr].mean()
        row = {"repeat": r, "fold": f, "n_train": len(tr), "n_test": len(te),
               **regression_metrics(y[te], p),
               "mean_mse": float(np.mean((y[te] - y[tr].mean()) ** 2))}
        if protocol != "kfold":
            row["test_day"] = int(days[te][0])
        if hasattr(model, "best_params_"):
            params.append(model.best_params_)
            row["best_params"] = str(model.best_params_)
        fold_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    used = mask.any(0)  # forward-chaining never tests days 1-2
    if protocol == "kfold":
        # each repeat covers every row once; score each repeat, then average
        per_rep = [regression_metrics(y, preds[r]) for r in range(preds.shape[0])]
        summ = {k: float(np.mean([m[k] for m in per_rep])) for k in per_rep[0]}
        summ["mse_std"] = float(np.std([m["mse"] for m in per_rep]))
        summ["mean_mse"] = float(np.mean([np.mean((y - mean_preds[r]) ** 2) for r in range(len(per_rep))]))
        pred = preds.mean(0)
    else:
        pred = preds[0]
        summ = regression_metrics(y[used], pred[used])
        summ["mse_std"] = float(folds["mse"].std())  # spread across held-out days
        summ["mean_mse"] = float(np.mean((y[used] - mean_preds[0][used]) ** 2))
        summ.update(ranking_metrics(y[used], pred[used], days[used]))
    summ["skill"] = 1 - summ["mse"] / summ["mean_mse"]
    summ = {"model": name, "subject": subject, "features": feature_set if isinstance(feature_set, str)
            else f"custom({len(feature_set)})", "protocol": protocol, **summ}
    return EvalResult(summ, folds, y[used], pred[used], days[used], {"best_params": params})


def cross_subject(make_model, feature_set="step+emg", train="S1", test="S2", df_train=None, df_test=None):
    """P4: fit on one subject, score on the other. Control params excluded by default
    because they are device-specific."""
    dtr = load_subject(train) if df_train is None else df_train
    dte = load_subject(test) if df_test is None else df_test
    Xtr, ytr, gtr = get_xy(dtr, feature_set)
    Xte, yte, gte = get_xy(dte, feature_set)
    model = _fit(make_model(), Xtr.to_numpy(float), ytr, gtr)
    p = model.predict(Xte.to_numpy(float))
    out = {"train": train, "test": test, "features": feature_set, **regression_metrics(yte, p),
           "mean_mse": float(np.mean((yte - ytr.mean()) ** 2))}
    out["skill"] = 1 - out["mse"] / out["mean_mse"]
    out.update(ranking_metrics(yte, p, gte))
    return out


def run_grid(models: dict, subjects=tuple(SUBJECTS), feature_sets=("all",), protocols=("lodo",), **kw):
    """Convenience: evaluate every (model, subject, feature set, protocol). Returns a tidy DataFrame."""
    rows = []
    for name, make in models.items():
        for s in subjects:
            for fs in feature_sets:
                for pr in protocols:
                    rows.append(evaluate(make, s, fs, pr, name=name, **kw).summary)
    return pd.DataFrame(rows)
