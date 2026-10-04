"""Sanity tests for the Track B ensemble / interpretation code (synthetic data, seconds)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ensemble import make_stacking, make_voting  # noqa: E402
from interpret import (learning_curve_days, per_day_mse,  # noqa: E402
                       permutation_importance_lodo, ranking_metrics)
from models_nn import make_paper_nn  # noqa: E402


def toy(n=150, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=["emg1", "emg2", "ctrl1", "LFz_peak", "step_width", "emg3"])
    y = 0.5 + X["emg1"].values - 0.5 * X["emg2"].values + 0.1 * rng.normal(size=n)
    day = np.repeat(np.arange(5), n // 5)
    return X, y, day


def test_ensembles_fit_predict_and_clone():
    X, y, _ = toy()
    for make in (lambda: make_voting(("ridge", "nn")), lambda: make_stacking(("ridge", "svr"))):
        m = clone(make())
        assert m.fit(X.values, y).predict(X.values).shape == (len(y),)


def test_network_is_recognised_as_regressor():
    from sklearn.base import is_regressor
    assert is_regressor(make_paper_nn())


def test_permutation_importance_finds_signal():
    X, y, day = toy()
    t = permutation_importance_lodo(lambda: make_paper_nn(3, n_seeds=2), X, y, day, n_repeats=3)
    assert t["feature"].iloc[0] in ("emg1", "emg2")


def test_learning_curve_and_per_day_shapes():
    X, y, day = toy()
    mk = lambda: make_paper_nn(2, n_seeds=1)
    lc = learning_curve_days(mk, X, y, day)
    assert list(lc["n_train_days"]) == [1, 2, 3, 4]
    assert len(per_day_mse(mk, X, y, day)) == 5


def test_ranking_metrics_perfect():
    y = np.arange(36.0)
    r = ranking_metrics(y, y)
    assert r["spearman"] == 1.0 and r["best5_in_pred_top10"] == 1.0

def test_topk_tuner_averages_models():
    from sklearn.linear_model import Ridge
    from tuning_b import TopKOptunaTuned
    X, y, day = toy()
    space = lambda t: {"alpha": t.suggest_float("alpha", 1e-3, 10, log=True)}
    m = TopKOptunaTuned(lambda p: Ridge(**p), space, n_trials=6, top_k=3)
    m.fit(X.values, y, groups=day)
    assert len(m.models_) == 3 and m.predict(X.values).shape == (len(y),)
    assert clone(m).top_k == 3
