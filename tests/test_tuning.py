"""Sanity tests for src/tuning.py (run: pytest -q)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import evaluate  # noqa: E402
from tuning import BUILDERS, OptunaTuned, SPACES, optuna_model  # noqa: E402


def test_tuner_uses_inner_lodo_and_records_params():
    r = evaluate(optuna_model("gb", n_trials=2), "S1", "step", "nested")
    assert len(r.extra["best_params"]) == 5 and np.isfinite(r.summary["mse"])


def test_custom_builder_plugs_in():
    from sklearn.linear_model import Ridge
    space = lambda t: {"alpha": t.suggest_float("alpha", 1e-3, 10, log=True)}
    m = OptunaTuned(lambda p: Ridge(**p), space, n_trials=3)
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = X @ [1.0, -2.0, 0.5]
    m.fit(X, y, groups=np.repeat([1, 2, 3, 4], 10))
    assert "alpha" in m.best_params_ and m.predict(X).shape == (40,)


def test_every_space_builds():
    import optuna
    for name in SPACES:
        BUILDERS[name](SPACES[name](optuna.create_study().ask()))
