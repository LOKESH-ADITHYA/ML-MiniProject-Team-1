"""Sanity tests for Track B models (synthetic data, runs in seconds)."""
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models_nn import make_extended_mlp, make_paper_nn, paper_nn_factory  # noqa: E402
from models_trees import TREE_MODELS  # noqa: E402


def toy(n=120, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 0.5 + X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_paper_nn_learns_and_shapes():
    X, y = toy()
    m = make_paper_nn(n_hidden=3, n_seeds=3).fit(X[:90], y[:90])
    p = m.predict(X[90:])
    assert p.shape == (30,)
    assert np.mean((p - y[90:]) ** 2) < np.var(y[90:])  # beats mean predictor


def test_deterministic():
    X, y = toy()
    a = make_paper_nn(2, n_seeds=2).fit(X, y).predict(X)
    b = make_paper_nn(2, n_seeds=2).fit(X, y).predict(X)
    assert np.allclose(a, b)


def test_clone_and_factory():
    X, y = toy()
    m = clone(paper_nn_factory(4, n_seeds=2)())
    assert m.fit(X, y).predict(X).shape == (len(y),)


def test_extended_mlp_runs():
    X, y = toy()
    assert make_extended_mlp(n_seeds=2).fit(X, y).predict(X).shape == (len(y),)


def test_trees_run():
    X, y = toy()
    for name, mk in TREE_MODELS.items():
        assert mk().fit(X, y).predict(X).shape == (len(y),), name