"""Track B (Person B): neural-network models.

Everything here follows the harness contract from the README:
    make_model() -> a fresh estimator with fit(X, y) / predict(X)
Scaling lives INSIDE the model (a Pipeline), so the harness can split first
and fit the scaler on the training rows only (no leakage).

Design notes
------------
* The paper used one hidden layer, tanh, linear output and MATLAB's Bayesian
  Regularization.  Python has no built-in equivalent, so we use full-batch
  L-BFGS with an L2 weight penalty (`alpha`) as the stand-in.  L-BFGS is the
  right optimiser for ~180 rows: no learning rate, no epochs to tune.
* Tiny networks on tiny data are unstable (the authors saw this too), so every
  model is an average of `n_seeds` networks started from different seeds.
* The target is standardised inside `fit` so `alpha` means the same thing for
  every subject, and predictions are mapped back to the original units.
"""
import warnings
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _as_tuple(h):
    """Accept 3, (3,) or (32, 16) and always return a tuple of ints."""
    if isinstance(h, (int, np.integer)):
        return (int(h),)
    return tuple(int(v) for v in h)


class SeedAveragedMLP(RegressorMixin, BaseEstimator):
    """Average of several small MLPs trained with L-BFGS + L2 penalty."""

    def __init__(self, hidden_layer_sizes=(3,), alpha=1.0, activation="tanh",
                 n_seeds=10, max_iter=2000, random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.activation = activation
        self.n_seeds = n_seeds
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() if y.std() > 0 else 1.0
        y_s = (y - self.y_mean_) / self.y_std_
        self.models_ = []
        for s in range(self.n_seeds):
            net = MLPRegressor(
                hidden_layer_sizes=_as_tuple(self.hidden_layer_sizes),
                activation=self.activation,
                solver="lbfgs",
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state + s,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                net.fit(X, y_s)
            self.models_.append(net)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        p = np.mean([m.predict(X) for m in self.models_], axis=0)
        return p * self.y_std_ + self.y_mean_


# ---------------------------------------------------------------- factories
# DEFAULT_ALPHA is a sensible starting value, NOT a tuned one.  Tuning comes
# later in src/tuning.py.
DEFAULT_ALPHA = 5.0


def make_paper_nn(n_hidden=3, alpha=DEFAULT_ALPHA, n_seeds=10):
    """Paper-style network: 1 hidden layer, tanh, linear output."""
    return make_pipeline(
        StandardScaler(),
        SeedAveragedMLP(hidden_layer_sizes=(n_hidden,), alpha=alpha,
                        activation="tanh", n_seeds=n_seeds),
    )


def make_extended_mlp(hidden=(32, 16), alpha=DEFAULT_ALPHA, n_seeds=10,
                      activation="relu"):
    """Wider / deeper variant (our 'improved' network, Tier 3)."""
    return make_pipeline(
        StandardScaler(),
        SeedAveragedMLP(hidden_layer_sizes=hidden, alpha=alpha,
                        activation=activation, n_seeds=n_seeds),
    )


def paper_nn_factory(n_hidden, alpha=DEFAULT_ALPHA, n_seeds=10):
    """Zero-argument factory, exactly what evaluate(make_model, ...) wants."""
    return partial(make_paper_nn, n_hidden=n_hidden, alpha=alpha, n_seeds=n_seeds)


def extended_mlp_factory(hidden=(32, 16), alpha=DEFAULT_ALPHA, n_seeds=10,
                         activation="relu"):
    return partial(make_extended_mlp, hidden=hidden, alpha=alpha,
                   n_seeds=n_seeds, activation=activation)