"""Track B: a more robust version of the shared Optuna tuner.

Why: with only ~143 training rows and 4 inner folds, picking the single best Optuna
trial often picks a lucky setting (the "winner's curse") and does worse on a new day
than the untuned default.  Two fixes, both built on tuning.OptunaTuned:

  * a smaller, more regularised search space (see run_tuning_b2.py)
  * TopKOptunaTuned: refit the best `top_k` trials and AVERAGE their predictions,
    so no single lucky setting dominates.

Works with evaluate(..., protocol="nested") exactly like OptunaTuned.
"""
import numpy as np

from tuning import OptunaTuned


class TopKOptunaTuned(OptunaTuned):
    """Optuna search, then average the predictions of the best `top_k` settings."""

    def __init__(self, build, space, n_trials=30, seed=0, top_k=5):
        super().__init__(build, space, n_trials=n_trials, seed=seed)
        self.top_k = top_k

    def fit(self, X, y, groups=None):
        super().fit(X, y, groups=groups)  # runs the study; self.model_ = best single setting
        X, y = np.asarray(X, float), np.asarray(y, float)
        done = [t for t in self.study_.trials if t.value is not None]
        best = sorted(done, key=lambda t: t.value)[: self.top_k]
        self.top_params_ = [t.params for t in best]
        self.models_ = [self.build(p).fit(X, y) for p in self.top_params_]
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return np.mean([m.predict(X) for m in self.models_], axis=0)