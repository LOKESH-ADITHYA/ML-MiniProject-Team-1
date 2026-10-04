"""Hyperparameter tuning with Optuna inside nested leave-one-day-out (protocol P3).

How it works
    For each outer fold (one held-out day), Optuna's TPE sampler tries `n_trials` settings.
    Each setting is scored by an INNER leave-one-day-out on the 4 training days only, so the
    test day never influences tuning. The best setting is refit on all 4 days and then scored
    on the held-out day by evaluation.evaluate().

Use with the shared harness
    from evaluation import evaluate
    from tuning import optuna_model
    res = evaluate(optuna_model("gb"), "S1", "all", "nested")

Plugging in your own model (e.g. a PyTorch net)
    def build(p):                       # p = dict of sampled settings
        return MyTorchRegressor(hidden=p["neurons"], weight_decay=p["weight_decay"], ...)
    res = evaluate(lambda: OptunaTuned(build, SPACES["nn_paper"]), "S1", "all", "nested")
    Any object with fit(X, y) / predict(X) works. Put scaling inside it (or wrap in a Pipeline).

Search spaces follow section 6.7 of the team plan.
"""
import numpy as np
import optuna
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------ the tuner
class OptunaTuned(BaseEstimator, RegressorMixin):
    """Estimator that tunes itself with Optuna on whatever data it is fit on.

    build      function(params: dict) -> fresh estimator
    space      function(trial) -> params dict (uses trial.suggest_*)
    n_trials   Optuna trials per fit (per outer fold)
    """
    needs_groups = True  # evaluation.evaluate passes training days as `groups`

    def __init__(self, build, space, n_trials=30, seed=0):
        self.build = build
        self.space = space
        self.n_trials = n_trials
        self.seed = seed

    def _inner_splits(self, X, groups):
        if groups is not None and len(np.unique(groups)) >= 2:
            return list(LeaveOneGroupOut().split(X, groups=groups))
        return list(KFold(5, shuffle=True, random_state=self.seed).split(X))

    def fit(self, X, y, groups=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        splits = self._inner_splits(X, groups)

        def objective(trial):
            params = self.space(trial)
            errs = []
            for tr, va in splits:
                m = self.build(params).fit(X[tr], y[tr])
                errs.append(np.mean((m.predict(X[va]) - y[va]) ** 2))
            return float(np.mean(errs))

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.study_ = optuna.create_study(direction="minimize", sampler=sampler)
        self.study_.optimize(objective, n_trials=self.n_trials)
        self.best_params_ = self.study_.best_params
        self.best_inner_mse_ = self.study_.best_value
        self.model_ = self.build(self.best_params_).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, float))


# ------------------------------------------------------------------ search spaces (plan 6.7)
def space_nn_paper(t):
    return {"neurons": t.suggest_int("neurons", 1, 10),
            "weight_decay": t.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
            "lr": t.suggest_float("lr", 1e-3, 1e-1, log=True),
            "epochs": t.suggest_int("epochs", 200, 2000, step=200),
            "activation": t.suggest_categorical("activation", ["tanh", "relu"])}


def space_nn_extended(t):
    p = space_nn_paper(t)
    p.pop("neurons")
    p.update({"layers": t.suggest_int("layers", 1, 3),
              "width": t.suggest_int("width", 4, 32),
              "dropout": t.suggest_float("dropout", 0.0, 0.3)})  # used by a torch builder
    return p


def space_forest(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": t.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
            "max_features": t.suggest_categorical("max_features", ["sqrt", 0.33, 0.5, 1.0])}


def space_gb(t):
    return {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 5),
            "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10)}


# ------------------------------------------------------------------ reference builders
# sklearn stand-ins so this file runs on its own. Swap in the PyTorch builders from
# models_nn.py once they exist (same parameter names).
def build_nn_paper(p):
    """Full-batch L-BFGS (closest sklearn analogue to MATLAB's trainbr); lr is unused by L-BFGS.
    sklearn's alpha is summed over the batch, so it needs larger values than torch weight decay;
    we rescale: alpha = weight_decay * 100."""
    return make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(p["neurons"],), activation=p["activation"], alpha=p["weight_decay"] * 100,
        max_iter=p["epochs"], solver="lbfgs", random_state=0))


def build_nn_extended(p):  # sklearn MLP has no dropout; the torch version should use p["dropout"]
    return make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(p["width"],) * p["layers"], activation=p["activation"], alpha=p["weight_decay"] * 100,
        max_iter=p["epochs"], solver="lbfgs", random_state=0))


def build_rf(p):
    return RandomForestRegressor(random_state=0, n_jobs=-1, **p)


def build_et(p):
    return ExtraTreesRegressor(random_state=0, n_jobs=-1, **p)


def build_gb(p):
    return GradientBoostingRegressor(random_state=0, **p)


SPACES = {
    "nn_paper": space_nn_paper,
    "nn_extended": space_nn_extended,
    "rf": space_forest,
    "et": space_forest,
    "gb": space_gb,
}
BUILDERS = {
    "nn_paper": build_nn_paper,
    "nn_extended": build_nn_extended,
    "rf": build_rf,
    "et": build_et,
    "gb": build_gb,
}


def optuna_model(name, n_trials=30, seed=0, build=None):
    """Factory for evaluate(..., protocol="nested"). Pass `build` to use a custom (e.g. torch) builder."""
    return lambda: OptunaTuned(build or BUILDERS[name], SPACES[name], n_trials=n_trials, seed=seed)
