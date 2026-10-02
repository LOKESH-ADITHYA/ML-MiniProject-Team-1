"""Classical and kernel models (Track A).

Every factory returns a fresh Pipeline with the scaler inside, so it is fit on training
folds only. Two flavours per model:

    MODELS[name]()           sensible defaults (internal CV where cheap, e.g. LassoCV)
    tuned_model(name)        wrapped in evaluation.tuned with GRIDS[name] -> protocol "nested"

Tree models (RF / Extra Trees / boosting) and neural nets live in Track B's files.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, Lasso,
                                  LassoCV, Ridge, RidgeCV)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from evaluation import tuned

LASSO_ALPHAS = np.logspace(-5, -1, 25)  # same lambda grid as the authors' k_fold_lr


class PLS(PLSRegression):
    """PLSRegression returns 2-D predictions; flatten so it behaves like other regressors."""

    def predict(self, X, copy=True):
        return super().predict(X, copy=copy).ravel()


def mean_predictor():
    return DummyRegressor(strategy="mean")


def lasso():
    """Paper baseline: L1 linear regression, lambda by inner 10-fold CV."""
    return make_pipeline(StandardScaler(), LassoCV(alphas=LASSO_ALPHAS, cv=10, max_iter=50_000))


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))


def elastic_net():
    return make_pipeline(StandardScaler(),
                         ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                                      alphas=np.logspace(-4, 0, 20), cv=10, max_iter=50_000))


def pls(n=4):
    return make_pipeline(StandardScaler(), PLS(n_components=n))


def bayesian_ridge():
    return make_pipeline(StandardScaler(), BayesianRidge())


def ard():
    """Automatic Relevance Determination: closest linear cousin of the paper's
    Bayesian-regularised training; also prunes irrelevant features."""
    return make_pipeline(StandardScaler(), ARDRegression(max_iter=1000))


def svr():
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.05))


def kernel_ridge():
    """KernelRidge has no intercept, so the target is standardised too (TransformedTargetRegressor)."""
    return TransformedTargetRegressor(
        make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=1.0, gamma=0.01)),
        transformer=StandardScaler())


def gaussian_process():
    """RBF + white noise; hyperparameters fit by marginal likelihood on training folds."""
    k = ConstantKernel(0.1, (1e-3, 10)) * RBF(5.0, (0.1, 100)) + WhiteKernel(0.01, (1e-4, 1))
    return make_pipeline(StandardScaler(),
                         GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2,
                                                  random_state=0))


MODELS = {
    "Mean": mean_predictor,
    "Lasso": lasso,
    "Ridge": ridge,
    "ElasticNet": elastic_net,
    "PLS": pls,
    "BayesianRidge": bayesian_ridge,
    "ARD": ard,
    "SVR": svr,
    "KernelRidge": kernel_ridge,
    "GP": gaussian_process,
}

# --------------------------------------------------------------- tuning grids (protocol P3)
_BASE = {
    "Lasso": (lambda: make_pipeline(StandardScaler(), Lasso(max_iter=50_000)),
              {"lasso__alpha": np.logspace(-4, 0, 13)}),
    "Ridge": (lambda: make_pipeline(StandardScaler(), Ridge()),
              {"ridge__alpha": np.logspace(-3, 3, 13)}),
    "ElasticNet": (lambda: make_pipeline(StandardScaler(), ElasticNet(max_iter=50_000)),
                   {"elasticnet__alpha": np.logspace(-4, 0, 9),
                    "elasticnet__l1_ratio": [0.1, 0.5, 0.9, 1.0]}),
    "PLS": (lambda: pls(), {"pls__n_components": list(range(1, 16))}),
    "SVR": (svr, {"svr__C": [0.1, 0.3, 1, 3, 10, 30, 100],
                  "svr__gamma": ["scale", 1e-3, 3e-3, 1e-2, 3e-2],
                  "svr__epsilon": [0.005, 0.02, 0.05, 0.1]}),
    "KernelRidge": (kernel_ridge, {"regressor__kernelridge__alpha": [0.01, 0.1, 1, 10],
                                   "regressor__kernelridge__gamma": [1e-3, 3e-3, 1e-2, 3e-2, 0.1]}),
}
GRIDS = {k: g for k, (_, g) in _BASE.items()}


def tuned_model(name):
    """Factory for protocol 'nested': grid search with inner leave-one-day-out.
    BayesianRidge, ARD and GP tune themselves (evidence / marginal likelihood), so they
    are returned as-is; Mean has nothing to tune."""
    if name in _BASE:
        base, grid = _BASE[name]
        return lambda: tuned(base(), grid)
    return MODELS[name]


TUNED_MODELS = {name: tuned_model(name) for name in MODELS}
