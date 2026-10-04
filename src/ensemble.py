"""Track B: ensembles of diverse models (simple average and stacking).

Members are built here from scikit-learn plus our own tanh network, with fixed
sensible settings (NOT tuned on the test days).  They were picked a priori for
diversity: a linear model, a kernel model, a small network and two tree models.

    make_voting()    average of the members' predictions
    make_stacking()  a non-negative linear model learns how to weight them

Both are zero-argument factories for evaluate(make_model, ...).
"""
from functools import partial

import numpy as np
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from models_nn import make_paper_nn
from models_trees import make_extra_trees, make_gradient_boosting


def make_ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def make_svr():
    return make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.02, gamma="scale"))


def make_gp():
    kernel = ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(0.1)
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=0))


MEMBERS = {
    "ridge": make_ridge,
    "svr": make_svr,
    "gp": make_gp,
    "nn": partial(make_paper_nn, n_hidden=5, n_seeds=5),
    "et": partial(make_extra_trees, n_estimators=200),
    "gb": partial(make_gradient_boosting, n_estimators=200, learning_rate=0.05),
}
DEFAULT_MEMBERS = ("ridge", "svr", "nn", "et", "gb")


def _estimators(members):
    return [(n, MEMBERS[n]()) for n in members]


def make_voting(members=DEFAULT_MEMBERS):
    return VotingRegressor(_estimators(members))


def make_stacking(members=DEFAULT_MEMBERS):
    return StackingRegressor(_estimators(members),
                             final_estimator=LinearRegression(positive=True), cv=5)


def voting_factory(members=DEFAULT_MEMBERS):
    return partial(make_voting, members=members)


def stacking_factory(members=DEFAULT_MEMBERS):
    return partial(make_stacking, members=members)