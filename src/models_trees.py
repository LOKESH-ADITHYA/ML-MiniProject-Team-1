"""Track B (Person B): tree-based models.

Trees need no feature scaling, so these are plain estimators (still
zero-argument factories, as the harness expects).  Defaults are sensible
starting points, NOT tuned; tuning comes later in src/tuning.py.
"""
from functools import partial

from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)


def make_random_forest(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                       random_state=0):
    return RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        max_features=max_features, random_state=random_state, n_jobs=-1)


def make_extra_trees(n_estimators=300, min_samples_leaf=2, max_features=1.0,
                     random_state=0):
    return ExtraTreesRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        max_features=max_features, random_state=random_state, n_jobs=-1)


def make_gradient_boosting(n_estimators=300, learning_rate=0.03, max_depth=2,
                           subsample=0.7, random_state=0):
    return GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, subsample=subsample, random_state=random_state)


TREE_MODELS = {
    "random_forest": make_random_forest,
    "extra_trees": make_extra_trees,
    "gradient_boosting": make_gradient_boosting,
}


def tree_factory(name, **kwargs):
    return partial(TREE_MODELS[name], **kwargs)