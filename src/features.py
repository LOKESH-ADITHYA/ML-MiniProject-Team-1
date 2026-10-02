"""Feature engineering and feature selection (Track A).

Engineering (label-free, so they can be added to the whole table before splitting):
    add_symmetry(df)      L/R mean and |L-R| for each bilateral step feature
    add_emg_bilateral(df) mean of each left/right EMG channel pair (8 pairs)
    add_emg_summary(df)   total / left / right EMG activation and L:R ratio
    day_center(df)        subtract each day's feature mean (removes day-to-day sensor offsets).
                          Note: transductive. It uses the held-out day's FEATURES (never its
                          targets), which at deployment means centring on that session so far.
    engineer(df, groups)  apply any combination and return (df, new_columns)

Selection (sklearn transformers, so they are fit inside training folds only):
    make_selector(method, k)   -> Pipeline([scaler, selector, Ridge]) factory
    methods: none, pca, forward, mi, rfe, stability, pls, ard
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import (RFE, SelectKBest, SequentialFeatureSelector,
                                       mutual_info_regression)
from sklearn.linear_model import ARDRegression, Lasso, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import EMG_COLS, EMG_LEFT, EMG_RIGHT, FEATURE_SETS
from models_classical import PLS

BILATERAL_STEP = [("LFz_peak", "RFz_peak", "Fz_peak"), ("L_peak_df", "R_peak_df", "peak_df"),
                  ("L_peak_pf", "R_peak_pf", "peak_pf"), ("L_step_time", "R_step_time", "step_time")]
EMG_PAIRS = list(zip(EMG_LEFT, EMG_RIGHT))  # (emg1, emg5), (emg2, emg6), ..., (emg12, emg16)


# ===================================================================== engineering
def add_symmetry(df):
    df = df.copy()
    new = []
    for l, r, n in BILATERAL_STEP:
        df[f"{n}_mean"] = (df[l] + df[r]) / 2
        df[f"{n}_asym"] = (df[l] - df[r]).abs()
        new += [f"{n}_mean", f"{n}_asym"]
    return df, new


def add_emg_bilateral(df):
    df = df.copy()
    new = []
    for i, (l, r) in enumerate(EMG_PAIRS, 1):
        df[f"emgpair{i}"] = (df[l] + df[r]) / 2
        new.append(f"emgpair{i}")
    return df, new


def add_emg_summary(df):
    df = df.copy()
    df["emg_total"] = df[EMG_COLS].sum(axis=1)
    df["emg_left"] = df[EMG_LEFT].sum(axis=1)
    df["emg_right"] = df[EMG_RIGHT].sum(axis=1)
    df["emg_lr_ratio"] = df["emg_left"] / df["emg_right"].replace(0, np.nan)
    df["emg_lr_ratio"] = df["emg_lr_ratio"].fillna(1.0)
    return df, ["emg_total", "emg_left", "emg_right", "emg_lr_ratio"]


def day_center(df, cols=None):
    """Subtract per-day feature means (label-free)."""
    df = df.copy()
    cols = cols or FEATURE_SETS["step+emg"]
    df[cols] = df[cols] - df.groupby("day")[cols].transform("mean")
    return df


ENGINEERED = {"symmetry": add_symmetry, "emg_bilateral": add_emg_bilateral, "emg_summary": add_emg_summary}


def engineer(df, groups=("symmetry", "emg_bilateral", "emg_summary"), base="step+emg"):
    """Add the chosen engineered groups. Returns (df, feature_columns = base + new)."""
    cols = list(FEATURE_SETS[base])
    for g in groups:
        df, new = ENGINEERED[g](df)
        cols += new
    return df, cols


# ===================================================================== selection
class StabilitySelector(BaseEstimator, TransformerMixin):
    """Lasso stability selection: refit Lasso on many half-samples, keep features chosen
    in at least `threshold` of them. Expects standardised input."""

    def __init__(self, alpha=0.01, n_resamples=50, threshold=0.7, random_state=0):
        self.alpha = alpha
        self.n_resamples = n_resamples
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        counts = np.zeros(X.shape[1])
        for _ in range(self.n_resamples):
            idx = rng.choice(n, n // 2, replace=False)
            counts += Lasso(alpha=self.alpha, max_iter=20_000).fit(X[idx], y[idx]).coef_ != 0
        self.freq_ = counts / self.n_resamples
        self.support_ = self.freq_ >= self.threshold
        if not self.support_.any():  # always keep at least the most stable feature
            self.support_[np.argmax(self.freq_)] = True
        return self

    def transform(self, X):
        return X[:, self.support_]

    def get_support(self):
        return self.support_


def _ridge():
    return RidgeCV(alphas=np.logspace(-3, 3, 13))


def make_selector(method="none", k=10, threshold=0.7):
    """Factory returning a fresh scaler -> selector -> Ridge pipeline.

    k is the number of features (or components for pca/pls) kept. For 'stability' the
    threshold sets k implicitly. For 'ard', ARD itself is the model (it prunes features).
    """
    def build():
        s = StandardScaler()
        if method == "none":
            return make_pipeline(s, _ridge())
        if method == "pca":
            return make_pipeline(s, PCA(n_components=k), _ridge())
        if method == "forward":
            return make_pipeline(s, SequentialFeatureSelector(
                Ridge(1.0), n_features_to_select=k, direction="forward",
                cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error"), _ridge())
        if method == "mi":
            return make_pipeline(s, SelectKBest(lambda X, y: mutual_info_regression(X, y, random_state=0), k=k),
                                 _ridge())
        if method == "rfe":
            return make_pipeline(s, RFE(Ridge(1.0), n_features_to_select=k), _ridge())
        if method == "stability":
            return make_pipeline(s, StabilitySelector(threshold=threshold), _ridge())
        if method == "pls":
            return make_pipeline(s, PLS(n_components=k))
        if method == "ard":
            return make_pipeline(s, ARDRegression(max_iter=1000))
        raise ValueError(method)
    return build


SELECTION_METHODS = ["pca", "forward", "mi", "rfe", "pls"]  # k-driven methods


def selected_features(method, X: pd.DataFrame, y, k=10, threshold=0.7):
    """Fit a selector on the full data (for REPORTING which features it picks, not for scoring)."""
    pipe = make_selector(method, k, threshold)().fit(X.to_numpy(float), y)
    step = pipe[1]
    if hasattr(step, "get_support"):
        return list(X.columns[step.get_support()])
    if method == "ard":
        coef = pipe[-1].coef_
        return list(X.columns[np.abs(coef) > 1e-3 * np.abs(coef).max()])
    return None
