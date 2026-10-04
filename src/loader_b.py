"""Track B: small standalone loader used by ensemble / analysis / demo code.

It reads data/processed_data.mat directly (same file and same cleaning as the
shared loader: drop the one negative-target row, day = original row // 36), so
the Track B analysis scripts do not depend on any function names in data.py.
experiments/check_loader.py confirms it agrees with the shared harness.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT / "data" / "processed_data.mat"
KEY = {"S1": "michael", "S2": "eley"}
ROWS_PER_DAY = 36


def feature_group(name):
    if name.startswith("emg"):
        return "emg"
    if name.startswith("ctrl"):
        return "ctrl"
    return "step"


def load_subject_b(subject, feature_set="all"):
    """Return X (DataFrame), y (array), day (0..4 array) for one subject."""
    m = sio.loadmat(MAT)
    names = [str(a[0]) for a in m["data_labels"][0]]
    X = pd.DataFrame(m[KEY[subject] + "_data"], columns=names)
    y = m[KEY[subject] + "_metabolics"].ravel().astype(float)
    day = np.arange(len(y)) // ROWS_PER_DAY
    keep = y >= 0  # the authors' code drops negative metabolic values
    X, y, day = X[keep].reset_index(drop=True), y[keep], day[keep]
    groups = {"all": {"step", "emg", "ctrl"}, "step+emg": {"step", "emg"},
              "step": {"step"}, "emg": {"emg"}}[feature_set]
    cols = [c for c in names if feature_group(c) in groups]
    return X[cols], y, day