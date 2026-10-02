"""Data layer (Track A).

Loads the authors' processed dataset, drops the invalid row, and attaches
day IDs and row-in-day positions so every protocol can group by day.

Source: processed_data.mat from https://github.com/ngeley/cs229project
(Krimsky & Ng, CS229 Stanford 2018; data from the Stanford Biomechatronics Lab).

Facts verified on the file:
- 180 rows x 29 features per subject, no NaNs.
- Rows are chronological: 5 days x 36 two-minute control conditions.
  The paper says 6 days, but the code concatenates 5. Day = row // 36.
- Subject 2 has one negative target (about -0.57). The authors' code drops it, and so do we -> 179 rows.
- EMG layout from their code: left = channels 1-4, 9-12; right = 5-8, 13-16.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_data.mat"
ROWS_PER_DAY = 36
N_DAYS = 5

STEP_COLS = ["LFz_peak", "L_peak_df", "L_peak_pf", "L_step_time",
             "RFz_peak", "R_peak_df", "R_peak_pf", "R_step_time", "step_width"]
EMG_COLS = [f"emg{i}" for i in range(1, 17)]
CTRL_COLS = ["ctrl1", "ctrl2", "ctrl3", "ctrl4"]
ALL_COLS = STEP_COLS + EMG_COLS + CTRL_COLS

EMG_LEFT = [f"emg{i}" for i in (1, 2, 3, 4, 9, 10, 11, 12)]
EMG_RIGHT = [f"emg{i}" for i in (5, 6, 7, 8, 13, 14, 15, 16)]

FEATURE_SETS = {
    "all": ALL_COLS,                    # 29
    "step+emg": STEP_COLS + EMG_COLS,   # 25, "no controls", most deployment-realistic
    "step": STEP_COLS,                  # 9
    "emg": EMG_COLS,                    # 16
}

SUBJECTS = {"S1": "michael", "S2": "eley"}  # paper Subject 1 / Subject 2
TARGET = "metabolic"
META_COLS = ["subject", "day", "row_in_day", "row"]


def load_subject(subject: str = "S1", path=DATA_PATH, drop_invalid: bool = True) -> pd.DataFrame:
    """One subject as a DataFrame: 29 feature columns + metabolic + subject/day/row_in_day/row.

    `day` is 1..5 and `row_in_day` is 0..35. Both are assigned BEFORE dropping
    the invalid row, so day boundaries stay correct.
    """
    if subject not in SUBJECTS:
        raise ValueError(f"subject must be one of {list(SUBJECTS)}")
    mat = loadmat(path)
    labels = [str(x[0]) for x in mat["data_labels"][0]]
    assert labels == ALL_COLS, "column order in .mat changed"
    key = SUBJECTS[subject]
    df = pd.DataFrame(mat[f"{key}_data"], columns=labels)
    df[TARGET] = mat[f"{key}_metabolics"].ravel()
    df["row"] = np.arange(len(df))
    df["day"] = df["row"] // ROWS_PER_DAY + 1
    df["row_in_day"] = df["row"] % ROWS_PER_DAY
    df["subject"] = subject
    if drop_invalid:
        df = df[df[TARGET] >= 0].reset_index(drop=True)
    return df


def load_all(**kw) -> pd.DataFrame:
    return pd.concat([load_subject(s, **kw) for s in SUBJECTS], ignore_index=True)


def get_xy(df: pd.DataFrame, feature_set="all"):
    """Return X (DataFrame), y (ndarray), days (ndarray).

    feature_set is a key of FEATURE_SETS or an explicit list of columns.
    """
    cols = FEATURE_SETS[feature_set] if isinstance(feature_set, str) else list(feature_set)
    return df[cols], df[TARGET].to_numpy(), df["day"].to_numpy()


def summary() -> pd.DataFrame:
    """Quick per-subject sanity table (rows, target variance, per-day means)."""
    rows = []
    for s in SUBJECTS:
        d = load_subject(s)
        r = {"subject": s, "rows": len(d), "target_var": d[TARGET].var(ddof=0),
             "lag1_autocorr": d[TARGET].autocorr(1)}
        r.update({f"day{k}_mean": v for k, v in d.groupby("day")[TARGET].mean().items()})
        rows.append(r)
    return pd.DataFrame(rows).set_index("subject")


if __name__ == "__main__":
    pd.set_option("display.precision", 4)
    print(summary().T)
