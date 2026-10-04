"""Track B experiment b5: ensembles vs their members (protocols kfold and lodo).

    python experiments/run_ensemble.py            # ~15 min
    python experiments/run_ensemble.py --quick    # fast check

Output: results/track_b/b5_ensemble.csv
"""
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from ensemble import (MEMBERS, make_gp, make_ridge, make_svr,  # noqa: E402
                      stacking_factory, voting_factory)
from evaluation import evaluate  # noqa: E402

OUT = ROOT / "results" / "track_b"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv

MODELS = {
    "ridge": make_ridge,
    "svr": make_svr,
    "gp": make_gp,
    "voting_all": voting_factory(("ridge", "svr", "nn", "et", "gb")),
    # chosen AFTER seeing that trees extrapolate poorly to a new day: report it as post-hoc
    "voting_smooth": voting_factory(("ridge", "gp", "nn")),
    "stacking": stacking_factory(("ridge", "svr", "nn", "et", "gb")),
}


def summary_dict(res):
    s = res.summary
    if isinstance(s, pd.DataFrame):
        s = s.iloc[0] if len(s) == 1 else s.mean(numeric_only=True)
    out = {}
    for k, v in dict(s).items():
        try:
            out[str(k).lower()] = float(v)
        except (TypeError, ValueError):
            pass
    return out


rows = []
fsets = ["all"] if QUICK else ["all", "step+emg"]
models = {k: MODELS[k] for k in (["ridge", "voting_all"] if QUICK else MODELS)}
for protocol in ["kfold", "lodo"]:
    for subject in ["S1", "S2"]:
        for fs in fsets:
            for name, make in models.items():
                t0 = time.time()
                res = evaluate(make, subject=subject, feature_set=fs, protocol=protocol)
                row = {"experiment": "b5", "model": name, "subject": subject, "feature_set": fs,
                       "protocol": protocol, **summary_dict(res), "seconds": round(time.time() - t0, 1)}
                rows.append(row)
                print(f"  b5 {name:<14} {subject} {fs:<9} {protocol:<6} mse={row['mse']:.4f}")
pd.DataFrame(rows).to_csv(OUT / "b5_ensemble.csv", index=False)
print(f"-> saved {OUT / 'b5_ensemble.csv'}")