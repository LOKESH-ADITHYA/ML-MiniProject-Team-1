"""Confirm the Track B loader (src/loader_b.py) agrees with the shared harness.

    python experiments/check_loader.py

Runs the same Ridge model through the shared evaluate() and through our own loop.

"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ensemble import make_ridge  # noqa: E402
from evaluation import evaluate  # noqa: E402
from loader_b import load_subject_b  # noqa: E402

for s in ["S1", "S2"]:
    X, y, day = load_subject_b(s, "all")
    p = np.zeros_like(y)
    for d in np.unique(day):
        te = day == d
        p[te] = make_ridge().fit(X.values[~te], y[~te]).predict(X.values[te])
    mine = float(np.mean((p - y) ** 2))
    theirs = float(evaluate(make_ridge, subject=s, feature_set="all", protocol="lodo").summary["mse"])
    ok = abs(mine - theirs) / theirs < 0.05
    print(f"{s}: rows={len(y)}  ours={mine:.5f}  harness={theirs:.5f}  {'OK' if ok else 'MISMATCH - tell Lokesh'}")