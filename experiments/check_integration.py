"""Check that Person A's harness is ready for Person B.  Run this FIRST:

    python experiments/check_integration.py

It prints PASS/FAIL per check.  If anything fails, copy the output to Person A.
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ok = True


def check(name, fn):
    global ok
    try:
        out = fn()
        print(f"PASS  {name}" + (f"  ->  {out}" if out is not None else ""))
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def imports():
    global evaluate
    from evaluation import evaluate  # noqa: F401
    import data  # noqa: F401


def run(subject, fs, protocol):
    def _f():
        res = evaluate(ridge, subject=subject, feature_set=fs, protocol=protocol)
        s = res.summary
        keys = list(s.index) if hasattr(s, "index") else list(s)
        mse = s["mse"] if "mse" in keys else "no 'mse' key!"
        print(f"      summary keys: {keys}")
        return f"mse={mse}"
    return _f


check("import evaluation + data", imports)
if ok:
    for subj in ["S1", "S2"]:
        check(f"{subj} all kfold", run(subj, "all", "kfold"))
        check(f"{subj} all lodo", run(subj, "all", "lodo"))
    for fs in ["step+emg", "step", "emg"]:
        check(f"S1 {fs} lodo", run("S1", fs, "lodo"))
print("\nALL GOOD - Person B can plug in." if ok else "\nSOMETHING FAILED - send this output to Person A.")