"""Sanity tests for Track A (run: pytest -q)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import FEATURE_SETS, load_subject  # noqa: E402
from evaluation import evaluate, make_folds, ranking_metrics, tuned  # noqa: E402
from features import engineer, make_selector  # noqa: E402
from models_classical import MODELS  # noqa: E402


def test_shapes_and_cleaning():
    s1, s2 = load_subject("S1"), load_subject("S2")
    assert len(s1) == 180 and len(s2) == 179
    assert (s2["metabolic"] >= 0).all()
    assert s1["day"].tolist() == sorted(s1["day"].tolist())
    assert s1.groupby("day").size().eq(36).all()
    assert {k: len(v) for k, v in FEATURE_SETS.items()} == {"all": 29, "step+emg": 25, "step": 9, "emg": 16}


def test_lodo_never_mixes_days():
    days = load_subject("S1")["day"].to_numpy()
    folds = list(make_folds(days, "lodo"))
    assert len(folds) == 5
    for _, _, tr, te in folds:
        assert len(set(days[te])) == 1 and days[te][0] not in set(days[tr])


def test_forward_chaining_respects_time():
    days = load_subject("S1")["day"].to_numpy()
    for _, _, tr, te in make_folds(days, "forward"):
        assert days[tr].max() < days[te].min()


def test_mean_predictor_has_zero_skill():
    r = evaluate(MODELS["Mean"], "S1", "all", "lodo").summary
    assert r["skill"] == pytest.approx(0.0)


def test_ridge_beats_mean():
    r = evaluate(MODELS["Ridge"], "S1", "all", "lodo").summary
    assert r["mse"] < 0.5 * r["mean_mse"]


def test_nested_tuning_gets_groups():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    make = lambda: tuned(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [1, 10]})
    r = evaluate(make, "S1", "all", "nested")
    assert len(r.extra["best_params"]) == 5


def test_ranking_metric_perfect():
    y = np.arange(36.0)
    out = ranking_metrics(y, y, np.ones(36))
    assert out["spearman_day"] == pytest.approx(1.0) and out["top5_in_top10"] == 1.0


def test_engineering_and_selector_run():
    df, cols = engineer(load_subject("S1"))
    assert len(cols) == 25 + 8 + 8 + 4
    r = evaluate(make_selector("rfe", 5), "S1", "step+emg", "lodo").summary
    assert np.isfinite(r["mse"])
