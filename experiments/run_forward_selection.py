"""Select a common model recipe for predicting FUTURE DAYS of the same subjects.

Primary protocol: expanding-window / forward-chaining.
For each subject independently:
  train days 1-2 -> test day 3
  train days 1-3 -> test day 4
  train days 1-4 -> test day 5

Every candidate uses the same folds and the same feature set. Hyperparameters are
fixed in advance (no tuning on the three outer test days), which is appropriate
for this very small dataset. Scaling, where used, is fitted inside each fold.

Run from the repository root:
    python experiments/run_forward_selection.py

Outputs:
    results/track_c/forward_predictions.csv
    results/track_c/forward_fold_metrics.csv
    results/track_c/forward_day_metrics.csv
    results/track_c/forward_model_leaderboard.csv
    results/track_c/forward_feature_check.csv
"""
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from data import SUBJECTS, get_xy, load_subject

OUT = ROOT / "results" / "track_c"
OUT.mkdir(parents=True, exist_ok=True)

# Fixed, intentionally modest shortlist. Do not tune these using test-day results.
# The same model recipe is evaluated on both subjects, but fitted separately for
# each subject because the stated goal is future-day prediction for known subjects.
MODELS = {
    "Mean": lambda: DummyRegressor(strategy="mean"),
    "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "ElasticNet": lambda: make_pipeline(
        StandardScaler(),
        ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=50000,
            random_state=0
        )
    ),
    "Lasso": lambda: make_pipeline(
        StandardScaler(), Lasso(alpha=0.01, max_iter=50000)
    ),
    "BayesianRidge": lambda: make_pipeline(
        StandardScaler(), BayesianRidge()
    ),
    "PLS": lambda: make_pipeline(
        StandardScaler(), PLSRegression(n_components=4, scale=False)
    ),
    "SVR": lambda: make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.05)
    ),
    "ExtraTrees": lambda: ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=0,
        n_jobs=-1
    ),
    "GradientBoosting": lambda: GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=2,
        loss="squared_error",
        random_state=0
    ),
}


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def forward_splits(days):
    """Yield expanding-window splits; never train on the test day or a later day."""
    days = np.asarray(days)
    unique_days = sorted(np.unique(days).tolist())

    # With five days this produces test days 3, 4, and 5.
    for test_day in unique_days[2:]:
        train_idx = np.flatnonzero(days < test_day)
        test_idx = np.flatnonzero(days == test_day)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        assert np.max(days[train_idx]) < test_day
        yield test_day, train_idx, test_idx


def evaluate_subject(subject, feature_set="all"):
    df = load_subject(subject)
    X_df, y, days = get_xy(df, feature_set)

    X = X_df.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    days = np.asarray(days)

    prediction_rows = []
    fold_rows = []

    for model_name, factory in MODELS.items():
        for test_day, tr, te in forward_splits(days):
            model = factory()
            model.fit(X[tr], y[tr])

            pred = np.asarray(
                model.predict(X[te]), dtype=float
            ).reshape(-1)

            # Baseline uses only the training fold's historical target mean.
            baseline_pred = np.full(len(te), float(np.mean(y[tr])))

            model_scores = regression_metrics(y[te], pred)
            base_scores = regression_metrics(y[te], baseline_pred)

            fold_rows.append({
                "subject": subject,
                "features": feature_set,
                "model": model_name,
                "test_day": int(test_day),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                **model_scores,
                "baseline_mse": base_scores["mse"],
                "skill_vs_training_mean": (
                    float(1 - model_scores["mse"] / base_scores["mse"])
                    if base_scores["mse"] > 0 else np.nan
                ),
            })

            for local_i, row_i in enumerate(te):
                prediction_rows.append({
                    "subject": subject,
                    "features": feature_set,
                    "model": model_name,
                    "test_day": int(days[row_i]),
                    "row_index": int(row_i),
                    "y_true": float(y[row_i]),
                    "y_pred": float(pred[local_i]),
                    "baseline_pred": float(baseline_pred[local_i]),
                })

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)

    # Aggregate genuinely out-of-sample predictions from days 3-5.
    subject_rows = []
    day_rows = []

    for model_name in MODELS:
        p = predictions[predictions.model == model_name]

        scores = regression_metrics(p.y_true, p.y_pred)
        baseline_scores = regression_metrics(
            p.y_true, p.baseline_pred
        )

        subject_rows.append({
            "features": feature_set,
            "model": model_name,
            "subject": subject,
            **scores,
            "baseline_mse": baseline_scores["mse"],
            "skill_vs_training_mean": (
                float(1 - scores["mse"] / baseline_scores["mse"])
                if baseline_scores["mse"] > 0 else np.nan
            ),
            "n_test_rows": int(len(p)),
            "n_test_days": int(p.test_day.nunique()),
        })

        for day, pd_day in p.groupby("test_day"):
            day_scores = regression_metrics(
                pd_day.y_true, pd_day.y_pred
            )

            day_rows.append({
                "features": feature_set,
                "model": model_name,
                "subject": subject,
                "test_day": int(day),
                "n_test": int(len(pd_day)),
                **day_scores,
            })

    return (
        predictions,
        folds,
        pd.DataFrame(day_rows),
        pd.DataFrame(subject_rows),
    )


def main():
    all_predictions = []
    all_folds = []
    all_days = []
    all_subject_rows = []
    all_leaderboards = []

    for features in ("all", "step+emg"):
        print(f"\nEvaluating feature set: {features}")

        # Model selection is based on "all".
        # "step+emg" is a separate feature-set sensitivity check.
        results = [
            evaluate_subject(subject, features)
            for subject in SUBJECTS
        ]

        preds = pd.concat(
            [result[0] for result in results], ignore_index=True
        )
        folds = pd.concat(
            [result[1] for result in results], ignore_index=True
        )
        days = pd.concat(
            [result[2] for result in results], ignore_index=True
        )
        subject_rows = pd.concat(
            [result[3] for result in results], ignore_index=True
        )

        # Give both subjects equal weight in the shared model-family ranking.
        leaderboard_rows = []

        for model_name in MODELS:
            d = subject_rows[subject_rows.model == model_name]

            leaderboard_rows.append({
                "features": features,
                "model": model_name,
                "avg_subject_mse": float(d.mse.mean()),
                "worst_subject_mse": float(d.mse.max()),
                "avg_subject_rmse": float(d.rmse.mean()),
                "avg_subject_mae": float(d.mae.mean()),
                "avg_skill_vs_training_mean": float(
                    d.skill_vs_training_mean.mean()
                ),
                "S1_mse": float(
                    d.loc[d.subject == "S1", "mse"].iloc[0]
                ),
                "S2_mse": float(
                    d.loc[d.subject == "S2", "mse"].iloc[0]
                ),
                "S1_skill": float(
                    d.loc[
                        d.subject == "S1",
                        "skill_vs_training_mean"
                    ].iloc[0]
                ),
                "S2_skill": float(
                    d.loc[
                        d.subject == "S2",
                        "skill_vs_training_mean"
                    ].iloc[0]
                ),
            })

        leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
            [
                "avg_subject_mse",
                "worst_subject_mse",
                "avg_subject_mae"
            ],
            ascending=True
        ).reset_index(drop=True)

        leaderboard.insert(
            0,
            "selection_scope",
            "primary" if features == "all"
            else "secondary_feature_check"
        )

        all_predictions.append(preds)
        all_folds.append(folds)
        all_days.append(days)
        all_subject_rows.append(subject_rows)
        all_leaderboards.append(leaderboard)

        print(leaderboard.round(4).to_string(index=False))

    pd.concat(
        all_predictions, ignore_index=True
    ).to_csv(OUT / "forward_predictions.csv", index=False)

    pd.concat(
        all_folds, ignore_index=True
    ).to_csv(OUT / "forward_fold_metrics.csv", index=False)

    pd.concat(
        all_days, ignore_index=True
    ).to_csv(OUT / "forward_day_metrics.csv", index=False)

    pd.concat(
        all_subject_rows, ignore_index=True
    ).to_csv(OUT / "forward_subject_metrics.csv", index=False)

    leaderboard_all = pd.concat(
        all_leaderboards, ignore_index=True
    )

    leaderboard_all.to_csv(
        OUT / "forward_model_leaderboard.csv", index=False
    )

    primary = leaderboard_all[
        leaderboard_all.selection_scope == "primary"
    ].sort_values(
        [
            "avg_subject_mse",
            "worst_subject_mse",
            "avg_subject_mae"
        ]
    )

    best = primary.iloc[0]

    feature_check = leaderboard_all[
        [
            "selection_scope",
            "features",
            "model",
            "avg_subject_mse",
            "worst_subject_mse",
            "avg_subject_mae",
            "S1_mse",
            "S2_mse",
        ]
    ]

    feature_check.to_csv(
        OUT / "forward_feature_check.csv", index=False
    )

    print(
        "\nPRIMARY RESULT "
        "(all 29 features; same model recipe applied to both subjects)"
    )
    print(f"Recommended candidate: {best['model']}")
    print(
        f"Subject 1 MSE: {best['S1_mse']:.6f}; "
        f"Subject 2 MSE: {best['S2_mse']:.6f}"
    )
    print(
        "Equal-weight average subject MSE: "
        f"{best['avg_subject_mse']:.6f}"
    )
    print(f"Worst-subject MSE: {best['worst_subject_mse']:.6f}")

    print(
        "\nImportant: only three future-day folds are available per subject. "
        "Treat the top-ranked model as a candidate, not a proven winner."
    )
    print(f"\nSaved results to: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()