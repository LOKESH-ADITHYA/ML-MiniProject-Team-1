# Predicting Metabolic Cost During Human-in-the-Loop Optimization
UE24CS352A ML mini-project. Team: NAME_A and NAME_B

Data: processed_data.mat from https://github.com/ngeley/cs229project
Paper: Krimsky and Ng, Stanford CS229, 2018.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Run
```
python src/data.py                          # dataset sanity summary
python experiments/run_track_a.py           # Track A experiments -> results/track_a/*.csv, results/figures/*.png (~10 min)
python experiments/run_track_a.py e1 e5     # just some experiments; add --quick for a fast check
pytest -q                                   # sanity tests
```
Notebooks: `01_eda` and `03_honest_evaluation` (Track A); `02_reproduction` and `04_improvements` (Track B).

## Using the evaluation harness (for every model, both tracks)
```python
from evaluation import evaluate, tuned
res = evaluate(make_model, subject="S1", feature_set="all", protocol="lodo")
res.summary   # mse, rmse, mae, r2, skill vs mean predictor, spearman_day, top5_in_top10
```
- `make_model` is a zero-argument function that returns a fresh estimator with `fit`/`predict`. Put scaling and selection inside it (a Pipeline).
- Protocols: `kfold` (P1, paper-style), `lodo` (P2), `nested` (P3), `forward` (P5). P4 is `cross_subject(...)`.
- For `nested`, wrap the model as `tuned(estimator, grid)`, or give your own tuner `needs_groups = True` and a `fit(X, y, groups=...)` that runs its inner leave-one-day-out. The harness passes the training days as `groups`.
- `feature_set` is `"all" | "step+emg" | "step" | "emg"` or a list of columns. Pass `df=` to use engineered columns.

## Code map
| File | Owner | Contents |
|---|---|---|
| src/data.py | A | loader, bad-row removal, day / row_in_day IDs, feature groups |
| src/evaluation.py | A | protocols P1–P5, metrics, ranking metric, nested-tuning wrapper |
| src/models_classical.py | A | Mean, Lasso, Ridge, ElasticNet, PLS, BayesianRidge, ARD, SVR, KernelRidge, GP (+ tuning grids) |
| src/features.py | A | engineered features, PCA / forward / MI / RFE / stability / PLS / ARD selectors |
| experiments/run_track_a.py | A | experiments E1a, E2, E3a, E5, E6, E7a |
| src/models_nn.py, tuning.py, ensemble.py, analysis.py | B | neural nets, Optuna, ensembles, SHAP / learning curves |

