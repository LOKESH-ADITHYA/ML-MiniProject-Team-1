# Predicting Metabolic Cost During Human-in-the-Loop Optimization

UE24CS352A Machine Learning mini-project.
**Team:** Lokesh Adithya (Track A) and Sanjana (Track B)

Reproduction and extension of *Predicting Metabolic Cost During Human-in-the-Loop Optimization*
(Krimsky and Ng, Stanford CS229, 2018).

## Problem

Ankle-exoskeleton settings are tuned to minimise the wearer's metabolic cost (energy use), but
metabolic cost is slow, noisy and needs a lab and a face mask. We predict the normalised metabolic
cost from easier signals: muscle activity (EMG), step features and the exoskeleton's control
parameters. It is a small-data regression problem: 29 features, about 180 rows per subject, 2 subjects.

## Data

`data/processed_data.mat` from the authors' repository
(https://github.com/ngeley/cs229project). Please cite the paper and repository.

- 2 subjects (Subject 1 = `michael`, Subject 2 = `eley`), 180 rows each, 29 features:
  9 step features, 16 EMG channels, 4 control parameters.
- Rows are in time order: 5 days x 36 two-minute blocks per subject, so the day ID is
  row index // 36.
- One Subject 2 row has a negative target and is removed (179 rows left).
- Only the processed features are available; the raw signals are not.

## Approach

1. **Reproduce** the paper: Lasso baseline, one-hidden-layer tanh network (1-10 neurons), four feature
   sets (all, step+EMG, step only, EMG only).
2. **Evaluate honestly.** The paper uses random k-fold, but neighbouring blocks are nearly identical.
   We add leave-one-day-out (LODO), nested CV and a cross-subject test, and report both.
3. **Improve:** tuned models, a wider model set (linear, kernel, trees, networks), target-aware feature
   selection, engineered features, and analysis of which signals matter.

| Protocol | Meaning |
|---|---|
| kfold (P1) | paper-style random k-fold |
| lodo (P2) | train on 4 days, test on the unseen day (honest test) |
| nested (P3) | LODO outside, tuning inside |
| cross_subject (P4) | train on one subject, test on the other |
| forward (P5) | train on earlier days, test on the next |

## Setup

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Run

    python src/data.py                          # dataset sanity summary
    pytest -q                                   # sanity tests
    python experiments/run_track_a.py           # Track A experiments -> results/track_a/*.csv, results/figures/*.png (~10 min)
    python experiments/run_track_a.py e1 e5     # just some experiments; add --quick for a fast check
    python experiments/check_integration.py     # checks the Track A harness is usable by Track B
    python experiments/run_track_b.py           # Track B experiments -> results/track_b/*.csv, results/figures/*.png (~10 min)
    python experiments/run_track_b.py b1 b3     # b1 NN kfold, b2 NN LODO, b3 trees, b4 extended MLP
    python experiments/run_track_b.py --quick   # fast smoke test

Notebooks: `01_eda` and `03_honest_evaluation` (Track A, Lokesh Adithya);
`02_reproduction` and `04_improvements` (Track B, Sanjana).

## Using the evaluation harness (for every model, both tracks)

    from evaluation import evaluate, tuned
    res = evaluate(make_model, subject="S1", feature_set="all", protocol="lodo")
    res.summary   # mse, rmse, mae, r2, skill vs mean predictor, spearman_day, top5_in_top10

`make_model` is a zero-argument function that returns a fresh estimator with fit/predict.
Put scaling and selection inside it (a Pipeline) so nothing leaks from test rows.

Protocols: kfold (P1, paper-style), lodo (P2), nested (P3), forward (P5). P4 is `cross_subject(...)`.

For nested, wrap the model as `tuned(estimator, grid)`, or give your own tuner `needs_groups = True`
and a `fit(X, y, groups=...)` that runs its inner leave-one-day-out. The harness passes the training
days as groups.

`feature_set` is `"all" | "step+emg" | "step" | "emg"` or a list of columns. Pass `df=` to use
engineered columns.

## Code map

| File | Owner | Contents |
|---|---|---|
| src/data.py | Lokesh Adithya | loader, bad-row removal, day / row_in_day IDs, feature groups |
| src/evaluation.py | Lokesh Adithya | protocols P1-P5, metrics, ranking metric, nested-tuning wrapper |
| src/models_classical.py | Lokesh Adithya | Mean, Lasso, Ridge, ElasticNet, PLS, BayesianRidge, ARD, SVR, KernelRidge, GP (+ tuning grids) |
| src/features.py | Lokesh Adithya | engineered features, PCA / forward / MI / RFE / stability / PLS / ARD selectors |
| experiments/run_track_a.py | Lokesh Adithya | experiments E1a, E2, E3a, E5, E6, E7a |
| src/models_nn.py | Sanjana | paper-style tanh network and extended MLP (seed-averaged, L-BFGS + L2) |
| src/models_trees.py | Sanjana | Random Forest, Extra Trees, Gradient Boosting |
| src/analysis.py | Sanjana | neuron-sweep plots, comparison with the paper's Table I, leaderboard |
| experiments/run_track_b.py | Sanjana | experiments b1-b4 |
| experiments/check_integration.py | Sanjana | checks the Track A harness works for Track B |
| tests/test_models_b.py | Sanjana | sanity tests for the Track B models |
| src/tuning.py, ensemble.py | Sanjana | Optuna tuning and ensembles (in progress) |

## Track B results so far (MSE, lower is better)

Default settings, not yet tuned. Paper-style test = random 10-fold; honest test = leave-one-day-out.

| | Paper's best | Ours, paper-style test | Ours, honest test |
|---|---|---|---|
| Subject 1, all features | 0.0089 | 0.0086 (extended MLP), 0.0089 (tanh net) | 0.0163 (extended MLP), 0.0176 (tanh net) |
| Subject 2, all features | 0.0176 | 0.0106 (Extra Trees), 0.0115 (tanh net) | 0.0251 (Gradient Boosting) |

Findings so far: the paper's trends reproduce (EMG beats step features; networks stop improving after
about 3-5 neurons). Random k-fold is clearly optimistic: honest errors are roughly double. Control
parameters help more on the honest test than the paper suggests. Subject 2 comparisons with the paper
are not exact because we drop one negative-target row.

## Notes and limitations

- Python has no MATLAB-style Bayesian Regularization, so the network uses full-batch L-BFGS with an L2
  weight penalty (`alpha`) as the stand-in. Defaults are starting values; tuning is done inside nested CV.
- Only 2 subjects and about 180 rows each, so conclusions about generalisation are limited.

## Reference

Krimsky, E. and Ng, E. *Predicting Metabolic Cost During Human-in-the-Loop Optimization.* Stanford CS229,
2018. Code and data: https://github.com/ngeley/cs229project