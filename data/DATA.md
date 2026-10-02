# Dataset

File: `processed_data.mat` (79 KB), copied unchanged from https://github.com/ngeley/cs229project
(MD5 f386194db18a12532497cead93f791a8). Data collected by the Stanford Biomechatronics Laboratory.
Cite: E. Krimsky, E. Ng, "Predicting Metabolic Cost During Human-in-the-Loop Optimization", CS229, Stanford, 2018.
The source repo has no licence file; used here for coursework with attribution.

| Variable | Shape | Meaning |
|---|---|---|
| michael_data / eley_data | 180 x 29 | features for Subject 1 / Subject 2 |
| michael_metabolics / eley_metabolics | 180 x 1 | target: normalised metabolic rate |
| data_labels | 1 x 29 | column names |

Columns
- Step (9): LFz_peak, L_peak_df, L_peak_pf, L_step_time, RFz_peak, R_peak_df, R_peak_pf, R_step_time, step_width.
  These are per-30 s medians. GRF is normalised by body weight.
- EMG (16): emg1–emg16, the RMS of the filtered and rectified signal, normalised to that day's normal-walking peak.
  Per the authors' code, left = 1–4 and 9–12, right = 5–8 and 13–16. Muscle names are not documented.
- Control (4): ctrl1–ctrl4, the exo torque parameters (peak time, rise time, peak magnitude, settling time).
- Target: metabolic rate minus the standing baseline, divided by that day's normal-walking rate, from the last 30 s of each 2-min condition.

Derived by `src/data.py`
- `day` = row // 36 + 1. Rows are chronological: 5 days x 36 conditions. The paper says 6 days, but the code uses 5.
  Verified because the control setting at every day's first row is identical.
- `row_in_day` = row % 36.
- Subject 2 row with a negative target (≈ −0.57) is dropped, as in the authors' code, leaving 179 rows.
