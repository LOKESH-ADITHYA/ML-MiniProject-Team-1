
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed (Codespaces)
import matplotlib.pyplot as plt
import pandas as pd

NOISE_FLOOR = 0.0025  # paper's minimum achievable MSE

# Paper Table I.  (linear regression MSE, neural-net MSE) per feature set.
PAPER_TABLE1 = {
    "S1": {"const": 0.0657, "all": (0.0089, 0.0089), "step+emg": (0.0089, 0.0104),
           "step": (0.0200, 0.0138), "emg": (0.0158, 0.0118)},
    "S2": {"const": 0.0465, "all": (0.0176, 0.0301), "step+emg": (0.0168, 0.0313),
           "step": (0.0217, 0.0233), "emg": (0.0203, 0.0199)},
}
FEATURE_SETS = ["all", "step+emg", "step", "emg"]
FS_TITLE = {"all": "All features", "step+emg": "Step + EMG",
            "step": "Step only", "emg": "EMG only"}


def load_results(path):
    return pd.read_csv(path)


def plot_neuron_sweep(df, protocol, out_path):
    """Figure-1-style plot: MSE vs number of hidden neurons, 4 panels."""
    d = df[(df["protocol"] == protocol) & (df["model"] == "paper_nn")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, fs in zip(axes.ravel(), FEATURE_SETS):
        for subj, color in [("S1", "tab:blue"), ("S2", "tab:red")]:
            s = d[(d["feature_set"] == fs) & (d["subject"] == subj)].sort_values("n_hidden")
            if len(s):
                ax.plot(s["n_hidden"], s["mse"], "o-", color=color, label=subj)
        ax.axhline(NOISE_FLOOR, color="k", ls=":", lw=1, label="noise floor")
        ax.set_title(FS_TITLE[fs])
        ax.set_xlabel("hidden neurons")
        ax.set_ylabel("MSE")
        ax.legend()
    fig.suptitle(f"Paper-style tanh network, protocol = {protocol}")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def best_nn_vs_paper(df, protocol):
    """For each subject x feature set: our best-neuron NN vs the paper's NN."""
    d = df[(df["protocol"] == protocol) & (df["model"] == "paper_nn")]
    rows = []
    for subj in ["S1", "S2"]:
        for fs in FEATURE_SETS:
            s = d[(d["subject"] == subj) & (d["feature_set"] == fs)]
            if s.empty:
                continue
            best = s.loc[s["mse"].idxmin()]
            lin, nn = PAPER_TABLE1[subj][fs]
            rows.append({"subject": subj, "feature_set": fs, "protocol": protocol,
                         "our_best_mse": round(best["mse"], 4),
                         "our_best_neurons": int(best["n_hidden"]),
                         "paper_nn_mse": nn, "paper_linreg_mse": lin,
                         "paper_const_mse": PAPER_TABLE1[subj]["const"]})
    return pd.DataFrame(rows)


def leaderboard(df, protocol):
    """Best MSE per model family, per subject x feature set."""
    d = df[df["protocol"] == protocol]
    g = d.groupby(["subject", "feature_set", "model"])["mse"].min().reset_index()
    return g.pivot_table(index=["subject", "feature_set"], columns="model",
                         values="mse").round(4)