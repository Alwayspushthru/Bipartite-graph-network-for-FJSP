# -*- coding: utf-8 -*-
"""Plot ablation validation-makespan curves for the FJSP bipartite-graph network.

Reads the four single-seed TensorBoard-exported CSVs from runs/exp and produces
a paper-quality figure: full-range convergence on the main axes plus a zoomed
inset on the late-training band where the variants separate.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "runs", "exp")

# label -> (csv file, color, linestyle)
CURVES = {
    "Full (ours)":      ("run-exp_Full_vali_makespan_mean.csv",            "#d62728", "-"),
    "Mean-Agg":         ("run-exp_meanAgg_vali_makespan_mean.csv",         "#1f77b4", "--"),
    "w/o Pair-Bias":    ("run-exp_NopairBias-train_vali_makespan_mean.csv", "#2ca02c", "-."),
    "w/o GRU":          ("run-exp_NoGRU_vali_makespan_mean.csv",           "#9467bd", ":"),
}


def load(path):
    steps, vals = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["Step"]))
            vals.append(float(row["Value"]))
    return np.asarray(steps), np.asarray(vals)


def ema(x, alpha=0.6):
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * x[i]
    return out


plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

fig, ax = plt.subplots(figsize=(7.0, 4.3))

data = {}
for label, (fname, color, ls) in CURVES.items():
    s, v = load(os.path.join(EXP, fname))
    data[label] = (s, v, color, ls)
    sm = ema(v)
    ax.plot(s, v, color=color, lw=0.8, alpha=0.22)            # raw (faint, honest)
    ax.plot(s, sm, color=color, ls=ls, lw=1.8, label=label)   # smoothed

ax.set_xlabel("Training step")
ax.set_ylabel("Validation makespan (mean)")
ax.set_xlim(0, 1000)
ax.set_ylim(490, 1110)
ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99))
ax.grid(True, ls=":", lw=0.5, alpha=0.5)

# ---- zoomed inset on the late-training band where variants separate ----
axins = ax.inset_axes([0.40, 0.45, 0.56, 0.46])
for label, (s, v, color, ls) in data.items():
    sm = ema(v)
    axins.plot(s, v, color=color, lw=0.7, alpha=0.22)
    axins.plot(s, sm, color=color, ls=ls, lw=1.6)
axins.set_xlim(200, 1000)
axins.set_ylim(505, 525)
axins.grid(True, ls=":", lw=0.5, alpha=0.5)
axins.tick_params(labelsize=8)
axins.set_title("zoom: steps 200-1000", fontsize=8, pad=2)

# guide lines connecting the zoom region to the inset
ax.indicate_inset_zoom(axins, edgecolor="gray", lw=0.8, alpha=0.7)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(HERE, f"ablation_vali_makespan.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)
