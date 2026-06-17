# -*- coding: utf-8 -*-
"""Per-dataset box plots of optimality gap (%) per method (2x2 small multiples).

Reads test_results/data.xlsx and produces a paper-quality figure: one small box
plot per benchmark dataset (2x2), each on its own y-axis so every dataset is
shown at its natural scale, with jittered per-instance points. Four dispatching
rules (FIFO/MOR/MWKR/SPT, shown in greys as baselines) are compared against the
learning methods HGNN/DANIEL/LMLPF and Ours.

Gap is computed as (makespan-UB)/UB because the spreadsheet's gap columns hold
Excel formulas rather than numeric values.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XLSX = os.path.join(ROOT, "test_results", "data.xlsx")

# left-to-right order; method -> (makespan column index, color)
# dispatching rules use distinct colours kept clear of the learning palette;
# learning palette matches plot_ablation_curves.py
UB_COL = 2
METHODS = {
    "SPT":    (9,  "#ff7f0e"),  # orange
    "FIFO":   (3,  "#17becf"),  # cyan
    "MOR":    (5,  "#8c564b"),  # brown
    "MWKR":   (7,  "#e377c2"),  # pink
    "HGNN":   (11, "#1f77b4"),  # blue
    "DANIEL": (13, "#2ca02c"),  # green
    "LMLPF":  (15, "#9467bd"),  # purple
    "Ours":   (17, "#d62728"),  # red
}
# dataset key in file -> display label
DATASETS = {
    "Brandimarte":  "Brandimarte",
    "Hurink_rdata": "Hurink-rdata",
    "Hurink_edata": "Hurink-edata",
    "Hurink_vdata": "Hurink-vdata",
}


def load():
    """Return {dataset: {method: [gap%, ...]}}, summary rows excluded."""
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    out = {d: {m: [] for m in METHODS} for d in DATASETS}
    for r in rows[2:]:                       # skip the two header rows
        ds = r[0]
        if ds not in DATASETS:               # skips 'Avg. Gap' and blanks
            continue
        ub = r[UB_COL]
        if not isinstance(ub, (int, float)) or not ub:
            continue
        for m, (col, _) in METHODS.items():
            ms = r[col]
            if isinstance(ms, (int, float)):
                out[ds][m].append((ms - ub) / ub * 100.0)
    return out


plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

data = load()
methods = list(METHODS)
datasets = list(DATASETS)
n_m = len(methods)
box_w = 0.62

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
axes = axes.ravel()

for ax, ds in zip(axes, datasets):
    rng = np.random.default_rng(0)
    for mi, m in enumerate(methods):
        vals = np.asarray(data[ds][m], dtype=float)
        color = METHODS[m][1]
        ax.boxplot(
            vals, positions=[mi], widths=box_w,
            patch_artist=True, showfliers=False, manage_ticks=False,
            medianprops=dict(color="black", lw=1.2),
            whiskerprops=dict(color=color, lw=1.0),
            capprops=dict(color=color, lw=1.0),
            boxprops=dict(facecolor=color, edgecolor=color, alpha=0.40, lw=1.0),
        )
        jit = rng.uniform(-box_w * 0.30, box_w * 0.30, size=vals.size)
        ax.scatter(mi + jit, vals, s=8, color=color, alpha=0.60,
                   edgecolors="none", zorder=3)

    n_inst = len(data[ds][methods[0]])
    ax.set_title(f"{DATASETS[ds]}  (n={n_inst})", fontsize=10)
    ax.set_xticks(range(n_m))
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(bottom=-2)
    ax.set_ylabel("Optimality gap (%)")
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
    ax.margins(x=0.06)

handles = [Patch(facecolor=METHODS[m][1], edgecolor=METHODS[m][1],
                 alpha=0.6, label=m) for m in methods]
fig.legend(handles=handles, loc="upper center", ncol=8,
           bbox_to_anchor=(0.5, 1.05), columnspacing=1.1, handlelength=1.1)

for ext in ("pdf", "png"):
    out = os.path.join(HERE, f"gap_boxplot.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)
