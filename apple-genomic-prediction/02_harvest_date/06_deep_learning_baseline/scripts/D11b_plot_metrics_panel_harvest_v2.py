# -*- coding: utf-8 -*-

################################################################################
### D11b_plot_metrics_panel_harvest_v2.py
### Clean 2x2 panel plot of D10 metrics across 25 splits for Harvest_date
################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
base_dir = "Output/DeepLearning_Harvest"
metrics_file = os.path.join(base_dir, "Harvest_date_metrics_splits.csv")
plot_dir = os.path.join(base_dir, "Plots")
os.makedirs(plot_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Load metrics
# ------------------------------------------------------------------------------
metrics = pd.read_csv(metrics_file)
metrics_splits = metrics[metrics["Split"] != "Mean"].copy().reset_index(drop=True)

x = np.arange(1, len(metrics_splits) + 1)

# ------------------------------------------------------------------------------
# Metric specs
# ------------------------------------------------------------------------------
metric_specs = [
    ("RMSE", "RMSE", "tab:blue"),
    ("MAE", "MAE", "tab:green"),
    ("r2", "R²", "tab:orange"),
    ("r", "Pearson r", "tab:purple"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (col, title, color) in zip(axes, metric_specs):
    y = metrics_splits[col].astype(float).values

    mean_val = np.mean(y)
    sd_val = np.std(y, ddof=1)

    # shaded band
    ax.axhspan(
        mean_val - sd_val,
        mean_val + sd_val,
        alpha=0.15,
        color=color,
        label=f"±1 SD = {sd_val:.3f}"
    )

    # mean line
    ax.axhline(
        mean_val,
        linestyle="--",
        linewidth=2,
        color=color,
        label=f"Mean = {mean_val:.3f}"
    )

    # points
    ax.scatter(
        x, y,
        s=55,
        color=color,
        edgecolor="black",
        linewidth=0.5,
        label="Splits",
        zorder=3
    )

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel(title)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.grid(True, alpha=0.25)

    # cleaner legend order
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)
    ax.legend(uniq_h, uniq_l, loc="best", fontsize=9)

fig.suptitle("Harvest_date deep learning test metrics across 25 splits", fontsize=20, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_file = os.path.join(plot_dir, "metrics_panel_25_splits_v2.png")
plt.savefig(out_file, dpi=220, bbox_inches="tight")
plt.close()

print("Saved:")
print(out_file)

# Also save a small lookup table split number -> split name
lookup = pd.DataFrame({
    "Split_number": x,
    "Split_name": metrics_splits["Split"].tolist()
})
lookup_file = os.path.join(plot_dir, "metrics_panel_25_splits_lookup.csv")
lookup.to_csv(lookup_file, index=False)

print("Saved lookup:")
print(lookup_file)