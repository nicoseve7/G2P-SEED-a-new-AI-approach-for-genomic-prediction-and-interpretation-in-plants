# -*- coding: utf-8 -*-

################################################################################
### D11c_plot_observed_vs_predicted_harvest.py
### Clean thesis-style observed vs predicted plot for Harvest_date
################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from math import sqrt

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
base_dir = os.path.join(
    "02_harvest_date",
    "06_deep_learning_baseline",
    "output",
    "DeepLearning_Harvest"
)

pred_file = os.path.join(
    base_dir,
    "Predictions",
    "Harvest_date_predictions_formatted.csv"
)

plot_dir = os.path.join(
    base_dir,
    "Plots"
)

os.makedirs(plot_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------------------------
pred = pd.read_csv(pred_file)

# use only test rows
pred_test = pred[pred["Testing"] == 1].copy()

y_true = pred_test["Observed"].astype(float).values
y_pred = pred_test["Predicted"].astype(float).values

# ------------------------------------------------------------------------------
# Global metrics on all test rows
# ------------------------------------------------------------------------------
rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
r = pearsonr(y_true, y_pred)[0]

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 8))

plt.scatter(
    y_true,
    y_pred,
    alpha=0.18,
    s=28
)

mn = min(y_true.min(), y_pred.min())
mx = max(y_true.max(), y_pred.max())

plt.plot([mn, mx], [mn, mx], linewidth=2)

plt.xlabel("Observed Harvest_date")
plt.ylabel("Predicted Harvest_date")
plt.title("Observed vs Predicted on test rows")

text = (
    f"RMSE = {rmse:.2f}\n"
    f"MAE = {mae:.2f}\n"
    f"R² = {r2:.3f}\n"
    f"Pearson r = {r:.3f}\n"
    f"n = {len(pred_test)}"
)

plt.text(
    0.04, 0.96, text,
    transform=plt.gca().transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85)
)

plt.tight_layout()

out_file = os.path.join(plot_dir, "observed_vs_predicted_test_clean.png")
plt.savefig(out_file, dpi=250, bbox_inches="tight")
plt.close()

print("Saved:")
print(out_file)
print()
print(f"RMSE = {rmse:.4f}")
print(f"MAE = {mae:.4f}")
print(f"R² = {r2:.4f}")
print(f"Pearson r = {r:.4f}")
print(f"n = {len(pred_test)}")
