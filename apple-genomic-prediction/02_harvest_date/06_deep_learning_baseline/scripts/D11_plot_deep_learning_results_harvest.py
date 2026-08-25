# -*- coding: utf-8 -*-

################################################################################
### D11_plot_deep_learning_results_harvest.py
### Analisi e grafici dei risultati deep learning per Harvest_date
################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
base_dir = os.path.join(
    "02_harvest_date",
    "06_deep_learning_baseline",
    "output",
    "DeepLearning_Harvest"
)
metrics_file = os.path.join(base_dir, "Harvest_date_metrics_splits.csv")
pred_file = os.path.join(base_dir, "Predictions", "Harvest_date_predictions_formatted.csv")

plot_dir = os.path.join(base_dir, "Plots")
os.makedirs(plot_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
metrics = pd.read_csv(metrics_file)
pred = pd.read_csv(pred_file)

# remove final Mean row for per-split plots
metrics_splits = metrics[metrics["Split"] != "Mean"].copy()

# ------------------------------------------------------------------------------
# Helper columns
# ------------------------------------------------------------------------------
if "Run" not in pred.columns and "CV" in pred.columns:
    pred["Run"] = pred["CV"].str.extract(r"CV(\d)_Split")[0]
    pred["Fold"] = pred["CV"].str.extract(r"Split(\d)")[0]

if "Run" in pred.columns:
    pred["Run"] = pd.to_numeric(pred["Run"], errors="coerce")
if "Fold" in pred.columns:
    pred["Fold"] = pd.to_numeric(pred["Fold"], errors="coerce")

# use only test rows for evaluation plots
pred_test = pred[pred["Testing"] == 1].copy()

# residuals
pred_test["Residual"] = pred_test["Observed"] - pred_test["Predicted"]
pred_test["AbsResidual"] = pred_test["Residual"].abs()

# ------------------------------------------------------------------------------
# Save a simple summary text
# ------------------------------------------------------------------------------
summary_txt = os.path.join(plot_dir, "deep_learning_summary_harvest.txt")
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("=== DEEP LEARNING HARVEST_DATE SUMMARY ===\n\n")
    f.write("Metrics file: {}\n".format(metrics_file))
    f.write("Predictions file: {}\n\n".format(pred_file))

    f.write("Mean metrics from D10:\n")
    mean_row = metrics[metrics["Split"] == "Mean"]
    if len(mean_row) == 1:
        f.write(mean_row.to_string(index=False))
        f.write("\n\n")

    f.write("Per-split metrics summary:\n")
    f.write(metrics_splits.describe(include="all").to_string())
    f.write("\n\n")

    f.write("Test rows in formatted prediction file: {}\n".format(len(pred_test)))
    f.write("Unique environments: {}\n".format(pred_test["Envir"].nunique()))
    f.write("Unique genotypes: {}\n".format(pred_test["Genotype"].nunique()))

# ------------------------------------------------------------------------------
# 1. Boxplot metrics across splits
# ------------------------------------------------------------------------------
for col in ["RMSE", "MAE", "r2", "r"]:
    plt.figure(figsize=(6, 5))
    plt.boxplot(metrics_splits[col].dropna())
    plt.ylabel(col)
    plt.title(f"{col} across 25 splits")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"boxplot_{col}.png"), dpi=200)
    plt.close()

# ------------------------------------------------------------------------------
# 2. Metrics by split (line plot)
# ------------------------------------------------------------------------------
for col in ["RMSE", "MAE", "r2", "r"]:
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_splits["Split"], metrics_splits[col], marker="o")
    plt.xticks(rotation=90)
    plt.ylabel(col)
    plt.title(f"{col} by split")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"line_{col}_by_split.png"), dpi=200)
    plt.close()

# ------------------------------------------------------------------------------
# 3. Observed vs Predicted on test rows
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(pred_test["Observed"], pred_test["Predicted"], alpha=0.35)
mn = min(pred_test["Observed"].min(), pred_test["Predicted"].min())
mx = max(pred_test["Observed"].max(), pred_test["Predicted"].max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.title("Observed vs Predicted (test rows)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "scatter_observed_vs_predicted_test.png"), dpi=200)
plt.close()

# ------------------------------------------------------------------------------
# 4. Residual histogram
# ------------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.hist(pred_test["Residual"], bins=40)
plt.xlabel("Residual (Observed - Predicted)")
plt.ylabel("Count")
plt.title("Residual distribution on test rows")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "hist_residuals_test.png"), dpi=200)
plt.close()

# ------------------------------------------------------------------------------
# 5. Absolute residuals by environment
# ------------------------------------------------------------------------------
env_perf = (
    pred_test.groupby("Envir")
    .agg(
        n=("Observed", "size"),
        RMSE=("Residual", lambda x: np.sqrt(np.mean(np.square(x)))),
        MAE=("AbsResidual", "mean"),
        MeanObserved=("Observed", "mean"),
        MeanPredicted=("Predicted", "mean"),
    )
    .reset_index()
    .sort_values("RMSE", ascending=False)
)

env_perf.to_csv(os.path.join(plot_dir, "performance_by_environment.csv"), index=False)

plt.figure(figsize=(10, 6))
plt.bar(env_perf["Envir"], env_perf["RMSE"])
plt.xticks(rotation=90)
plt.ylabel("RMSE")
plt.title("RMSE by environment (test rows)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "bar_rmse_by_environment.png"), dpi=200)
plt.close()

# ------------------------------------------------------------------------------
# 6. Metrics by run and by fold
# ------------------------------------------------------------------------------
metrics_splits["Run"] = metrics_splits["Split"].str.extract(r"CV(\d)_Split")[0].astype(int)
metrics_splits["Fold"] = metrics_splits["Split"].str.extract(r"Split(\d)")[0].astype(int)

run_summary = metrics_splits.groupby("Run")[["RMSE", "MAE", "r2", "r"]].mean().reset_index()
fold_summary = metrics_splits.groupby("Fold")[["RMSE", "MAE", "r2", "r"]].mean().reset_index()

run_summary.to_csv(os.path.join(plot_dir, "metrics_by_run.csv"), index=False)
fold_summary.to_csv(os.path.join(plot_dir, "metrics_by_fold.csv"), index=False)

for df_plot, xcol, fname, title in [
    (run_summary, "Run", "rmse_by_run.png", "Mean RMSE by run"),
    (fold_summary, "Fold", "rmse_by_fold.png", "Mean RMSE by fold"),
]:
    plt.figure(figsize=(6, 5))
    plt.bar(df_plot[xcol].astype(str), df_plot["RMSE"])
    plt.xlabel(xcol)
    plt.ylabel("RMSE")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, fname), dpi=200)
    plt.close()

# ------------------------------------------------------------------------------
# 7. Per-run observed vs predicted summary
# ------------------------------------------------------------------------------
run_pred_summary = (
    pred_test.groupby("Run")
    .agg(
        MeanObserved=("Observed", "mean"),
        MeanPredicted=("Predicted", "mean"),
        MAE=("AbsResidual", "mean"),
        RMSE=("Residual", lambda x: np.sqrt(np.mean(np.square(x))))
    )
    .reset_index()
)

run_pred_summary.to_csv(os.path.join(plot_dir, "prediction_summary_by_run.csv"), index=False)

print("Plots and summaries saved in:")
print(plot_dir)
