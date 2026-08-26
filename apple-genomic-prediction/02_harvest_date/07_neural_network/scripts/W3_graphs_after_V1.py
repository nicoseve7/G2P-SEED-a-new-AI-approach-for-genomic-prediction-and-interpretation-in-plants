# -*- coding: utf-8 -*-

################################################################################
### U2_graphs_after_U1.py
### Graphs and diagnostics for
### paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v2
################################################################################

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("Output")

METRICS_FILE = OUT_DIR / "metrics" / "metrics_Harvest_date_paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil.csv"
PRED_FILE = OUT_DIR / "predictions" / "predictions_Harvest_date_paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil_all_splits.csv"
LOSS_DIR = OUT_DIR / "loss_history"

GRAFICI_DIR = OUT_DIR / "grafici"
PER_SPLIT_DIR = GRAFICI_DIR / "per_split"
SUMMARY_DIR = GRAFICI_DIR / "summary"

TRAIT = "Harvest_date"
MODEL_NAME = "senza_suolo"


def make_dirs():
    PER_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def pearson_r(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if len(x) < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def load_metrics():
    df = pd.read_csv(METRICS_FILE)
    df = df[df["Split"] != "Mean"].copy()
    return df


def load_predictions():
    return pd.read_csv(PRED_FILE)


def load_loss_histories():
    files = sorted(LOSS_DIR.glob("loss_history_*.csv"))
    histories = {}
    for f in files:
        split_name = f.stem.replace("loss_history_", "")
        histories[split_name] = pd.read_csv(f)
    return histories


def build_padded_matrix(histories, column_name):
    split_names = sorted(histories.keys())
    max_len = max(len(histories[s]) for s in split_names)

    mat = np.full((len(split_names), max_len), np.nan)

    for i, split_name in enumerate(split_names):
        vals = histories[split_name][column_name].values.astype(float)
        mat[i, :len(vals)] = vals

    return split_names, mat


# def plot_metrics_across_splits(metrics_df):
#     split_ids = np.arange(1, len(metrics_df) + 1)

#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#     fig.suptitle(f"{TRAIT} - {MODEL_NAME} test metrics across 25 splits",
#                  fontsize=24, fontweight="bold", y=0.98)

#     specs = [("RMSE", "RMSE"), ("MAE", "MAE"), ("r2", "R²"), ("r", "Pearson r")]
#     axes = axes.flatten()

#     for ax, (col, title) in zip(axes, specs):
#         values = metrics_df[col].values.astype(float)
#         mean_val = np.mean(values)
#         sd_val = np.std(values, ddof=1)

#         ax.scatter(split_ids, values, s=55, alpha=0.95, edgecolor="#444444", linewidth=0.8, zorder=3)
#         ax.axhline(mean_val, linestyle="--", linewidth=2.2, label=f"Mean = {mean_val:.3f}", zorder=2)
#         ax.axhspan(mean_val - sd_val, mean_val + sd_val, alpha=0.16, label=f"±1 SD = {sd_val:.3f}", zorder=1)

#         ax.set_title(title, fontsize=20, fontweight="bold")
#         ax.set_xlabel("Split")
#         ax.set_ylabel(title)
#         ax.set_xticks(split_ids)
#         ax.grid(True, alpha=0.2)
#         ax.legend(frameon=True)

#     savefig(SUMMARY_DIR / f"{MODEL_NAME}_metrics_across_25_splits.png")

RMSE_DOT = "#1f77b4"
RMSE_LINE = "#1f77b4"
RMSE_BAND = "#1f77b4"

MAE_DOT = "#2ca02c"
MAE_LINE = "#2ca02c"
MAE_BAND = "#2ca02c"

R2_DOT = "#ff7f0e"
R2_LINE = "#ff7f0e"
R2_BAND = "#ffbb78"

R_DOT = "#9467bd"
R_LINE = "#9467bd"
R_BAND = "#c5b0d5"

def plot_metrics_across_splits(metrics_df):
    split_ids = np.arange(1, len(metrics_df) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{TRAIT} - {MODEL_NAME} test metrics across 25 splits",
        fontsize=24,
        fontweight="bold",
        y=0.98
    )

    specs = [
        ("RMSE", "RMSE", RMSE_DOT, RMSE_LINE, RMSE_BAND),
        ("MAE", "MAE", MAE_DOT, MAE_LINE, MAE_BAND),
        ("r2", "R²", R2_DOT, R2_LINE, R2_BAND),
        ("r", "Pearson r", R_DOT, R_LINE, R_BAND),
    ]

    axes = axes.flatten()

    for ax, (col, title, dot_c, line_c, band_c) in zip(axes, specs):
        values = metrics_df[col].values.astype(float)
        mean_val = np.mean(values)
        sd_val = np.std(values, ddof=1)

        ax.scatter(split_ids, values, s=55, alpha=0.95,
                   color=dot_c, edgecolor="#444444", linewidth=0.8, label="Splits", zorder=3)
        ax.axhline(mean_val, linestyle="--", linewidth=2.2, color=line_c,
                   label=f"Mean = {mean_val:.3f}", zorder=2)
        ax.axhspan(mean_val - sd_val, mean_val + sd_val, color=band_c, alpha=0.16,
                   label=f"±1 SD = {sd_val:.3f}", zorder=1)

        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_xlabel("Split", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(split_ids)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=True)

    savefig(SUMMARY_DIR / f"{MODEL_NAME}_metrics_across_25_splits.png")

def plot_global_observed_vs_predicted(pred_df):
    obs = pred_df["Observed"].values.astype(float)
    pred = pred_df["Predicted"].values.astype(float)

    r_val = pearson_r(obs, pred)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae = np.mean(np.abs(obs - pred))

    min_v = min(obs.min(), pred.min())
    max_v = max(obs.max(), pred.max())

    plt.figure(figsize=(8, 8))
    plt.scatter(obs, pred, alpha=0.5, s=30)
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=2)

    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"{MODEL_NAME} - Observed vs Predicted (all splits)", fontsize=18, fontweight="bold")
    plt.grid(True, alpha=0.25)

    txt = f"r = {r_val:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"
    plt.text(0.03, 0.97, txt, transform=plt.gca().transAxes, va="top",
             bbox=dict(boxstyle="round", alpha=0.15))

    savefig(SUMMARY_DIR / f"{MODEL_NAME}_observed_vs_predicted_all_splits.png")


def plot_residuals_summary(pred_df):
    df = pred_df.copy()
    df["Residual"] = df["Observed"] - df["Predicted"]

    plt.figure(figsize=(9, 6))
    plt.scatter(df["Predicted"], df["Residual"], alpha=0.5, s=25)
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.title(f"{MODEL_NAME} - Residuals vs Predicted", fontsize=18, fontweight="bold")
    plt.grid(True, alpha=0.25)
    savefig(SUMMARY_DIR / f"{MODEL_NAME}_residuals_vs_predicted.png")

    plt.figure(figsize=(9, 6))
    plt.hist(df["Residual"], bins=30, alpha=0.8, edgecolor="black")
    plt.axvline(df["Residual"].mean(), linestyle="--", linewidth=2, label=f"Mean = {df['Residual'].mean():.3f}")
    plt.xlabel("Residual (Observed - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"{MODEL_NAME} - Residual distribution", fontsize=18, fontweight="bold")
    plt.grid(True, alpha=0.2)
    plt.legend()
    savefig(SUMMARY_DIR / f"{MODEL_NAME}_residual_distribution.png")


def plot_loss_curves_per_split(histories):
    for split_name, hist in histories.items():
        plt.figure(figsize=(8, 5))
        plt.plot(hist["epoch"], hist["train_loss"], linewidth=2, label="Train loss")
        plt.plot(hist["epoch"], hist["val_loss"], linewidth=2, label="Validation loss")

        best_epoch = int(hist.loc[hist["val_loss"].idxmin(), "epoch"])
        plt.axvline(best_epoch, linestyle="--", linewidth=1.8, label=f"Best epoch = {best_epoch}")

        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(f"{MODEL_NAME} - Train vs Validation loss - {split_name}", fontsize=16, fontweight="bold")
        plt.grid(True, alpha=0.25)
        plt.legend()

        savefig(PER_SPLIT_DIR / f"{MODEL_NAME}_loss_curve_{split_name}.png")


def plot_mean_loss_curves(histories):
    _, train_mat = build_padded_matrix(histories, "train_loss")
    _, val_mat = build_padded_matrix(histories, "val_loss")

    epochs = np.arange(1, train_mat.shape[1] + 1)

    train_mean = np.nanmean(train_mat, axis=0)
    val_mean = np.nanmean(val_mat, axis=0)

    train_p10 = np.nanpercentile(train_mat, 10, axis=0)
    train_p90 = np.nanpercentile(train_mat, 90, axis=0)
    val_p10 = np.nanpercentile(val_mat, 10, axis=0)
    val_p90 = np.nanpercentile(val_mat, 90, axis=0)

    plt.figure(figsize=(11, 7))
    plt.fill_between(epochs, train_p10, train_p90, alpha=0.20, label="Train 10th-90th pct")
    plt.fill_between(epochs, val_p10, val_p90, alpha=0.20, label="Val 10th-90th pct")
    plt.plot(epochs, train_mean, linewidth=3, label="Mean train loss")
    plt.plot(epochs, val_mean, linewidth=3, label="Mean validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"{MODEL_NAME} - Mean train/validation loss across splits", fontsize=18, fontweight="bold")
    plt.grid(True, alpha=0.25)
    plt.legend()

    savefig(SUMMARY_DIR / f"{MODEL_NAME}_mean_train_val_loss.png")


def plot_best_epoch_distribution(metrics_df):
    vals = metrics_df["best_epoch"].values.astype(float)

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=12, alpha=0.8, edgecolor="black")
    plt.axvline(vals.mean(), linestyle="--", linewidth=2, label=f"Mean = {vals.mean():.2f}")
    plt.xlabel("Best epoch")
    plt.ylabel("Frequency")
    plt.title(f"{MODEL_NAME} - Best epoch distribution", fontsize=18, fontweight="bold")
    plt.grid(True, alpha=0.2)
    plt.legend()

    savefig(SUMMARY_DIR / f"{MODEL_NAME}_best_epoch_distribution.png")


def main():
    make_dirs()

    metrics_df = load_metrics()
    pred_df = load_predictions()
    histories = load_loss_histories()

    plot_metrics_across_splits(metrics_df)
    plot_global_observed_vs_predicted(pred_df)
    plot_residuals_summary(pred_df)
    plot_loss_curves_per_split(histories)
    plot_mean_loss_curves(histories)
    plot_best_epoch_distribution(metrics_df)

    print("All plots created successfully.")
    print(f"Summary plots: {SUMMARY_DIR}")
    print(f"Per-split plots: {PER_SPLIT_DIR}")


if __name__ == "__main__":
    main()