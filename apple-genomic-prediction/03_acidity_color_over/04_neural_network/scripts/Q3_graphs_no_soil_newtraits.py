################################################################################
### Q3_graphs_no_soil_newtraits.py
###
### Graphs and diagnostics after:
###   Q2_train_no_soil_newtraits.py
###
### For:
###   - Acidity
###   - Color_over
###
### Produces:
###   - metrics panel across 25 splits
###   - observed vs predicted all splits
###   - residuals vs predicted
###   - residual distribution
###   - per-split loss curves
###   - mean train/validation loss curve
###   - best epoch distribution
###
### Da eseguire da:
###   dalpaper/nuovitrattinosoil/
################################################################################

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

MODEL_FULL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"
MODEL_LABEL = "no_soil_v3"

BASE_MODEL_DIR = (
    Path("03_acidity_color_over")
    / "04_neural_network"
    / "output"
    / "no_soil_model"
)

DPI = 300


# =============================================================================
# COLORS
# =============================================================================

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


# =============================================================================
# UTILS
# =============================================================================

def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")


def pearson_r(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def split_sort_key(split_name: str):
    """
    Sort CV1_Split1, CV1_Split2, ..., CV5_Split5 naturally.
    """
    try:
        cv_part, split_part = str(split_name).split("_")
        cv_num = int(cv_part.replace("CV", ""))
        split_num = int(split_part.replace("Split", ""))
        return cv_num, split_num
    except Exception:
        return 999, 999


def get_trait_paths(trait: str):
    out_dir = BASE_MODEL_DIR / trait

    metrics_file = (
        out_dir
        / "metrics"
        / f"metrics_{trait}_{MODEL_FULL_NAME}.csv"
    )

    pred_file = (
        out_dir
        / "predictions"
        / f"predictions_{trait}_{MODEL_FULL_NAME}_all_splits.csv"
    )

    loss_dir = out_dir / "loss_history"

    grafici_dir = out_dir / "grafici"
    per_split_dir = grafici_dir / "per_split"
    summary_dir = grafici_dir / "summary"

    return {
        "out_dir": out_dir,
        "metrics_file": metrics_file,
        "pred_file": pred_file,
        "loss_dir": loss_dir,
        "grafici_dir": grafici_dir,
        "per_split_dir": per_split_dir,
        "summary_dir": summary_dir,
    }


def make_dirs(paths):
    paths["per_split_dir"].mkdir(parents=True, exist_ok=True)
    paths["summary_dir"].mkdir(parents=True, exist_ok=True)


def load_metrics(metrics_file: Path):
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found:\n{metrics_file}")

    df = pd.read_csv(metrics_file)
    df = df[df["Split"] != "Mean"].copy()

    df["Split"] = df["Split"].astype(str)
    df = df.sort_values("Split", key=lambda s: s.map(split_sort_key)).reset_index(drop=True)

    return df


def load_predictions(pred_file: Path):
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found:\n{pred_file}")

    df = pd.read_csv(pred_file)

    needed = {"Observed", "Predicted", "Split", "Envir", "Genotype"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file missing columns: {missing}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    df["Split"] = df["Split"].astype(str)
    df["Observed"] = pd.to_numeric(df["Observed"], errors="coerce")
    df["Predicted"] = pd.to_numeric(df["Predicted"], errors="coerce")
    df = df.dropna(subset=["Observed", "Predicted"]).copy()

    return df


def load_loss_histories(loss_dir: Path, trait: str):
    """
    Q2 saves:
        loss_history_<trait>_<split>.csv

    Example:
        loss_history_Acidity_CV1_Split1.csv
    """
    files = sorted(loss_dir.glob(f"loss_history_{trait}_CV*_Split*.csv"))

    histories = {}

    for f in files:
        split_name = f.stem.replace(f"loss_history_{trait}_", "")
        histories[split_name] = pd.read_csv(f)

    histories = dict(
        sorted(histories.items(), key=lambda kv: split_sort_key(kv[0]))
    )

    return histories


def build_padded_matrix(histories, column_name):
    split_names = list(histories.keys())

    if len(split_names) == 0:
        return [], np.empty((0, 0))

    max_len = max(len(histories[s]) for s in split_names)

    mat = np.full((len(split_names), max_len), np.nan)

    for i, split_name in enumerate(split_names):
        vals = histories[split_name][column_name].values.astype(float)
        mat[i, :len(vals)] = vals

    return split_names, mat


# =============================================================================
# PLOTS
# =============================================================================

def plot_metrics_across_splits(trait: str, metrics_df: pd.DataFrame, summary_dir: Path):
    split_ids = np.arange(1, len(metrics_df) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{trait} - {MODEL_LABEL} test metrics across 25 splits",
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

        ax.scatter(
            split_ids,
            values,
            s=55,
            alpha=0.95,
            color=dot_c,
            edgecolor="#444444",
            linewidth=0.8,
            label="Splits",
            zorder=3
        )

        ax.axhline(
            mean_val,
            linestyle="--",
            linewidth=2.2,
            color=line_c,
            label=f"Mean = {mean_val:.3f}",
            zorder=2
        )

        ax.axhspan(
            mean_val - sd_val,
            mean_val + sd_val,
            color=band_c,
            alpha=0.16,
            label=f"±1 SD = {sd_val:.3f}",
            zorder=1
        )

        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_xlabel("Split", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(split_ids)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=True)

    savefig(summary_dir / f"{trait}_{MODEL_LABEL}_metrics_across_25_splits.png")


def plot_global_observed_vs_predicted(trait: str, pred_df: pd.DataFrame, summary_dir: Path):
    obs = pred_df["Observed"].values.astype(float)
    pred = pred_df["Predicted"].values.astype(float)

    r_val = pearson_r(obs, pred)
    rmse_val = np.sqrt(np.mean((obs - pred) ** 2))
    mae_val = np.mean(np.abs(obs - pred))

    min_v = min(obs.min(), pred.min())
    max_v = max(obs.max(), pred.max())

    pad = 0.03 * (max_v - min_v)
    min_v -= pad
    max_v += pad

    plt.figure(figsize=(8, 8))
    plt.scatter(obs, pred, alpha=0.45, s=25)
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=2)

    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(
        f"{trait} - {MODEL_LABEL}\nObserved vs Predicted (all test splits)",
        fontsize=18,
        fontweight="bold"
    )
    plt.grid(True, alpha=0.25)

    txt = f"r = {r_val:.3f}\nRMSE = {rmse_val:.3f}\nMAE = {mae_val:.3f}"
    plt.text(
        0.03,
        0.97,
        txt,
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    plt.xlim(min_v, max_v)
    plt.ylim(min_v, max_v)

    savefig(summary_dir / f"{trait}_{MODEL_LABEL}_observed_vs_predicted_all_splits.png")


def plot_residuals_summary(trait: str, pred_df: pd.DataFrame, summary_dir: Path):
    df = pred_df.copy()
    df["Residual"] = df["Observed"] - df["Predicted"]

    plt.figure(figsize=(9, 6))
    plt.scatter(df["Predicted"], df["Residual"], alpha=0.45, s=25)
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.title(
        f"{trait} - {MODEL_LABEL}\nResiduals vs Predicted",
        fontsize=18,
        fontweight="bold"
    )
    plt.grid(True, alpha=0.25)
    savefig(summary_dir / f"{trait}_{MODEL_LABEL}_residuals_vs_predicted.png")

    plt.figure(figsize=(9, 6))
    plt.hist(df["Residual"], bins=30, alpha=0.8, edgecolor="black")
    plt.axvline(
        df["Residual"].mean(),
        linestyle="--",
        linewidth=2,
        label=f"Mean = {df['Residual'].mean():.3f}"
    )
    plt.xlabel("Residual (Observed - Predicted)")
    plt.ylabel("Frequency")
    plt.title(
        f"{trait} - {MODEL_LABEL}\nResidual distribution",
        fontsize=18,
        fontweight="bold"
    )
    plt.grid(True, alpha=0.2)
    plt.legend()
    savefig(summary_dir / f"{trait}_{MODEL_LABEL}_residual_distribution.png")


def plot_loss_curves_per_split(trait: str, histories: dict, per_split_dir: Path):
    if len(histories) == 0:
        print(f"[WARNING] No loss histories found for {trait}. Skipping per-split loss curves.")
        return

    for split_name, hist in histories.items():
        plt.figure(figsize=(8, 5))

        plt.plot(
            hist["epoch"],
            hist["train_loss"],
            linewidth=2,
            label="Train loss"
        )

        plt.plot(
            hist["epoch"],
            hist["val_loss"],
            linewidth=2,
            label="Validation loss"
        )

        best_epoch = int(hist.loc[hist["val_loss"].idxmin(), "epoch"])

        plt.axvline(
            best_epoch,
            linestyle="--",
            linewidth=1.8,
            label=f"Best epoch = {best_epoch}"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(
            f"{trait} - {MODEL_LABEL}\nTrain vs Validation loss - {split_name}",
            fontsize=16,
            fontweight="bold"
        )
        plt.grid(True, alpha=0.25)
        plt.legend()

        savefig(per_split_dir / f"{trait}_{MODEL_LABEL}_loss_curve_{split_name}.png")


def plot_mean_loss_curves(trait: str, histories: dict, summary_dir: Path):
    if len(histories) == 0:
        print(f"[WARNING] No loss histories found for {trait}. Skipping mean loss curve.")
        return

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

    plt.fill_between(
        epochs,
        train_p10,
        train_p90,
        alpha=0.20,
        label="Train 10th-90th pct"
    )

    plt.fill_between(
        epochs,
        val_p10,
        val_p90,
        alpha=0.20,
        label="Val 10th-90th pct"
    )

    plt.plot(
        epochs,
        train_mean,
        linewidth=3,
        label="Mean train loss"
    )

    plt.plot(
        epochs,
        val_mean,
        linewidth=3,
        label="Mean validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(
        f"{trait} - {MODEL_LABEL}\nMean train/validation loss across 25 splits",
        fontsize=18,
        fontweight="bold"
    )
    plt.grid(True, alpha=0.25)
    plt.legend()

    savefig(summary_dir / f"{trait}_{MODEL_LABEL}_mean_train_val_loss.png")


def plot_best_epoch_distribution(trait: str, metrics_df: pd.DataFrame, summary_dir: Path):
    vals = metrics_df["best_epoch"].values.astype(float)

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=12, alpha=0.8, edgecolor="black")
    plt.axvline(
        vals.mean(),
        linestyle="--",
        linewidth=2,
        label=f"Mean = {vals.mean():.2f}"
    )
    plt.xlabel("Best epoch")
    plt.ylabel("Frequency")
    plt.title(
        f"{trait} - {MODEL_LABEL}\nBest epoch distribution",
        fontsize=18,
        fontweight="bold"
    )
    plt.grid(True, alpha=0.2)
    plt.legend()

    savefig(summary_dir / f"{trait}_{MODEL_LABEL}_best_epoch_distribution.png")


def write_plot_report(
    trait: str,
    metrics_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    histories: dict,
    summary_dir: Path
):
    report_file = summary_dir / f"{trait}_{MODEL_LABEL}_plot_report.txt"

    metric_cols = ["RMSE", "MAE", "r2", "r", "best_epoch"]

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Q3 GRAPH REPORT - {trait} - {MODEL_LABEL}\n")
        f.write("=" * 80 + "\n\n")

        f.write("INPUT SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Metrics rows, excluding Mean: {metrics_df.shape[0]}\n")
        f.write(f"Prediction rows: {pred_df.shape[0]}\n")
        f.write(f"Loss histories found: {len(histories)}\n\n")

        f.write("MEAN METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(metrics_df[metric_cols].mean().to_string())
        f.write("\n\n")

        f.write("STD METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(metrics_df[metric_cols].std(ddof=1).to_string())
        f.write("\n\n")

        f.write("PER-SPLIT METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n")

    print(f"[REPORT] Saved: {report_file}")


# =============================================================================
# MAIN PER TRAIT
# =============================================================================

def process_one_trait(trait: str):
    print("\n" + "=" * 80)
    print(f"Q3 - Creating graphs for trait: {trait}")
    print("=" * 80)

    paths = get_trait_paths(trait)
    make_dirs(paths)

    metrics_df = load_metrics(paths["metrics_file"])
    pred_df = load_predictions(paths["pred_file"])
    histories = load_loss_histories(paths["loss_dir"], trait)

    print(f"Metrics rows: {metrics_df.shape[0]}")
    print(f"Prediction rows: {pred_df.shape[0]}")
    print(f"Loss histories: {len(histories)}")

    plot_metrics_across_splits(trait, metrics_df, paths["summary_dir"])
    plot_global_observed_vs_predicted(trait, pred_df, paths["summary_dir"])
    plot_residuals_summary(trait, pred_df, paths["summary_dir"])
    plot_loss_curves_per_split(trait, histories, paths["per_split_dir"])
    plot_mean_loss_curves(trait, histories, paths["summary_dir"])
    plot_best_epoch_distribution(trait, metrics_df, paths["summary_dir"])

    write_plot_report(
        trait=trait,
        metrics_df=metrics_df,
        pred_df=pred_df,
        histories=histories,
        summary_dir=paths["summary_dir"],
    )

    return {
        "Trait": trait,
        "metrics_rows": metrics_df.shape[0],
        "prediction_rows": pred_df.shape[0],
        "loss_histories": len(histories),
        "summary_dir": str(paths["summary_dir"]),
        "per_split_dir": str(paths["per_split_dir"]),
    }


def main():
    print("\n" + "=" * 80)
    print("Q3 - GRAPHS NO-SOIL V3 FOR NEW TRAITS")
    print("=" * 80)

    rows = []

    for trait in TRAITS:
        rows.append(process_one_trait(trait))

    summary = pd.DataFrame(rows)

    out_file = BASE_MODEL_DIR / "Q3_graphs_completed_traits_summary.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_file, index=False)

    print("\n" + "=" * 80)
    print("Q3 COMPLETED")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(out_file)


if __name__ == "__main__":
    main()
