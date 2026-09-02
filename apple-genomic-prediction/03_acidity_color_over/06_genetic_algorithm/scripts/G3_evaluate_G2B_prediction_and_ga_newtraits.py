# -*- coding: utf-8 -*-

################################################################################
### G3_evaluate_G2B_prediction_and_ga_newtraits.py
###
### Post-run evaluation for:
###   G2B_multiseed_ga_newtraits.py
###
### For:
###   - Acidity
###   - Color_over
###
### Produces, for each trait:
###   - test metrics plots
###   - y_true vs y_pred scatter
###   - residual plots
###   - innerCV vs test RMSE
###   - trainval vs test RMSE
###   - selected SNP count per seed
###   - hyperparameter frequency plots
###   - GA fitness evolution plots
###   - final report
###
### Input:
###   Output/06_ga_runs/G2B_multiseed_ga_newtraits/<TRAIT>/
###
### Output:
###   Output/06_ga_runs/G2B_multiseed_ga_newtraits/<TRAIT>/figures_G3/
###   Output/06_ga_runs/G2B_multiseed_ga_newtraits/<TRAIT>/tables_G3/
################################################################################

from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

BASE_RUN_DIR = Path("Output/06_ga_runs/G2B_multiseed_ga_newtraits")

DPI = 300
TOP_N = 20


# =============================================================================
# BASIC UTILS
# =============================================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def read_required(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def savefig(fig_dir: Path, name: str):
    path = fig_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")


def get_axis_labels(trait: str):
    if trait == "Acidity":
        return "Observed acidity", "Predicted acidity"

    if trait == "Color_over":
        return "Observed over color", "Predicted over color"

    return f"Observed {trait}", f"Predicted {trait}"


# =============================================================================
# PATHS
# =============================================================================

def get_trait_paths(trait: str):
    run_dir = BASE_RUN_DIR / trait
    fig_dir = run_dir / "figures_G3"
    table_dir = run_dir / "tables_G3"

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "run_dir": run_dir,
        "fig_dir": fig_dir,
        "table_dir": table_dir,
        "metrics_per_seed": run_dir / f"G2B_multiseed_metrics_per_seed_{trait}.csv",
        "metrics_summary": run_dir / f"G2B_multiseed_metrics_summary_{trait}.csv",
        "hyperparam_freq": run_dir / f"G2B_best_hyperparams_frequency_{trait}.csv",
        "ridge_freq": run_dir / f"G2B_ridge_alpha_frequency_{trait}.csv",
        "lambda_freq": run_dir / f"G2B_lambda_size_frequency_{trait}.csv",
        "n_selected_summary": run_dir / f"G2B_n_selected_summary_{trait}.csv",
    }

    return paths


# =============================================================================
# COLLECT PREDICTIONS
# =============================================================================

def collect_test_predictions(run_dir: Path, table_dir: Path, trait: str):
    rows = []

    for seed_dir in sorted(run_dir.glob("seed_*")):
        pred_file = seed_dir / "best_model_test_predictions.csv"

        if pred_file.exists():
            df = pd.read_csv(pred_file)

            if "Trait" not in df.columns:
                df["Trait"] = trait

            rows.append(df)

    if len(rows) == 0:
        raise FileNotFoundError(
            f"No best_model_test_predictions.csv files found in:\n{run_dir}/seed_*"
        )

    pred_all = pd.concat(rows, ignore_index=True)
    pred_all.to_csv(table_dir / f"G3_all_test_predictions_{trait}.csv", index=False)

    return pred_all


def collect_trainval_predictions(run_dir: Path, table_dir: Path, trait: str):
    rows = []

    for seed_dir in sorted(run_dir.glob("seed_*")):
        pred_file = seed_dir / "best_model_trainval_predictions.csv"

        if pred_file.exists():
            df = pd.read_csv(pred_file)

            if "Trait" not in df.columns:
                df["Trait"] = trait

            rows.append(df)

    if len(rows) == 0:
        print(f"[WARNING] {trait}: no trainval prediction files found.")
        return None

    pred_all = pd.concat(rows, ignore_index=True)
    pred_all.to_csv(table_dir / f"G3_all_trainval_predictions_{trait}.csv", index=False)

    return pred_all


# =============================================================================
# LOGBOOK LOADING
# =============================================================================

def load_best_logbook_for_seed(run_dir: Path, seed, best_ridge_alpha, best_lambda_size):
    """
    New G2B logbooks may include cxpb/mutpb in the filename:

        logbook_seed42_ridge10.0_lambda0.02_cxpb0.5_mutpb0.2.csv

    This function first tries flexible regex matching.
    """

    seed_dir = run_dir / f"seed_{seed}"

    if not seed_dir.exists():
        print(f"[WARNING] Seed dir not found: {seed_dir}")
        return None

    candidates = list(seed_dir.glob(f"logbook_seed{seed}_ridge*_lambda*.csv"))

    if len(candidates) == 0:
        print(f"[WARNING] Could not find any logbook for seed {seed}.")
        return None

    for c in candidates:
        name = c.name

        try:
            ridge_match = re.search(r"_ridge(.+?)_lambda", name)
            lambda_match = re.search(r"_lambda(.+?)(?:_cxpb|\.csv)", name)

            if ridge_match and lambda_match:
                ridge_val = float(ridge_match.group(1))
                lambda_val = float(lambda_match.group(1))

                if (
                    np.isclose(ridge_val, float(best_ridge_alpha))
                    and np.isclose(lambda_val, float(best_lambda_size))
                ):
                    return pd.read_csv(c)

        except Exception:
            continue

    print(
        f"[WARNING] Could not find matching logbook for seed {seed}, "
        f"ridge={best_ridge_alpha}, lambda={best_lambda_size}."
    )
    return None


# =============================================================================
# PLOTS: METRICS
# =============================================================================

def plot_metrics_per_seed(metrics_df, fig_dir: Path, table_dir: Path, trait: str):
    test_cols = ["test_RMSE", "test_MAE", "test_R2", "test_Pearson_r"]

    for col in test_cols:
        plt.figure(figsize=(8, 5))
        plt.plot(metrics_df["seed"], metrics_df[col], marker="o")
        plt.xlabel("Seed")
        plt.ylabel(col)
        plt.title(f"{trait} - {col} across repeated train/test splits")
        plt.grid(alpha=0.3)
        savefig(fig_dir, f"{trait}_{col}_per_seed.png")

    summary_rows = []

    for col in test_cols:
        summary_rows.append({
            "Trait": trait,
            "metric": col,
            "mean": metrics_df[col].mean(),
            "std": metrics_df[col].std(),
            "min": metrics_df[col].min(),
            "median": metrics_df[col].median(),
            "max": metrics_df[col].max(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / f"G3_test_metrics_summary_{trait}.csv", index=False)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(summary))
    plt.bar(x, summary["mean"], yerr=summary["std"], capsize=5)
    plt.xticks(x, summary["metric"], rotation=30, ha="right")
    plt.ylabel("Mean ± SD")
    plt.title(f"{trait} - test metrics summary across seeds")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_test_metrics_summary_barplot.png")


def plot_inner_vs_test(metrics_df, fig_dir: Path, trait: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(metrics_df["innerCV_RMSE_mean"], metrics_df["test_RMSE"])

    for _, row in metrics_df.iterrows():
        plt.text(
            row["innerCV_RMSE_mean"],
            row["test_RMSE"],
            str(int(row["seed"])),
            fontsize=8
        )

    plt.xlabel("Inner CV RMSE mean")
    plt.ylabel("Test RMSE")
    plt.title(f"{trait} - inner CV RMSE vs external test RMSE")
    plt.grid(alpha=0.3)
    savefig(fig_dir, f"{trait}_innerCV_RMSE_vs_test_RMSE.png")

    plt.figure(figsize=(8, 5))
    plt.plot(
        metrics_df["seed"],
        metrics_df["innerCV_RMSE_mean"],
        marker="o",
        label="Inner CV RMSE"
    )
    plt.plot(
        metrics_df["seed"],
        metrics_df["test_RMSE"],
        marker="o",
        label="Test RMSE"
    )
    plt.xlabel("Seed")
    plt.ylabel("RMSE")
    plt.title(f"{trait} - inner CV RMSE and test RMSE across seeds")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig(fig_dir, f"{trait}_innerCV_and_test_RMSE_per_seed.png")


def plot_trainval_vs_test(metrics_df, fig_dir: Path, trait: str):
    plt.figure(figsize=(8, 5))
    plt.plot(
        metrics_df["seed"],
        metrics_df["trainval_RMSE"],
        marker="o",
        label="Trainval RMSE"
    )
    plt.plot(
        metrics_df["seed"],
        metrics_df["test_RMSE"],
        marker="o",
        label="Test RMSE"
    )
    plt.xlabel("Seed")
    plt.ylabel("RMSE")
    plt.title(f"{trait} - trainval vs test RMSE across seeds")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig(fig_dir, f"{trait}_trainval_vs_test_RMSE_per_seed.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(metrics_df["trainval_RMSE"], metrics_df["test_RMSE"])

    for _, row in metrics_df.iterrows():
        plt.text(
            row["trainval_RMSE"],
            row["test_RMSE"],
            str(int(row["seed"])),
            fontsize=8
        )

    plt.xlabel("Trainval RMSE")
    plt.ylabel("Test RMSE")
    plt.title(f"{trait} - trainval RMSE vs test RMSE")
    plt.grid(alpha=0.3)
    savefig(fig_dir, f"{trait}_trainval_RMSE_vs_test_RMSE.png")


def plot_n_selected(metrics_df, fig_dir: Path, trait: str):
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["seed"].astype(str), metrics_df["n_selected_snps"])
    plt.xlabel("Seed")
    plt.ylabel("Number of selected SNPs")
    plt.title(f"{trait} - selected SNP subset size across seeds")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_n_selected_snps_per_seed.png")

    plt.figure(figsize=(6, 5))
    plt.hist(metrics_df["n_selected_snps"], bins=8)
    plt.xlabel("Number of selected SNPs")
    plt.ylabel("Count")
    plt.title(f"{trait} - distribution of selected SNP subset size")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_n_selected_snps_distribution.png")


# =============================================================================
# PLOTS: PREDICTIONS AND RESIDUALS
# =============================================================================

def plot_test_predictions(pred_all, fig_dir: Path, table_dir: Path, trait: str):
    y_true = pred_all["y_true"].values.astype(float)
    y_pred = pred_all["y_pred"].values.astype(float)

    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": pearson_r(y_true, y_pred),
    }

    with open(table_dir / f"G3_aggregated_test_prediction_metrics_{trait}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    pad = 0.08 * (max_val - min_val) if max_val > min_val else 1.0

    xlabel, ylabel = get_axis_labels(trait)

    plt.figure(figsize=(6.8, 6.8))

    plt.scatter(
        y_true,
        y_pred,
        s=22,
        alpha=0.35,
        edgecolors="none"
    )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=1.8,
        color="black",
        alpha=0.75,
        label="Ideal prediction"
    )

    plt.xlim(min_val - pad, max_val + pad)
    plt.ylim(min_val - pad, max_val + pad)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    text = (
        f"RMSE = {metrics['RMSE']:.3f}\n"
        f"MAE = {metrics['MAE']:.3f}\n"
        f"R² = {metrics['R2']:.3f}\n"
        f"r = {metrics['Pearson_r']:.3f}"
    )

    plt.text(
        0.04,
        0.96,
        text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none")
    )

    plt.legend(frameon=False, loc="lower right")
    plt.grid(alpha=0.25)
    plt.title(f"{trait} - external test predictions across all seeds")
    savefig(fig_dir, f"{trait}_test_ytrue_vs_ypred_all_seeds.png")

    # One scatter per seed
    for seed, df_seed in pred_all.groupby("seed"):
        y_true_s = df_seed["y_true"].values.astype(float)
        y_pred_s = df_seed["y_pred"].values.astype(float)

        min_s = min(np.min(y_true_s), np.min(y_pred_s))
        max_s = max(np.max(y_true_s), np.max(y_pred_s))
        pad_s = 0.08 * (max_s - min_s) if max_s > min_s else 1.0

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_s, y_pred_s, alpha=0.75)
        plt.plot(
            [min_s, max_s],
            [min_s, max_s],
            linestyle="--",
            color="black",
            alpha=0.75
        )
        plt.xlim(min_s - pad_s, max_s + pad_s)
        plt.ylim(min_s - pad_s, max_s + pad_s)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{trait} - test predictions - seed {seed}")
        plt.grid(alpha=0.3)
        savefig(fig_dir, f"{trait}_test_ytrue_vs_ypred_seed_{seed}.png")


def plot_residuals(pred_all, fig_dir: Path, table_dir: Path, trait: str):
    pred_all = pred_all.copy()
    pred_all["residual"] = pred_all["y_true"] - pred_all["y_pred"]

    plt.figure(figsize=(7, 5))
    plt.hist(pred_all["residual"], bins=25)
    plt.xlabel("Residual = observed - predicted")
    plt.ylabel("Count")
    plt.title(f"{trait} - distribution of test residuals across all seeds")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_test_residuals_distribution.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(pred_all["y_pred"], pred_all["residual"], alpha=0.65)
    plt.axhline(0, linestyle="--", color="black", alpha=0.75)
    plt.xlabel("Predicted value")
    plt.ylabel("Residual = observed - predicted")
    plt.title(f"{trait} - test residuals vs predicted values")
    plt.grid(alpha=0.3)
    savefig(fig_dir, f"{trait}_test_residuals_vs_predicted.png")

    residual_summary = (
        pred_all
        .groupby("seed")
        .agg(
            residual_mean=("residual", "mean"),
            residual_std=("residual", "std"),
            residual_min=("residual", "min"),
            residual_median=("residual", "median"),
            residual_max=("residual", "max"),
        )
        .reset_index()
    )

    residual_summary.to_csv(
        table_dir / f"G3_residual_summary_by_seed_{trait}.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))
    plt.plot(residual_summary["seed"], residual_summary["residual_mean"], marker="o")
    plt.axhline(0, linestyle="--", color="black", alpha=0.75)
    plt.xlabel("Seed")
    plt.ylabel("Mean residual")
    plt.title(f"{trait} - mean test residual by seed")
    plt.grid(alpha=0.3)
    savefig(fig_dir, f"{trait}_mean_test_residual_by_seed.png")


# =============================================================================
# PLOTS: HYPERPARAMETERS
# =============================================================================

def plot_hyperparameters(metrics_df, fig_dir: Path, table_dir: Path, trait: str):
    ridge_counts = (
        metrics_df
        .groupby("best_ridge_alpha")
        .size()
        .reset_index(name="n_seeds")
        .sort_values("best_ridge_alpha")
    )

    ridge_counts.to_csv(
        table_dir / f"G3_ridge_alpha_frequency_{trait}.csv",
        index=False
    )

    plt.figure(figsize=(6, 5))
    plt.bar(ridge_counts["best_ridge_alpha"].astype(str), ridge_counts["n_seeds"])
    plt.xlabel("Best ridge_alpha")
    plt.ylabel("Number of seeds")
    plt.title(f"{trait} - frequency of selected ridge_alpha")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_ridge_alpha_frequency.png")

    lambda_counts = (
        metrics_df
        .groupby("best_lambda_size")
        .size()
        .reset_index(name="n_seeds")
        .sort_values("best_lambda_size")
    )

    lambda_counts.to_csv(
        table_dir / f"G3_lambda_size_frequency_{trait}.csv",
        index=False
    )

    plt.figure(figsize=(6, 5))
    plt.bar(lambda_counts["best_lambda_size"].astype(str), lambda_counts["n_seeds"])
    plt.xlabel("Best lambda_size")
    plt.ylabel("Number of seeds")
    plt.title(f"{trait} - frequency of selected lambda_size")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_lambda_size_frequency.png")

    combo_counts = (
        metrics_df
        .groupby(["best_ridge_alpha", "best_lambda_size"])
        .size()
        .reset_index(name="n_seeds")
        .sort_values("n_seeds", ascending=False)
    )

    combo_counts["label"] = (
        "ridge=" + combo_counts["best_ridge_alpha"].astype(str)
        + ", lambda=" + combo_counts["best_lambda_size"].astype(str)
    )

    combo_counts.to_csv(
        table_dir / f"G3_best_hyperparam_combo_frequency_{trait}.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))
    plt.bar(combo_counts["label"], combo_counts["n_seeds"])
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Best hyperparameter combination")
    plt.ylabel("Number of seeds")
    plt.title(f"{trait} - frequency of selected hyperparameter combinations")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_best_hyperparameter_combo_frequency.png")

    # Optional: CXPB/MUTPB used
    if {"cxpb", "mutpb"}.issubset(metrics_df.columns):
        used = (
            metrics_df[["cxpb", "mutpb"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        used.to_csv(table_dir / f"G3_ga_operator_config_used_{trait}.csv", index=False)


# =============================================================================
# FITNESS EVOLUTION
# =============================================================================

def plot_fitness_evolution(metrics_df, run_dir: Path, fig_dir: Path, table_dir: Path, trait: str):
    all_best_logbooks = []

    for _, row in metrics_df.iterrows():
        seed = int(row["seed"])
        ridge = row["best_ridge_alpha"]
        lam = row["best_lambda_size"]

        logbook = load_best_logbook_for_seed(run_dir, seed, ridge, lam)

        if logbook is None:
            continue

        if "gen" not in logbook.columns or "min" not in logbook.columns:
            print(f"[WARNING] Logbook for seed {seed} does not contain expected columns.")
            continue

        logbook = logbook.copy()
        logbook["Trait"] = trait
        logbook["seed"] = seed
        logbook["best_ridge_alpha"] = ridge
        logbook["best_lambda_size"] = lam

        all_best_logbooks.append(logbook)

    if len(all_best_logbooks) == 0:
        print(f"[WARNING] {trait}: no logbooks available for fitness evolution plots.")
        return

    log_all = pd.concat(all_best_logbooks, ignore_index=True)
    log_all.to_csv(
        table_dir / f"G3_best_config_logbooks_all_seeds_{trait}.csv",
        index=False
    )

    fitness_summary = (
        log_all
        .groupby("gen")
        .agg(
            fitness_mean=("min", "mean"),
            fitness_std=("min", "std"),
            fitness_median=("min", "median"),
            fitness_q25=("min", lambda x: np.percentile(x, 25)),
            fitness_q75=("min", lambda x: np.percentile(x, 75)),
            fitness_min=("min", "min"),
            fitness_max=("min", "max"),
        )
        .reset_index()
    )

    fitness_summary.to_csv(
        table_dir / f"G3_fitness_evolution_summary_{trait}.csv",
        index=False
    )

    x = fitness_summary["gen"].values
    y_mean = fitness_summary["fitness_mean"].values
    y_median = fitness_summary["fitness_median"].values
    y_q25 = fitness_summary["fitness_q25"].values
    y_q75 = fitness_summary["fitness_q75"].values

    # Mean line
    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        y_mean,
        linewidth=2.0,
        label="Mean best fitness"
    )
    plt.xlabel("Generation")
    plt.ylabel("Best fitness, lower is better")
    plt.title(f"{trait} - GA fitness evolution across repeated runs")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    savefig(fig_dir, f"{trait}_fitness_evolution_mean_points_line.png")

    # Median points line
    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        y_median,
        marker="o",
        markersize=4.5,
        linewidth=2.2,
        label="Median best fitness"
    )
    plt.xlabel("Generation")
    plt.ylabel("Best fitness, lower is better")
    plt.title(f"{trait} - GA fitness evolution across repeated runs")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    savefig(fig_dir, f"{trait}_fitness_evolution_median_points_line.png")

    # Median + IQR
    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        y_median,
        marker="o",
        markersize=3.5,
        linewidth=2.4,
        label="Median best fitness"
    )
    plt.fill_between(
        x,
        y_q25,
        y_q75,
        alpha=0.18,
        label="Interquartile range"
    )
    plt.xlabel("Generation")
    plt.ylabel("Best fitness, lower is better")
    plt.title(f"{trait} - GA convergence across repeated runs")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    savefig(fig_dir, f"{trait}_fitness_evolution_median_IQR_thesis.png")

    # Individual trajectories + median
    plt.figure(figsize=(8, 5))

    for seed, sub in log_all.groupby("seed"):
        plt.plot(
            sub["gen"],
            sub["min"],
            linewidth=1.0,
            alpha=0.25
        )

    plt.plot(
        x,
        y_median,
        linewidth=2.8,
        label="Median best fitness"
    )

    plt.xlabel("Generation")
    plt.ylabel("Best fitness, lower is better")
    plt.title(f"{trait} - GA best-fitness trajectories across seeds")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    savefig(fig_dir, f"{trait}_fitness_evolution_individual_seeds_with_median_supplementary.png")


# =============================================================================
# REPORT
# =============================================================================

def write_report(metrics_df, pred_all, run_dir: Path, fig_dir: Path, table_dir: Path, trait: str):
    report_path = run_dir / f"G3_prediction_and_ga_report_{trait}.txt"

    test_metrics = {
        "test_RMSE_mean": metrics_df["test_RMSE"].mean(),
        "test_RMSE_std": metrics_df["test_RMSE"].std(),
        "test_MAE_mean": metrics_df["test_MAE"].mean(),
        "test_MAE_std": metrics_df["test_MAE"].std(),
        "test_R2_mean": metrics_df["test_R2"].mean(),
        "test_R2_std": metrics_df["test_R2"].std(),
        "test_Pearson_r_mean": metrics_df["test_Pearson_r"].mean(),
        "test_Pearson_r_std": metrics_df["test_Pearson_r"].std(),
        "n_selected_mean": metrics_df["n_selected_snps"].mean(),
        "n_selected_std": metrics_df["n_selected_snps"].std(),
    }

    y_true = pred_all["y_true"].values.astype(float)
    y_pred = pred_all["y_pred"].values.astype(float)

    aggregate_prediction_metrics = {
        "aggregated_test_RMSE": rmse(y_true, y_pred),
        "aggregated_test_MAE": float(mean_absolute_error(y_true, y_pred)),
        "aggregated_test_R2": float(r2_score(y_true, y_pred)),
        "aggregated_test_Pearson_r": pearson_r(y_true, y_pred),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"G3 REPORT - G2B PREDICTION AND GA EVALUATION - {trait}\n")
        f.write("DATASET_LABEL: no_soil_top1000_regions_newtraits\n")
        f.write("=" * 80 + "\n\n")

        f.write("TEST PERFORMANCE ACROSS SEEDS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test RMSE: {test_metrics['test_RMSE_mean']:.4f} ± {test_metrics['test_RMSE_std']:.4f}\n")
        f.write(f"Test MAE: {test_metrics['test_MAE_mean']:.4f} ± {test_metrics['test_MAE_std']:.4f}\n")
        f.write(f"Test R2: {test_metrics['test_R2_mean']:.4f} ± {test_metrics['test_R2_std']:.4f}\n")
        f.write(f"Test Pearson r: {test_metrics['test_Pearson_r_mean']:.4f} ± {test_metrics['test_Pearson_r_std']:.4f}\n")
        f.write(f"Selected SNPs: {test_metrics['n_selected_mean']:.2f} ± {test_metrics['n_selected_std']:.2f}\n\n")

        f.write("AGGREGATED TEST PREDICTIONS ACROSS ALL SEEDS\n")
        f.write("-" * 80 + "\n")

        for k, v in aggregate_prediction_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\nOUTPUT FOLDERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Figures: {fig_dir}\n")
        f.write(f"Tables: {table_dir}\n")

    print(f"[REPORT] Saved: {report_path}")


# =============================================================================
# ONE TRAIT
# =============================================================================

def run_one_trait(trait: str):
    print("\n" + "#" * 80)
    print(f"G3 - EVALUATE G2B PREDICTION AND GA - {trait}")
    print("#" * 80)

    paths = get_trait_paths(trait)

    run_dir = paths["run_dir"]
    fig_dir = paths["fig_dir"]
    table_dir = paths["table_dir"]

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found for {trait}:\n{run_dir}")

    metrics_df = read_required(paths["metrics_per_seed"])
    metrics_df.to_csv(table_dir / f"G3_metrics_per_seed_copy_{trait}.csv", index=False)

    pred_all = collect_test_predictions(run_dir, table_dir, trait)
    collect_trainval_predictions(run_dir, table_dir, trait)

    plot_metrics_per_seed(metrics_df, fig_dir, table_dir, trait)
    plot_inner_vs_test(metrics_df, fig_dir, trait)
    plot_trainval_vs_test(metrics_df, fig_dir, trait)
    plot_n_selected(metrics_df, fig_dir, trait)
    plot_test_predictions(pred_all, fig_dir, table_dir, trait)
    plot_residuals(pred_all, fig_dir, table_dir, trait)
    plot_hyperparameters(metrics_df, fig_dir, table_dir, trait)
    plot_fitness_evolution(metrics_df, run_dir, fig_dir, table_dir, trait)
    write_report(metrics_df, pred_all, run_dir, fig_dir, table_dir, trait)

    print("\n" + "#" * 80)
    print(f"G3 FINISHED FOR {trait}")
    print("#" * 80)
    print(f"Figures saved in:\n  {fig_dir}")
    print(f"Tables saved in:\n  {table_dir}")

    # Return compact summary for global table
    summary = {
        "Trait": trait,
        "test_RMSE_mean": metrics_df["test_RMSE"].mean(),
        "test_RMSE_std": metrics_df["test_RMSE"].std(),
        "test_MAE_mean": metrics_df["test_MAE"].mean(),
        "test_MAE_std": metrics_df["test_MAE"].std(),
        "test_R2_mean": metrics_df["test_R2"].mean(),
        "test_R2_std": metrics_df["test_R2"].std(),
        "test_Pearson_r_mean": metrics_df["test_Pearson_r"].mean(),
        "test_Pearson_r_std": metrics_df["test_Pearson_r"].std(),
        "n_selected_snps_mean": metrics_df["n_selected_snps"].mean(),
        "n_selected_snps_std": metrics_df["n_selected_snps"].std(),
        "fig_dir": str(fig_dir),
        "table_dir": str(table_dir),
    }

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 80)
    print("G3 - EVALUATE G2B NEW TRAITS")
    print("#" * 80)

    all_summaries = []

    for trait in TRAITS:
        summary = run_one_trait(trait)
        all_summaries.append(summary)

    summary_all = pd.DataFrame(all_summaries)

    out_file = BASE_RUN_DIR / "G3_prediction_and_ga_summary_all_traits.csv"
    summary_all.to_csv(out_file, index=False)

    print("\n" + "=" * 80)
    print("G3 ALL TRAITS FINISHED")
    print("=" * 80)
    print(summary_all.to_string(index=False))
    print("\nSaved all-trait summary:")
    print(out_file)


if __name__ == "__main__":
    main()