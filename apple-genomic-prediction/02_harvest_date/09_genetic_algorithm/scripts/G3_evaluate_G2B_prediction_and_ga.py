# -*- coding: utf-8 -*-

################################################################################
### G3_evaluate_G2B_prediction_and_ga.py
###
### Post-run evaluation for:
###   G2B_multiseed_ga_variable_split_inner3cv.py
###
### Produces:
###   - test metrics plots
###   - y_true vs y_pred scatter
###   - residual plots
###   - innerCV vs test RMSE
###   - trainval vs test RMSE
###   - selected SNP count per seed
###   - hyperparameter frequency plots
###   - GA fitness evolution plots
###
### Input:
###   Output/04_ga_runs/G2B_multiseed_variable_split_inner3cv/
###
### Output:
###   Output/04_ga_runs/G2B_multiseed_variable_split_inner3cv/figures_G3/
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

RUN_DIR = (
    Path("02_harvest_date")
    / "09_genetic_algorithm"
    / "output"
    / "04_ga_runs"
    / "G2B_no_soil_multiseed_variable_split_inner3cv"
)
FIG_DIR = RUN_DIR / "figures_G3"
TABLE_DIR = RUN_DIR / "tables_G3"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PER_SEED = RUN_DIR / "G2B_multiseed_metrics_per_seed.csv"
METRICS_SUMMARY = RUN_DIR / "G2B_multiseed_metrics_summary.csv"
HYPERPARAM_FREQ = RUN_DIR / "G2B_best_hyperparams_frequency.csv"
RIDGE_FREQ = RUN_DIR / "G2B_ridge_alpha_frequency.csv"
LAMBDA_FREQ = RUN_DIR / "G2B_lambda_size_frequency.csv"
N_SELECTED_SUMMARY = RUN_DIR / "G2B_n_selected_summary.csv"

DPI = 300
TOP_N = 20


# =============================================================================
# UTILS
# =============================================================================

def savefig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson_r(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def read_required(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def collect_test_predictions():
    rows = []

    for seed_dir in sorted(RUN_DIR.glob("seed_*")):
        pred_file = seed_dir / "best_model_test_predictions.csv"

        if pred_file.exists():
            df = pd.read_csv(pred_file)
            rows.append(df)

    if len(rows) == 0:
        raise FileNotFoundError("No best_model_test_predictions.csv files found in seed_* folders.")

    pred_all = pd.concat(rows, ignore_index=True)
    pred_all.to_csv(TABLE_DIR / "G3_all_test_predictions.csv", index=False)

    return pred_all


def collect_trainval_predictions():
    rows = []

    for seed_dir in sorted(RUN_DIR.glob("seed_*")):
        pred_file = seed_dir / "best_model_trainval_predictions.csv"

        if pred_file.exists():
            df = pd.read_csv(pred_file)
            rows.append(df)

    if len(rows) == 0:
        print("[WARNING] No trainval prediction files found.")
        return None

    pred_all = pd.concat(rows, ignore_index=True)
    pred_all.to_csv(TABLE_DIR / "G3_all_trainval_predictions.csv", index=False)

    return pred_all


def load_best_logbook_for_seed(seed, best_ridge_alpha, best_lambda_size):
    seed_dir = RUN_DIR / f"seed_{seed}"

    exact_name = f"logbook_seed{seed}_ridge{best_ridge_alpha}_lambda{best_lambda_size}.csv"
    exact_path = seed_dir / exact_name

    if exact_path.exists():
        return pd.read_csv(exact_path)

    # Fallback: search by regex, useful if float formatting differs.
    candidates = list(seed_dir.glob(f"logbook_seed{seed}_ridge*_lambda*.csv"))

    for c in candidates:
        name = c.name

        try:
            ridge_match = re.search(r"_ridge(.+?)_lambda", name)
            lambda_match = re.search(r"_lambda(.+?)\.csv", name)

            if ridge_match and lambda_match:
                ridge_val = float(ridge_match.group(1))
                lambda_val = float(lambda_match.group(1))

                if np.isclose(ridge_val, float(best_ridge_alpha)) and np.isclose(lambda_val, float(best_lambda_size)):
                    return pd.read_csv(c)

        except Exception:
            continue

    print(f"[WARNING] Could not find logbook for seed {seed}.")
    return None


# =============================================================================
# PLOTS: METRICS
# =============================================================================

def plot_metrics_per_seed(metrics_df):
    test_cols = ["test_RMSE", "test_MAE", "test_R2", "test_Pearson_r"]

    for col in test_cols:
        plt.figure(figsize=(8, 5))
        plt.plot(metrics_df["seed"], metrics_df[col], marker="o")
        plt.xlabel("Seed")
        plt.ylabel(col)
        plt.title(f"{col} across repeated train/test splits")
        plt.grid(alpha=0.3)
        savefig(f"{col}_per_seed.png")

    # Combined compact panel as separate figures are already saved;
    # here save a grouped table-style bar chart for test metrics.
    summary_rows = []

    for col in test_cols:
        summary_rows.append({
            "metric": col,
            "mean": metrics_df[col].mean(),
            "std": metrics_df[col].std(),
            "min": metrics_df[col].min(),
            "median": metrics_df[col].median(),
            "max": metrics_df[col].max(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "G3_test_metrics_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(summary))
    plt.bar(x, summary["mean"], yerr=summary["std"], capsize=5)
    plt.xticks(x, summary["metric"], rotation=30, ha="right")
    plt.ylabel("Mean ± SD")
    plt.title("Test metrics summary across seeds")
    plt.grid(axis="y", alpha=0.3)
    savefig("test_metrics_summary_barplot.png")


def plot_inner_vs_test(metrics_df):
    plt.figure(figsize=(6, 6))
    plt.scatter(metrics_df["innerCV_RMSE_mean"], metrics_df["test_RMSE"])

    for _, row in metrics_df.iterrows():
        plt.text(row["innerCV_RMSE_mean"], row["test_RMSE"], str(int(row["seed"])), fontsize=8)

    plt.xlabel("Inner CV RMSE mean")
    plt.ylabel("Test RMSE")
    plt.title("Inner CV RMSE vs external test RMSE")
    plt.grid(alpha=0.3)
    savefig("innerCV_RMSE_vs_test_RMSE.png")

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["seed"], metrics_df["innerCV_RMSE_mean"], marker="o", label="Inner CV RMSE")
    plt.plot(metrics_df["seed"], metrics_df["test_RMSE"], marker="o", label="Test RMSE")
    plt.xlabel("Seed")
    plt.ylabel("RMSE")
    plt.title("Inner CV RMSE and test RMSE across seeds")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig("innerCV_and_test_RMSE_per_seed.png")


def plot_trainval_vs_test(metrics_df):
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["seed"], metrics_df["trainval_RMSE"], marker="o", label="Trainval RMSE")
    plt.plot(metrics_df["seed"], metrics_df["test_RMSE"], marker="o", label="Test RMSE")
    plt.xlabel("Seed")
    plt.ylabel("RMSE")
    plt.title("Trainval vs test RMSE across seeds")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig("trainval_vs_test_RMSE_per_seed.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(metrics_df["trainval_RMSE"], metrics_df["test_RMSE"])

    for _, row in metrics_df.iterrows():
        plt.text(row["trainval_RMSE"], row["test_RMSE"], str(int(row["seed"])), fontsize=8)

    plt.xlabel("Trainval RMSE")
    plt.ylabel("Test RMSE")
    plt.title("Trainval RMSE vs test RMSE")
    plt.grid(alpha=0.3)
    savefig("trainval_RMSE_vs_test_RMSE.png")


def plot_n_selected(metrics_df):
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["seed"].astype(str), metrics_df["n_selected_snps"])
    plt.xlabel("Seed")
    plt.ylabel("Number of selected SNPs")
    plt.title("Selected SNP subset size across seeds")
    plt.grid(axis="y", alpha=0.3)
    savefig("n_selected_snps_per_seed.png")

    plt.figure(figsize=(6, 5))
    plt.hist(metrics_df["n_selected_snps"], bins=8)
    plt.xlabel("Number of selected SNPs")
    plt.ylabel("Count")
    plt.title("Distribution of selected SNP subset size")
    plt.grid(axis="y", alpha=0.3)
    savefig("n_selected_snps_distribution.png")


# =============================================================================
# PLOTS: PREDICTIONS AND RESIDUALS
# =============================================================================

def plot_test_predictions(pred_all):
    y_true = pred_all["y_true"].values
    y_pred = pred_all["y_pred"].values

    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": pearson_r(y_true, y_pred),
    }

    with open(TABLE_DIR / "G3_aggregated_test_prediction_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    #plt.figure(figsize=(6, 6))
    #plt.scatter(y_true, y_pred, alpha=0.65)
    #plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.figure(figsize=(6.5, 6.5))

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

    plt.xlim(min_val - 3, max_val + 3)
    plt.ylim(min_val - 3, max_val + 3)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("Observed harvest date")
    plt.ylabel("Predicted harvest date")
    plt.title("External test predictions across repeated splits")

    text = (
        f"RMSE = {metrics['RMSE']:.2f}\n"
        f"R² = {metrics['R2']:.2f}\n"
        f"r = {metrics['Pearson_r']:.2f}"
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
    plt.xlabel("Observed Harvest_date")
    plt.ylabel("Predicted Harvest_date")
    plt.title(
        "External test predictions across all seeds\n"
        f"RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.2f}, r={metrics['Pearson_r']:.2f}"
    )
    plt.grid(alpha=0.3)
    savefig("test_ytrue_vs_ypred_all_seeds.png")

    # One scatter per seed
    for seed, df_seed in pred_all.groupby("seed"):
        y_true_s = df_seed["y_true"].values
        y_pred_s = df_seed["y_pred"].values

        min_s = min(np.min(y_true_s), np.min(y_pred_s))
        max_s = max(np.max(y_true_s), np.max(y_pred_s))

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_s, y_pred_s, alpha=0.75)
        plt.plot([min_s, max_s], [min_s, max_s], linestyle="--")
        plt.xlabel("Observed Harvest_date")
        plt.ylabel("Predicted Harvest_date")
        plt.title(f"Test predictions - seed {seed}")
        plt.grid(alpha=0.3)
        savefig(f"test_ytrue_vs_ypred_seed_{seed}.png")


def plot_residuals(pred_all):
    pred_all = pred_all.copy()
    pred_all["residual"] = pred_all["y_true"] - pred_all["y_pred"]

    plt.figure(figsize=(7, 5))
    plt.hist(pred_all["residual"], bins=25)
    plt.xlabel("Residual = observed - predicted")
    plt.ylabel("Count")
    plt.title("Distribution of test residuals across all seeds")
    plt.grid(axis="y", alpha=0.3)
    savefig("test_residuals_distribution.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(pred_all["y_pred"], pred_all["residual"], alpha=0.65)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Harvest_date")
    plt.ylabel("Residual = observed - predicted")
    plt.title("Test residuals vs predicted values")
    plt.grid(alpha=0.3)
    savefig("test_residuals_vs_predicted.png")

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

    residual_summary.to_csv(TABLE_DIR / "G3_residual_summary_by_seed.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(residual_summary["seed"], residual_summary["residual_mean"], marker="o")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Seed")
    plt.ylabel("Mean residual")
    plt.title("Mean test residual by seed")
    plt.grid(alpha=0.3)
    savefig("mean_test_residual_by_seed.png")


# =============================================================================
# PLOTS: HYPERPARAMETERS
# =============================================================================

def plot_hyperparameters(metrics_df):
    # Ridge alpha frequency
    ridge_counts = (
        metrics_df
        .groupby("best_ridge_alpha")
        .size()
        .reset_index(name="n_seeds")
        .sort_values("best_ridge_alpha")
    )
    ridge_counts.to_csv(TABLE_DIR / "G3_ridge_alpha_frequency.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.bar(ridge_counts["best_ridge_alpha"].astype(str), ridge_counts["n_seeds"])
    plt.xlabel("Best ridge_alpha")
    plt.ylabel("Number of seeds")
    plt.title("Frequency of selected ridge_alpha")
    plt.grid(axis="y", alpha=0.3)
    savefig("ridge_alpha_frequency.png")

    # Lambda size frequency
    lambda_counts = (
        metrics_df
        .groupby("best_lambda_size")
        .size()
        .reset_index(name="n_seeds")
        .sort_values("best_lambda_size")
    )
    lambda_counts.to_csv(TABLE_DIR / "G3_lambda_size_frequency.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.bar(lambda_counts["best_lambda_size"].astype(str), lambda_counts["n_seeds"])
    plt.xlabel("Best lambda_size")
    plt.ylabel("Number of seeds")
    plt.title("Frequency of selected lambda_size")
    plt.grid(axis="y", alpha=0.3)
    savefig("lambda_size_frequency.png")

    # Combined best hyperparameters
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
    combo_counts.to_csv(TABLE_DIR / "G3_best_hyperparam_combo_frequency.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(combo_counts["label"], combo_counts["n_seeds"])
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Best hyperparameter combination")
    plt.ylabel("Number of seeds")
    plt.title("Frequency of selected hyperparameter combinations")
    plt.grid(axis="y", alpha=0.3)
    savefig("best_hyperparameter_combo_frequency.png")


# =============================================================================
# PLOTS: FITNESS EVOLUTION
# =============================================================================

# def plot_fitness_evolution(metrics_df):
#     all_best_logbooks = []

#     for _, row in metrics_df.iterrows():
#         seed = int(row["seed"])
#         ridge = row["best_ridge_alpha"]
#         lam = row["best_lambda_size"]

#         logbook = load_best_logbook_for_seed(seed, ridge, lam)

#         if logbook is None:
#             continue

#         if "gen" not in logbook.columns or "min" not in logbook.columns:
#             print(f"[WARNING] Logbook for seed {seed} does not contain expected columns.")
#             continue

#         logbook = logbook.copy()
#         logbook["seed"] = seed
#         logbook["best_ridge_alpha"] = ridge
#         logbook["best_lambda_size"] = lam
#         all_best_logbooks.append(logbook)

#         plt.figure(figsize=(8, 5))
#         plt.plot(logbook["gen"], logbook["min"], marker="o", label="Best fitness")
#         if "mean" in logbook.columns:
#             plt.plot(logbook["gen"], logbook["mean"], marker="o", alpha=0.7, label="Mean fitness")
#         plt.xlabel("Generation")
#         plt.ylabel("Fitness")
#         plt.title(f"GA fitness evolution - seed {seed}")
#         plt.legend()
#         plt.grid(alpha=0.3)
#         savefig(f"fitness_evolution_seed_{seed}.png")

#     if len(all_best_logbooks) == 0:
#         print("[WARNING] No logbooks available for fitness evolution plots.")
#         return

    # log_all = pd.concat(all_best_logbooks, ignore_index=True)
    # log_all.to_csv(TABLE_DIR / "G3_best_config_logbooks_all_seeds.csv", index=False)

    # # Mean and std of best fitness across seeds per generation
    # fitness_summary = (
    #     log_all
    #     .groupby("gen")
    #     .agg(
    #         min_fitness_mean=("min", "mean"),
    #         min_fitness_std=("min", "std"),
    #         mean_fitness_mean=("mean", "mean") if "mean" in log_all.columns else ("min", "mean"),
    #     )
    #     .reset_index()
    # )

    # fitness_summary.to_csv(TABLE_DIR / "G3_fitness_evolution_summary.csv", index=False)

    # plt.figure(figsize=(8, 5))
    # x = fitness_summary["gen"].values
    # y = fitness_summary["min_fitness_mean"].values
    # yerr = fitness_summary["min_fitness_std"].fillna(0).values

    # plt.plot(x, y, marker="o", label="Mean best fitness")
    # plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.title("GA best fitness evolution across seeds")
    # plt.legend()
    # plt.grid(alpha=0.3)
    # savefig("fitness_evolution_mean_across_seeds.png")
# def plot_fitness_evolution(metrics_df):
#     all_best_logbooks = []

#     for _, row in metrics_df.iterrows():
#         seed = int(row["seed"])
#         ridge = row["best_ridge_alpha"]
#         lam = row["best_lambda_size"]

#         logbook = load_best_logbook_for_seed(seed, ridge, lam)

#         if logbook is None:
#             continue

#         if "gen" not in logbook.columns or "min" not in logbook.columns:
#             print(f"[WARNING] Logbook for seed {seed} does not contain expected columns.")
#             continue

#         logbook = logbook.copy()
#         logbook["seed"] = seed
#         logbook["best_ridge_alpha"] = ridge
#         logbook["best_lambda_size"] = lam
#         all_best_logbooks.append(logbook)

#     if len(all_best_logbooks) == 0:
#         print("[WARNING] No logbooks available for fitness evolution plots.")
#         return

#     log_all = pd.concat(all_best_logbooks, ignore_index=True)
#     log_all.to_csv(TABLE_DIR / "G3_best_config_logbooks_all_seeds.csv", index=False)

#     # -------------------------------------------------------------------------
#     # Elegant plot: median + IQR of best fitness across seeds
#     # -------------------------------------------------------------------------

    # fitness_summary = (
    #     log_all
    #     .groupby("gen")
    #     .agg(
    #         fitness_median=("min", "median"),
    #         fitness_q25=("min", lambda x: np.percentile(x, 25)),
    #         fitness_q75=("min", lambda x: np.percentile(x, 75)),
    #         fitness_mean=("min", "mean"),
    #         fitness_std=("min", "std"),
    #     )
    #     .reset_index()
    # )

    # fitness_summary.to_csv(TABLE_DIR / "G3_fitness_evolution_summary.csv", index=False)

    # plt.figure(figsize=(8, 5))

    # x = fitness_summary["gen"].values
    # median = fitness_summary["fitness_median"].values
    # q25 = fitness_summary["fitness_q25"].values
    # q75 = fitness_summary["fitness_q75"].values

    # plt.plot(
    #     x,
    #     median,
    #     linewidth=2.5,
    #     label="Median best fitness"
    # )

    # plt.fill_between(
    #     x,
    #     q25,
    #     q75,
    #     alpha=0.18,
    #     label="Interquartile range"
    # )

    # plt.xlabel("Generation")
    # plt.ylabel("Fitness score, lower is better")
    # plt.title("GA convergence across repeated runs")
    # plt.grid(alpha=0.25)
    # plt.legend(frameon=False)

    # savefig("fitness_evolution_median_IQR_across_seeds.png")

    # # -------------------------------------------------------------------------
    # # Optional: individual seed curves, lighter and cleaner
    # # -------------------------------------------------------------------------

    # plt.figure(figsize=(8, 5))

    # for seed, sub in log_all.groupby("seed"):
    #     plt.plot(
    #         sub["gen"],
    #         sub["min"],
    #         linewidth=1.2,
    #         alpha=0.35
    #     )

    # plt.plot(
    #     x,
    #     median,
    #     linewidth=2.8,
    #     label="Median"
    # )

    # plt.xlabel("Generation")
    # plt.ylabel("Fitness score, lower is better")
    # plt.title("GA best-fitness trajectories across seeds")
    # plt.grid(alpha=0.25)
    # plt.legend(frameon=False)

    # savefig("fitness_evolution_individual_seeds_with_median.png")
def plot_fitness_evolution(metrics_df):
    """
    Creates thesis-style GA convergence plots.

    Produced figures:
    1. fitness_evolution_mean_points_line.png
       - mean best fitness across seeds
       - points + connected line, similar to the course GA plot

    2. fitness_evolution_median_points_line.png
       - median best fitness across seeds
       - points + connected line

    3. fitness_evolution_median_IQR_thesis.png
       - median best fitness + IQR band
       - more rigorous and thesis-ready

    4. fitness_evolution_individual_seeds_with_median_supplementary.png
       - individual seed trajectories + median
       - useful for appendix/supplementary material
    """

    all_best_logbooks = []

    for _, row in metrics_df.iterrows():
        seed = int(row["seed"])
        ridge = row["best_ridge_alpha"]
        lam = row["best_lambda_size"]

        logbook = load_best_logbook_for_seed(seed, ridge, lam)

        if logbook is None:
            continue

        if "gen" not in logbook.columns or "min" not in logbook.columns:
            print(f"[WARNING] Logbook for seed {seed} does not contain expected columns.")
            continue

        logbook = logbook.copy()
        logbook["seed"] = seed
        logbook["best_ridge_alpha"] = ridge
        logbook["best_lambda_size"] = lam

        all_best_logbooks.append(logbook)

    if len(all_best_logbooks) == 0:
        print("[WARNING] No logbooks available for fitness evolution plots.")
        return

    log_all = pd.concat(all_best_logbooks, ignore_index=True)
    log_all.to_csv(TABLE_DIR / "G3_best_config_logbooks_all_seeds.csv", index=False)

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

    fitness_summary.to_csv(TABLE_DIR / "G3_fitness_evolution_summary.csv", index=False)

    x = fitness_summary["gen"].values
    y_mean = fitness_summary["fitness_mean"].values
    y_std = fitness_summary["fitness_std"].fillna(0).values
    y_median = fitness_summary["fitness_median"].values
    y_q25 = fitness_summary["fitness_q25"].values
    y_q75 = fitness_summary["fitness_q75"].values

    # =========================================================================
    # 1. Stile terza immagine: mean + points + line
    # =========================================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        y_mean,
        #marker="o",
        #markersize=4.5,
        linewidth=2.0,
        label="Mean best fitness"
    )

    plt.xlabel("Generation")
    plt.ylabel("Best fitness, lower is better")
    plt.title("GA fitness evolution across repeated runs")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    savefig("fitness_evolution_mean_points_line.png")

    # =========================================================================
    # 2. Stile terza immagine, ma con mediana
    # =========================================================================

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
    plt.title("GA fitness evolution across repeated runs")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    savefig("fitness_evolution_median_points_line.png")

    # =========================================================================
    # 3. Versione elegante da tesi: mediana + IQR
    # =========================================================================

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
    plt.title("GA convergence across repeated runs")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    savefig("fitness_evolution_median_IQR_thesis.png")

    # =========================================================================
    # 4. Versione supplementare: traiettorie individuali + mediana
    # =========================================================================

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
    plt.title("GA best-fitness trajectories across seeds")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    savefig("fitness_evolution_individual_seeds_with_median_supplementary.png")

# =============================================================================
# REPORT
# =============================================================================

def write_report(metrics_df, pred_all):
    report_path = RUN_DIR / "G3_prediction_and_ga_report.txt"

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

    y_true = pred_all["y_true"].values
    y_pred = pred_all["y_pred"].values

    aggregate_prediction_metrics = {
        "aggregated_test_RMSE": rmse(y_true, y_pred),
        "aggregated_test_MAE": float(mean_absolute_error(y_true, y_pred)),
        "aggregated_test_R2": float(r2_score(y_true, y_pred)),
        "aggregated_test_Pearson_r": pearson_r(y_true, y_pred),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("G3 REPORT - G2B NO-SOIL PREDICTION AND GA EVALUATION\n")
        f.write("DATASET_LABEL: no_soil_top1000_regions\n\n")
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
        f.write(f"Figures: {FIG_DIR}\n")
        f.write(f"Tables: {TABLE_DIR}\n")

    print(f"[REPORT] Saved: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 80)
    print("G3 - EVALUATE G2B NO-SOIL PREDICTION AND GA")
    print("#" * 80)

    metrics_df = read_required(METRICS_PER_SEED)
    metrics_df.to_csv(TABLE_DIR / "G3_metrics_per_seed_copy.csv", index=False)

    pred_all = collect_test_predictions()
    collect_trainval_predictions()

    plot_metrics_per_seed(metrics_df)
    plot_inner_vs_test(metrics_df)
    plot_trainval_vs_test(metrics_df)
    plot_n_selected(metrics_df)
    plot_test_predictions(pred_all)
    plot_residuals(pred_all)
    plot_hyperparameters(metrics_df)
    plot_fitness_evolution(metrics_df)
    write_report(metrics_df, pred_all)

    print("\n" + "#" * 80)
    print("G3 FINISHED")
    print("#" * 80)
    print(f"Figures saved in:\n  {FIG_DIR}")
    print(f"Tables saved in:\n  {TABLE_DIR}")


if __name__ == "__main__":
    main()
