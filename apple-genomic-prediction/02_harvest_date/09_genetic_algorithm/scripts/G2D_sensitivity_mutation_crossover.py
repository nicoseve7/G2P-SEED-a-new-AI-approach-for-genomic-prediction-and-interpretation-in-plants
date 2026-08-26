# Questo script fa una sensitivity analysis compatta su:

# CXPB  # crossover probability
# MUTPB # mutation probability

# tenendo fissi:

# POP_SIZE = 200
# N_GEN = 100

# Usa una griglia piccola e 5 seed:

# SEEDS = [42, 43, 44, 45, 46]

# Produce:

# Output/04_ga_runs/G2D_sensitivity_mutation_crossover/

# con:

# G2D_results_per_config_seed.csv
# G2D_summary_by_config.csv
# G2D_final_report.txt
# contour_test_RMSE.png
# contour_test_R2.png
# contour_test_Pearson_r.png
# contour_n_selected_snps.png

# -*- coding: utf-8 -*-

################################################################################
### G2D_sensitivity_mutation_crossover.py
###
### Sensitivity analysis for GA mutation and crossover probabilities.
###
### Starting point:
###   Same data and same logic as G2B:
###   - repeated 80/20 train-test split
###   - inner 3-fold CV fitness inside trainval
###   - Ridge evaluation with PCA as fixed covariates
###   - GA selects SNPs only
###
### Difference from G2B:
###   This script tests multiple CXPB / MUTPB combinations.
###
### Goal:
###   Check whether GA performance and selected subset size are robust to
###   mutation/crossover settings.
###
### Input:
###   Output/03_ga_inputs/
###       X_snp_top1000_regions_50kb_Harvest_date.csv
###       X_pca_20_Harvest_date.csv
###       y_mean_adjusted_by_genotype_Harvest_date.csv
###       snp_metadata_top1000_regions_50kb_Harvest_date.csv
###
### Output:
###   Output/04_ga_runs/G2D_sensitivity_mutation_crossover/
################################################################################


import os
import re
import json
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from deap import base, creator, tools, algorithms


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

TRAIT = "Harvest_date"
WINDOW_LABEL = "50kb"

GA_INPUT_DIR = Path("Output/03_ga_inputs")

X_SNP_FILE = GA_INPUT_DIR / "X_snp_top1000_regions_50kb_Harvest_date.csv"
X_PCA_FILE = GA_INPUT_DIR / "X_pca_20_Harvest_date.csv"
Y_FILE = GA_INPUT_DIR / "y_mean_adjusted_by_genotype_Harvest_date.csv"

OUT_DIR = Path("Output/04_ga_runs/G2D_no_soil_sensitivity_mutation_crossover")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_LABEL = "no_soil_top1000_regions"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Sensitivity seeds.
# Uso 5 seed per contenere i tempi.
SEEDS = [42, 43, 44, 45, 46]

TEST_SIZE = 0.20
INNER_CV_FOLDS = 3

# Qui manteniamo la stessa logica di G2B.
# Per ogni seed cambia anche train/test esterno.
FIX_DATA_SPLITS = False
SPLIT_SEED = 42

# Hyperparameter grid for Ridge/lambda.
# Per sensitivity analysis possiamo tenere la stessa griglia di G2B.
RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0]
LAMBDA_SIZE_GRID = [0.02, 0.05, 0.10, 0.20]

# Fixed GA size.
POP_SIZE = 200
N_GEN = 100
TOURNSIZE = 3

# Sensitivity grid.
# CXPB = crossover probability
# MUTPB = mutation probability
CXPB_GRID = [0.2, 0.5, 0.8]
MUTPB_GRID = [0.1, 0.2, 0.4]

# Baseline G2B setting is included:
# CXPB=0.5, MUTPB=0.2

INIT_PROB_ONE = 0.05
BIT_FLIP_INDPB = None

MIN_SELECTED_SNPS = 1
MAX_SELECTED_SNPS = None

USE_PCA = True
SAVE_LOGBOOKS = False

DPI = 300


# =============================================================================
# UTILS
# =============================================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson_r(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def clean_genotype_id(x):
    return str(x).strip()


def detect_column(df, candidates, required=True, label="column"):
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if required:
        raise ValueError(
            f"Could not detect {label}. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def safe_read_csv(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def savefig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")


# =============================================================================
# LOAD INPUTS
# =============================================================================

def load_ga_inputs():
    print("\n" + "=" * 80)
    print("[LOAD] Loading GA inputs")
    print("=" * 80)

    print(f"[INFO] Loading SNP matrix:\n  {X_SNP_FILE}")
    X_snp_df = safe_read_csv(X_SNP_FILE)

    print(f"[INFO] Loading PCA matrix:\n  {X_PCA_FILE}")
    X_pca_df = safe_read_csv(X_PCA_FILE)

    print(f"[INFO] Loading target:\n  {Y_FILE}")
    y_df = safe_read_csv(Y_FILE)

    genotype_col_snp = detect_column(
        X_snp_df,
        ["Genotype", "genotype", "ID", "id"],
        required=True,
        label="genotype column in X_snp"
    )

    genotype_col_pca = detect_column(
        X_pca_df,
        ["Genotype", "genotype", "ID", "id"],
        required=True,
        label="genotype column in X_pca"
    )

    genotype_col_y = detect_column(
        y_df,
        ["Genotype", "genotype", "ID", "id"],
        required=True,
        label="genotype column in y"
    )

    target_col = detect_column(
        y_df,
        [
            "Harvest_date_adjusted_mean",
            "mean_Harvest_date_adjusted",
            "Harvest_date_mean_adjusted",
            "Harvest_date_adjusted",
            "Harvest_date",
            "target",
            "y"
        ],
        required=False,
        label="target column"
    )

    if target_col is None:
        numeric_cols = [
            c for c in y_df.columns
            if c != genotype_col_y and pd.api.types.is_numeric_dtype(y_df[c])
        ]

        if len(numeric_cols) == 1:
            target_col = numeric_cols[0]
            print(f"[INFO] Target column automatically selected: {target_col}")
        else:
            raise ValueError(
                "Could not detect target column. "
                f"Available columns: {list(y_df.columns)}"
            )

    X_snp_df = X_snp_df.rename(columns={genotype_col_snp: "Genotype"})
    X_pca_df = X_pca_df.rename(columns={genotype_col_pca: "Genotype"})
    y_df = y_df.rename(columns={genotype_col_y: "Genotype", target_col: "y"})

    X_snp_df["Genotype"] = X_snp_df["Genotype"].apply(clean_genotype_id)
    X_pca_df["Genotype"] = X_pca_df["Genotype"].apply(clean_genotype_id)
    y_df["Genotype"] = y_df["Genotype"].apply(clean_genotype_id)

    y_df = y_df[["Genotype", "y"]].copy()

    snp_cols = [c for c in X_snp_df.columns if c != "Genotype"]
    pca_cols = [c for c in X_pca_df.columns if c != "Genotype"]

    df = y_df.merge(X_snp_df, on="Genotype", how="inner")

    if USE_PCA:
        df = df.merge(X_pca_df, on="Genotype", how="inner")
    else:
        pca_cols = []

    df = df.dropna(subset=["y"]).copy()

    if df["Genotype"].duplicated().any():
        print("[WARNING] Duplicate genotypes detected. Keeping first occurrence.")
        df = df.drop_duplicates(subset=["Genotype"], keep="first").copy()

    valid_snp_cols = []
    dropped_rows = []

    for c in snp_cols:
        if c not in df.columns:
            dropped_rows.append({"snp": c, "reason": "not_in_merged_df"})
            continue

        if df[c].isna().all():
            dropped_rows.append({"snp": c, "reason": "all_missing"})
            continue

        if df[c].nunique(dropna=True) <= 1:
            dropped_rows.append({"snp": c, "reason": "zero_variance"})
            continue

        valid_snp_cols.append(c)

    dropped_df = pd.DataFrame(dropped_rows)
    dropped_df.to_csv(TABLE_DIR / "G2D_dropped_snp_columns.csv", index=False)

    for c in valid_snp_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())

    for c in pca_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())

    print("\n[INFO] Input summary")
    print(f"  Final aligned dataset shape: {df.shape}")
    print(f"  Number of genotypes: {df['Genotype'].nunique()}")
    print(f"  SNP columns valid: {len(valid_snp_cols)}")
    print(f"  PCA columns: {len(pca_cols)}")

    metadata = {
        "TRAIT": TRAIT,
        "WINDOW_LABEL": WINDOW_LABEL,
        "X_SNP_FILE": str(X_SNP_FILE),
        "X_PCA_FILE": str(X_PCA_FILE),
        "Y_FILE": str(Y_FILE),
        "n_genotypes": int(df["Genotype"].nunique()),
        "n_snp_cols_valid": int(len(valid_snp_cols)),
        "n_pca_cols": int(len(pca_cols)),
        "USE_PCA": USE_PCA,
    }

    with open(OUT_DIR / "G2D_dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return df, valid_snp_cols, pca_cols


# =============================================================================
# INNER CV FITNESS
# =============================================================================

def evaluate_mask_inner_cv(
    selected_mask,
    X_snp_trainval_raw,
    X_pca_trainval_raw,
    y_trainval,
    inner_cv_splits,
    ridge_alpha,
):
    fold_metrics = []

    for inner_train_idx, inner_val_idx in inner_cv_splits:
        X_snp_inner_train_raw = X_snp_trainval_raw[inner_train_idx]
        X_snp_inner_val_raw = X_snp_trainval_raw[inner_val_idx]

        X_pca_inner_train_raw = X_pca_trainval_raw[inner_train_idx]
        X_pca_inner_val_raw = X_pca_trainval_raw[inner_val_idx]

        y_inner_train = y_trainval[inner_train_idx]
        y_inner_val = y_trainval[inner_val_idx]

        snp_scaler = StandardScaler()
        X_snp_inner_train = snp_scaler.fit_transform(X_snp_inner_train_raw)
        X_snp_inner_val = snp_scaler.transform(X_snp_inner_val_raw)

        if X_pca_trainval_raw.shape[1] > 0:
            pca_scaler = StandardScaler()
            X_pca_inner_train = pca_scaler.fit_transform(X_pca_inner_train_raw)
            X_pca_inner_val = pca_scaler.transform(X_pca_inner_val_raw)
        else:
            X_pca_inner_train = np.zeros((len(inner_train_idx), 0))
            X_pca_inner_val = np.zeros((len(inner_val_idx), 0))

        X_train_selected = X_snp_inner_train[:, selected_mask]
        X_val_selected = X_snp_inner_val[:, selected_mask]

        if X_pca_inner_train.shape[1] > 0:
            X_train_model = np.hstack([X_train_selected, X_pca_inner_train])
            X_val_model = np.hstack([X_val_selected, X_pca_inner_val])
        else:
            X_train_model = X_train_selected
            X_val_model = X_val_selected

        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train_model, y_inner_train)
        pred_val = model.predict(X_val_model)

        fold_metrics.append({
            "RMSE": rmse(y_inner_val, pred_val),
            "MAE": float(mean_absolute_error(y_inner_val, pred_val)),
            "R2": float(r2_score(y_inner_val, pred_val)),
            "Pearson_r": pearson_r(y_inner_val, pred_val),
        })

    rmse_values = [m["RMSE"] for m in fold_metrics]
    mae_values = [m["MAE"] for m in fold_metrics]
    r2_values = [m["R2"] for m in fold_metrics]
    r_values = [m["Pearson_r"] for m in fold_metrics]

    return {
        "innerCV_RMSE_mean": float(np.nanmean(rmse_values)),
        "innerCV_RMSE_std": float(np.nanstd(rmse_values, ddof=1)),
        "innerCV_MAE_mean": float(np.nanmean(mae_values)),
        "innerCV_MAE_std": float(np.nanstd(mae_values, ddof=1)),
        "innerCV_R2_mean": float(np.nanmean(r2_values)),
        "innerCV_R2_std": float(np.nanstd(r2_values, ddof=1)),
        "innerCV_Pearson_r_mean": float(np.nanmean(r_values)),
        "innerCV_Pearson_r_std": float(np.nanstd(r_values, ddof=1)),
    }


# =============================================================================
# GA CORE
# =============================================================================

def make_toolbox_inner_cv(
    X_snp_trainval_raw,
    X_pca_trainval_raw,
    y_trainval,
    inner_cv_splits,
    ridge_alpha,
    lambda_size,
    seed,
    cxpb,
    mutpb,
):
    n_snps = X_snp_trainval_raw.shape[1]

    if BIT_FLIP_INDPB is None:
        indpb = 1.0 / n_snps
    else:
        indpb = BIT_FLIP_INDPB

    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    rng = np.random.default_rng(seed)

    def init_individual():
        arr = rng.binomial(1, INIT_PROB_ONE, size=n_snps).astype(int)

        if arr.sum() < MIN_SELECTED_SNPS:
            idx = rng.choice(n_snps, size=MIN_SELECTED_SNPS, replace=False)
            arr[idx] = 1

        if MAX_SELECTED_SNPS is not None and arr.sum() > MAX_SELECTED_SNPS:
            active = np.where(arr == 1)[0]
            keep = rng.choice(active, size=MAX_SELECTED_SNPS, replace=False)
            arr[:] = 0
            arr[keep] = 1

        return creator.Individual(arr.tolist())

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        mask = np.array(individual, dtype=bool)
        n_selected = int(mask.sum())

        if n_selected < MIN_SELECTED_SNPS:
            return (1e9,)

        if MAX_SELECTED_SNPS is not None and n_selected > MAX_SELECTED_SNPS:
            return (1e9,)

        try:
            metrics = evaluate_mask_inner_cv(
                selected_mask=mask,
                X_snp_trainval_raw=X_snp_trainval_raw,
                X_pca_trainval_raw=X_pca_trainval_raw,
                y_trainval=y_trainval,
                inner_cv_splits=inner_cv_splits,
                ridge_alpha=ridge_alpha,
            )

            mean_rmse = metrics["innerCV_RMSE_mean"]
            fitness = mean_rmse + lambda_size * n_selected

            if not np.isfinite(fitness):
                return (1e9,)

            return (float(fitness),)

        except Exception:
            return (1e9,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

    return toolbox


def run_ga_once_inner_cv(
    X_snp_trainval_raw,
    X_pca_trainval_raw,
    y_trainval,
    inner_cv_splits,
    ridge_alpha,
    lambda_size,
    seed,
    cxpb,
    mutpb,
    verbose=False,
):
    set_all_seeds(seed)

    toolbox = make_toolbox_inner_cv(
        X_snp_trainval_raw=X_snp_trainval_raw,
        X_pca_trainval_raw=X_pca_trainval_raw,
        y_trainval=y_trainval,
        inner_cv_splits=inner_cv_splits,
        ridge_alpha=ridge_alpha,
        lambda_size=lambda_size,
        seed=seed,
        cxpb=cxpb,
        mutpb=mutpb,
    )

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("std", np.std)

    pop, logbook = algorithms.eaSimple(
        population=pop,
        toolbox=toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=N_GEN,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    best_ind = hof[0]
    best_mask = np.array(best_ind, dtype=bool)
    best_fitness = float(best_ind.fitness.values[0])
    logbook_df = pd.DataFrame(logbook)

    return best_mask, best_fitness, logbook_df


def evaluate_subset_final(
    X_snp_train_raw,
    X_pca_train_raw,
    y_train,
    X_snp_eval_raw,
    X_pca_eval_raw,
    y_eval,
    selected_mask,
    ridge_alpha,
):
    snp_scaler = StandardScaler()
    X_snp_train = snp_scaler.fit_transform(X_snp_train_raw)
    X_snp_eval = snp_scaler.transform(X_snp_eval_raw)

    if X_pca_train_raw.shape[1] > 0:
        pca_scaler = StandardScaler()
        X_pca_train = pca_scaler.fit_transform(X_pca_train_raw)
        X_pca_eval = pca_scaler.transform(X_pca_eval_raw)
    else:
        X_pca_train = np.zeros((X_snp_train_raw.shape[0], 0))
        X_pca_eval = np.zeros((X_snp_eval_raw.shape[0], 0))

    X_train_selected = X_snp_train[:, selected_mask]
    X_eval_selected = X_snp_eval[:, selected_mask]

    if X_pca_train.shape[1] > 0:
        X_train_model = np.hstack([X_train_selected, X_pca_train])
        X_eval_model = np.hstack([X_eval_selected, X_pca_eval])
    else:
        X_train_model = X_train_selected
        X_eval_model = X_eval_selected

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train_model, y_train)
    pred = model.predict(X_eval_model)

    metrics = {
        "RMSE": rmse(y_eval, pred),
        "MAE": float(mean_absolute_error(y_eval, pred)),
        "R2": float(r2_score(y_eval, pred)),
        "Pearson_r": pearson_r(y_eval, pred),
    }

    return metrics, pred, model


# =============================================================================
# ONE CONFIG + ONE SEED
# =============================================================================

def run_one_config_seed(df, snp_cols, pca_cols, seed, cxpb, mutpb):
    print("\n" + "=" * 80)
    print(f"[RUN] seed={seed}, CXPB={cxpb}, MUTPB={mutpb}")
    print("=" * 80)

    set_all_seeds(seed)

    genotypes = df["Genotype"].values
    y = df["y"].values.astype(float)

    X_snp_raw = df[snp_cols].values.astype(float)

    if USE_PCA and len(pca_cols) > 0:
        X_pca_raw = df[pca_cols].values.astype(float)
    else:
        X_pca_raw = np.zeros((df.shape[0], 0), dtype=float)

    idx_all = np.arange(df.shape[0])

    split_seed = SPLIT_SEED if FIX_DATA_SPLITS else seed

    idx_trainval, idx_test = train_test_split(
        idx_all,
        test_size=TEST_SIZE,
        random_state=split_seed,
        shuffle=True,
    )

    X_snp_trainval_raw = X_snp_raw[idx_trainval]
    X_pca_trainval_raw = X_pca_raw[idx_trainval]
    y_trainval = y[idx_trainval]

    X_snp_test_raw = X_snp_raw[idx_test]
    X_pca_test_raw = X_pca_raw[idx_test]
    y_test = y[idx_test]

    inner_cv = KFold(
        n_splits=INNER_CV_FOLDS,
        shuffle=True,
        random_state=split_seed + 2000,
    )

    inner_cv_splits = list(inner_cv.split(np.arange(len(idx_trainval))))

    best = None
    tuning_rows = []

    for ridge_alpha in RIDGE_ALPHA_GRID:
        for lambda_size in LAMBDA_SIZE_GRID:

            combo_seed = (
                seed
                + int(ridge_alpha * 1000)
                + int(lambda_size * 10000)
                + int(cxpb * 100)
                + int(mutpb * 1000)
            )

            print(
                f"[TUNE] seed={seed}, cxpb={cxpb}, mutpb={mutpb}, "
                f"ridge={ridge_alpha}, lambda={lambda_size}"
            )

            selected_mask, best_fitness, logbook_df = run_ga_once_inner_cv(
                X_snp_trainval_raw=X_snp_trainval_raw,
                X_pca_trainval_raw=X_pca_trainval_raw,
                y_trainval=y_trainval,
                inner_cv_splits=inner_cv_splits,
                ridge_alpha=ridge_alpha,
                lambda_size=lambda_size,
                seed=combo_seed,
                cxpb=cxpb,
                mutpb=mutpb,
                verbose=False,
            )

            inner_metrics = evaluate_mask_inner_cv(
                selected_mask=selected_mask,
                X_snp_trainval_raw=X_snp_trainval_raw,
                X_pca_trainval_raw=X_pca_trainval_raw,
                y_trainval=y_trainval,
                inner_cv_splits=inner_cv_splits,
                ridge_alpha=ridge_alpha,
            )

            n_selected = int(selected_mask.sum())

            row = {
                "seed": seed,
                "split_seed": split_seed,
                "cxpb": cxpb,
                "mutpb": mutpb,
                "ridge_alpha": ridge_alpha,
                "lambda_size": lambda_size,
                "ga_best_fitness": best_fitness,
                "n_selected": n_selected,
                **inner_metrics,
            }

            tuning_rows.append(row)

            if best is None or best_fitness < best["ga_best_fitness"]:
                best = row.copy()
                best["selected_mask"] = selected_mask.copy()

            print(
                f"    fitness={best_fitness:.3f}, "
                f"innerCV_RMSE={inner_metrics['innerCV_RMSE_mean']:.3f}, "
                f"n_selected={n_selected}"
            )

    best_ridge_alpha = best["ridge_alpha"]
    best_lambda_size = best["lambda_size"]
    selected_mask = best["selected_mask"]

    test_metrics, test_pred, _ = evaluate_subset_final(
        X_snp_train_raw=X_snp_trainval_raw,
        X_pca_train_raw=X_pca_trainval_raw,
        y_train=y_trainval,
        X_snp_eval_raw=X_snp_test_raw,
        X_pca_eval_raw=X_pca_test_raw,
        y_eval=y_test,
        selected_mask=selected_mask,
        ridge_alpha=best_ridge_alpha,
    )

    trainval_metrics, trainval_pred, _ = evaluate_subset_final(
        X_snp_train_raw=X_snp_trainval_raw,
        X_pca_train_raw=X_pca_trainval_raw,
        y_train=y_trainval,
        X_snp_eval_raw=X_snp_trainval_raw,
        X_pca_eval_raw=X_pca_trainval_raw,
        y_eval=y_trainval,
        selected_mask=selected_mask,
        ridge_alpha=best_ridge_alpha,
    )

    result = {
        "seed": seed,
        "split_seed": split_seed,
        "cxpb": cxpb,
        "mutpb": mutpb,

        "best_ridge_alpha": best_ridge_alpha,
        "best_lambda_size": best_lambda_size,

        "best_ga_fitness": best["ga_best_fitness"],

        "innerCV_RMSE_mean": best["innerCV_RMSE_mean"],
        "innerCV_RMSE_std": best["innerCV_RMSE_std"],
        "innerCV_MAE_mean": best["innerCV_MAE_mean"],
        "innerCV_R2_mean": best["innerCV_R2_mean"],
        "innerCV_Pearson_r_mean": best["innerCV_Pearson_r_mean"],

        "trainval_RMSE": trainval_metrics["RMSE"],
        "trainval_MAE": trainval_metrics["MAE"],
        "trainval_R2": trainval_metrics["R2"],
        "trainval_Pearson_r": trainval_metrics["Pearson_r"],

        "test_RMSE": test_metrics["RMSE"],
        "test_MAE": test_metrics["MAE"],
        "test_R2": test_metrics["R2"],
        "test_Pearson_r": test_metrics["Pearson_r"],

        "n_selected_snps": int(selected_mask.sum()),
    }

    print(
        f"[RESULT] seed={seed}, cxpb={cxpb}, mutpb={mutpb} | "
        f"test_RMSE={test_metrics['RMSE']:.3f}, "
        f"test_R2={test_metrics['R2']:.3f}, "
        f"r={test_metrics['Pearson_r']:.3f}, "
        f"nSNP={int(selected_mask.sum())}"
    )

    return result, pd.DataFrame(tuning_rows)


# =============================================================================
# SUMMARY AND PLOTS
# =============================================================================

def summarize_results(results_df):
    summary = (
        results_df
        .groupby(["cxpb", "mutpb"])
        .agg(
            n_runs=("seed", "count"),

            test_RMSE_mean=("test_RMSE", "mean"),
            test_RMSE_std=("test_RMSE", "std"),
            test_MAE_mean=("test_MAE", "mean"),
            test_MAE_std=("test_MAE", "std"),
            test_R2_mean=("test_R2", "mean"),
            test_R2_std=("test_R2", "std"),
            test_Pearson_r_mean=("test_Pearson_r", "mean"),
            test_Pearson_r_std=("test_Pearson_r", "std"),

            innerCV_RMSE_mean=("innerCV_RMSE_mean", "mean"),
            innerCV_RMSE_std=("innerCV_RMSE_mean", "std"),

            n_selected_snps_mean=("n_selected_snps", "mean"),
            n_selected_snps_std=("n_selected_snps", "std"),
        )
        .reset_index()
    )

    summary.to_csv(OUT_DIR / "G2D_summary_by_config.csv", index=False)

    return summary


def make_heatmap_table(summary, value_col):
    pivot = summary.pivot(index="mutpb", columns="cxpb", values=value_col)
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def plot_heatmap(summary, value_col, title, filename, cmap="viridis", lower_better=False):
    pivot = make_heatmap_table(summary, value_col)

    plt.figure(figsize=(7, 5))

    data = pivot.values

    im = plt.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )

    plt.colorbar(im, label=value_col)

    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=[str(c) for c in pivot.columns]
    )

    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=[str(m) for m in pivot.index]
    )

    plt.xlabel("Crossover probability (CXPB)")
    plt.ylabel("Mutation probability (MUTPB)")
    plt.title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if pd.notna(val):
                plt.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val < np.nanmean(data) else "black",
                    fontsize=9,
                )

    savefig(filename)


def plot_contour(summary, value_col, title, filename, lower_better=False):
    pivot = make_heatmap_table(summary, value_col)

    x = np.array(pivot.columns, dtype=float)
    y = np.array(pivot.index, dtype=float)
    z = pivot.values.astype(float)

    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(7, 5))

    contour = plt.contourf(X, Y, z, levels=20, cmap="viridis")
    lines = plt.contour(X, Y, z, levels=8, colors="black", alpha=0.35, linewidths=0.8)
    plt.clabel(lines, inline=True, fontsize=8, fmt="%.2f")

    plt.scatter(X.flatten(), Y.flatten(), s=35, edgecolor="black", linewidth=0.5)

    plt.colorbar(contour, label=value_col)

    plt.xlabel("Crossover probability (CXPB)")
    plt.ylabel("Mutation probability (MUTPB)")
    plt.title(title)

    savefig(filename)


def make_all_plots(summary):
    summary.to_csv(TABLE_DIR / "G2D_summary_by_config_copy.csv", index=False)

    # Heatmaps
    plot_heatmap(
        summary,
        value_col="innerCV_RMSE_mean",
        title="Sensitivity landscape: mean inner-CV RMSE",
        filename="heatmap_innerCV_RMSE_mean.png",
        lower_better=True,
    )

    # plot_heatmap(
    #     summary,
    #     value_col="test_R2_mean",
    #     title="Sensitivity landscape: mean test R²",
    #     filename="heatmap_test_R2_mean.png",
    # )

    # plot_heatmap(
    #     summary,
    #     value_col="test_Pearson_r_mean",
    #     title="Sensitivity landscape: mean test Pearson r",
    #     filename="heatmap_test_Pearson_r_mean.png",
    # )

    # plot_heatmap(
    #     summary,
    #     value_col="n_selected_snps_mean",
    #     title="Sensitivity landscape: mean number of selected SNPs",
    #     filename="heatmap_n_selected_snps_mean.png",
    # )

    # Contours
    plot_contour(
        summary,
        value_col="innerCV_RMSE_mean",
        title="Mutation/crossover landscape: mean inner-CV RMSE",
        filename="contour_innerCV_RMSE_mean.png",
        lower_better=True,
    )

    # plot_contour(
    #     summary,
    #     value_col="test_R2_mean",
    #     title="Mutation/crossover landscape: mean test R²",
    #     filename="contour_test_R2_mean.png",
    # )

    # plot_contour(
    #     summary,
    #     value_col="test_Pearson_r_mean",
    #     title="Mutation/crossover landscape: mean test Pearson r",
    #     filename="contour_test_Pearson_r_mean.png",
    # )

    # plot_contour(
    #     summary,
    #     value_col="n_selected_snps_mean",
    #     title="Mutation/crossover landscape: mean selected SNPs",
    #     filename="contour_n_selected_snps_mean.png",
    # )

    # Line plot by configuration
    summary = summary.copy()
    summary["config"] = (
        "cx=" + summary["cxpb"].astype(str)
        + ", mut=" + summary["mutpb"].astype(str)
    )

    summary_sorted = summary.sort_values("innerCV_RMSE_mean", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.bar(summary_sorted["config"], summary_sorted["innerCV_RMSE_mean"], yerr=summary_sorted["innerCV_RMSE_std"], capsize=4)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Inner-CV RMSE, mean ± SD")
    plt.xlabel("GA configuration")
    plt.title("GA sensitivity: inner-CV RMSE by mutation/crossover configuration")
    plt.grid(axis="y", alpha=0.3)
    savefig("bar_innerCV_RMSE_by_config.png")


def write_report(results_df, summary):
    report_path = OUT_DIR / "G2D_final_report.txt"

    # best_rmse = summary.sort_values("test_RMSE_mean", ascending=True).iloc[0]
    # best_r2 = summary.sort_values("test_R2_mean", ascending=False).iloc[0]
    # best_r = summary.sort_values("test_Pearson_r_mean", ascending=False).iloc[0]

    best_innercv = summary.sort_values(
        ["innerCV_RMSE_mean", "innerCV_RMSE_std"],
        ascending=[True, True]
    ).iloc[0]

    baseline = summary[
        (np.isclose(summary["cxpb"], 0.5))
        & (np.isclose(summary["mutpb"], 0.2))
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("G2D FINAL REPORT - NO SOIL MUTATION/CROSSOVER SENSITIVITY\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"SEEDS: {SEEDS}\n")
        f.write(f"DATASET_LABEL: {DATASET_LABEL}\n")
        f.write(f"CXPB_GRID: {CXPB_GRID}\n")
        f.write(f"MUTPB_GRID: {MUTPB_GRID}\n")
        f.write(f"POP_SIZE: {POP_SIZE}\n")
        f.write(f"N_GEN: {N_GEN}\n")
        f.write(f"INNER_CV_FOLDS: {INNER_CV_FOLDS}\n")
        f.write(f"RIDGE_ALPHA_GRID: {RIDGE_ALPHA_GRID}\n")
        f.write(f"LAMBDA_SIZE_GRID: {LAMBDA_SIZE_GRID}\n\n")

        # f.write("BEST CONFIGURATION BY TEST RMSE\n")
        # f.write("-" * 80 + "\n")
        # f.write(best_rmse.to_string())
        # f.write("\n\n")

        # f.write("BEST CONFIGURATION BY TEST R2\n")
        # f.write("-" * 80 + "\n")
        # f.write(best_r2.to_string())
        # f.write("\n\n")

        # f.write("BEST CONFIGURATION BY TEST PEARSON R\n")
        # f.write("-" * 80 + "\n")
        # f.write(best_r.to_string())
        # f.write("\n\n")

        f.write("SELECTED CONFIGURATION BY INNER-CV RMSE\n")
        f.write("-" * 80 + "\n")
        f.write(best_innercv.to_string())
        f.write("\n\n")

        f.write("BASELINE CONFIGURATION CXPB=0.5, MUTPB=0.2\n")
        f.write("-" * 80 + "\n")
        if baseline.shape[0] > 0:
            f.write(baseline.iloc[0].to_string())
        else:
            f.write("Baseline configuration not found in grid.")
        f.write("\n\n")

        f.write("FULL SUMMARY TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")

        f.write("INTERPRETATION NOTE\n")
        f.write("-" * 80 + "\n")
        f.write(
            "This sensitivity analysis evaluates whether the GA refinement is strongly dependent "
            "on mutation and crossover probabilities. Configurations were compared using the "
            "mean inner three-fold cross-validation RMSE across five seeds. External test "
            "performance was not used to select the crossover and mutation probabilities. "
            "The purpose was to assess robustness and identify a reasonable configuration "
            "for the final multi-seed GA analysis.\n"
        )

    print(f"[REPORT] Saved: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 80)
    print("G2D - GA MUTATION/CROSSOVER SENSITIVITY - NO SOIL")
    print("#" * 80)

    config = {
        "TRAIT": TRAIT,
        "WINDOW_LABEL": WINDOW_LABEL,
        "DATASET_LABEL": DATASET_LABEL,
        "SEEDS": SEEDS,
        "TEST_SIZE": TEST_SIZE,
        "INNER_CV_FOLDS": INNER_CV_FOLDS,
        "RIDGE_ALPHA_GRID": RIDGE_ALPHA_GRID,
        "LAMBDA_SIZE_GRID": LAMBDA_SIZE_GRID,
        "POP_SIZE": POP_SIZE,
        "N_GEN": N_GEN,
        "TOURNSIZE": TOURNSIZE,
        "CXPB_GRID": CXPB_GRID,
        "MUTPB_GRID": MUTPB_GRID,
        "INIT_PROB_ONE": INIT_PROB_ONE,
        "BIT_FLIP_INDPB": BIT_FLIP_INDPB,
        "USE_PCA": USE_PCA,
    }

    with open(OUT_DIR / "G2D_config.json", "w") as f:
        json.dump(config, f, indent=4)

    df, snp_cols, pca_cols = load_ga_inputs()

    # all_results = []
    # all_tuning_rows = []

    # total_runs = len(CXPB_GRID) * len(MUTPB_GRID) * len(SEEDS)
    # run_counter = 0
    progress_file = OUT_DIR / "G2D_results_per_config_seed_PROGRESS.csv"

    if progress_file.exists():
        print(f"[RESUME] Found progress file:\n  {progress_file}")
        previous_results = pd.read_csv(progress_file)

        completed_keys = set(
            zip(
                previous_results["cxpb"].astype(float),
                previous_results["mutpb"].astype(float),
                previous_results["seed"].astype(int)
            )
        )

        all_results = previous_results.to_dict("records")

        print(f"[RESUME] Completed runs found: {len(completed_keys)}")

    else:
        print("[RESUME] No progress file found. Starting from scratch.")
        completed_keys = set()
        all_results = []

    all_tuning_rows = []

    total_runs = len(CXPB_GRID) * len(MUTPB_GRID) * len(SEEDS)
    run_counter = len(completed_keys)

    # for cxpb in CXPB_GRID:
    #     for mutpb in MUTPB_GRID:
    #         for seed in SEEDS:
    #             run_counter += 1
    #             print(f"\n[PROGRESS] Run {run_counter}/{total_runs}")

    #             result, tuning_df = run_one_config_seed(
    #                 df=df,
    #                 snp_cols=snp_cols,
    #                 pca_cols=pca_cols,
    #                 seed=seed,
    #                 cxpb=cxpb,
    #                 mutpb=mutpb,
    #             )

    #             all_results.append(result)

    #             tuning_df["cxpb"] = cxpb
    #             tuning_df["mutpb"] = mutpb
    #             tuning_df["seed"] = seed
    #             all_tuning_rows.append(tuning_df)

    #             pd.DataFrame(all_results).to_csv(
    #                 OUT_DIR / "G2D_results_per_config_seed_PROGRESS.csv",
    #                 index=False
    #             )

    for cxpb in CXPB_GRID:
        for mutpb in MUTPB_GRID:
            for seed in SEEDS:

                key = (float(cxpb), float(mutpb), int(seed))

                if key in completed_keys:
                    print(f"[SKIP] Already completed: CXPB={cxpb}, MUTPB={mutpb}, seed={seed}")
                    continue

                run_counter += 1
                print(f"\n[PROGRESS] Run {run_counter}/{total_runs}")
                print(f"[RUNNING] CXPB={cxpb}, MUTPB={mutpb}, seed={seed}")

                result, tuning_df = run_one_config_seed(
                    df=df,
                    snp_cols=snp_cols,
                    pca_cols=pca_cols,
                    seed=seed,
                    cxpb=cxpb,
                    mutpb=mutpb,
                )

                all_results.append(result)

                tuning_df["cxpb"] = cxpb
                tuning_df["mutpb"] = mutpb
                tuning_df["seed"] = seed
                all_tuning_rows.append(tuning_df)

                pd.DataFrame(all_results).to_csv(
                    OUT_DIR / "G2D_results_per_config_seed_PROGRESS.csv",
                    index=False
                )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "G2D_results_per_config_seed.csv", index=False)

    # tuning_all = pd.concat(all_tuning_rows, ignore_index=True)
    # tuning_all.to_csv(OUT_DIR / "G2D_tuning_results_all.csv", index=False)

    if len(all_tuning_rows) > 0:
        tuning_new = pd.concat(all_tuning_rows, ignore_index=True)

        tuning_partial_file = OUT_DIR / "G2D_tuning_results_all_PROGRESS.csv"

        if tuning_partial_file.exists():
            tuning_old = pd.read_csv(tuning_partial_file)
            tuning_all = pd.concat([tuning_old, tuning_new], ignore_index=True)
        else:
            tuning_all = tuning_new

        tuning_all.to_csv(tuning_partial_file, index=False)
        tuning_all.to_csv(OUT_DIR / "G2D_tuning_results_all.csv", index=False)

    else:
        print("[INFO] No new tuning rows generated in this run.")

    summary = summarize_results(results_df)
    make_all_plots(summary)
    write_report(results_df, summary)

    print("\n" + "#" * 80)
    print("G2D FINISHED")
    print("#" * 80)
    print(f"Outputs saved in:\n  {OUT_DIR.resolve()}")
    print(f"Figures saved in:\n  {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()