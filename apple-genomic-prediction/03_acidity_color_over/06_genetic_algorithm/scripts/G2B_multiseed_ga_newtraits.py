# -*- coding: utf-8 -*-

################################################################################
### G2B_multiseed_ga_newtraits.py
###
### Multi-seed Genetic Algorithm for SNP selection from no-soil SHAP-ranked
### regions for:
###   - Acidity
###   - Color_over
###
### For each trait:
###   - reads GA inputs from Output/05_ga_inputs/
###   - optionally reads best CXPB/MUTPB from G2D
###   - runs 10 seeds
###   - each seed changes:
###       1. external train/test split
###       2. inner 3-fold CV split
###       3. GA randomness
###
### Fitness:
###   mean innerCV RMSE + lambda_size * n_selected_SNPs
###
### Ridge always includes PCA as fixed covariates.
### GA selects SNPs only.
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

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from deap import base, creator, tools, algorithms


warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

WINDOW_LABEL = "50kb"
TOP_K_REGIONS = 1000
DATASET_LABEL = "no_soil_top1000_regions_newtraits"

GA_INPUT_DIR = Path("Output/05_ga_inputs")
Q9_SUMMARY_FILE = GA_INPUT_DIR / "Q9_ga_inputs_summary_all_traits.csv"

G2D_BASE_DIR = Path(
    "Output/06_ga_runs/G2D_sensitivity_mutation_crossover_newtraits"
)

BASE_OUT_DIR = Path(
    "Output/06_ga_runs/G2B_multiseed_ga_newtraits"
)
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# If True, tries to load CXPB/MUTPB from G2D best-config json.
# If not found, falls back to DEFAULT_CXPB/DEFAULT_MUTPB.
USE_G2D_BEST_CONFIG = True
DEFAULT_CXPB = 0.5
DEFAULT_MUTPB = 0.2

# Conservative version: each seed changes external train/test split.
FIX_DATA_SPLITS = False
SPLIT_SEED = 42

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

TEST_SIZE = 0.20
INNER_CV_FOLDS = 3

# Inner tuning
RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0]
LAMBDA_SIZE_GRID = [0.02, 0.05, 0.10, 0.20]

# GA settings
POP_SIZE = 200
N_GEN = 100
TOURNSIZE = 3

INIT_PROB_ONE = 0.05
BIT_FLIP_INDPB = None

MIN_SELECTED_SNPS = 1
MAX_SELECTED_SNPS = None

USE_PCA = True
SAVE_PREDICTIONS = True
SAVE_GA_LOGBOOKS = True


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
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def clean_genotype_id(x):
    return str(x).replace("G_", "").strip()


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


def split_list_cell(x):
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return []

    parts = re.split(r"[,\|; ]+", x)
    parts = [p.strip() for p in parts if p.strip() != ""]

    return parts


def safe_read_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def resolve_path_from_summary(value):
    p = Path(str(value))

    if p.exists():
        return p

    p2 = Path(".") / p

    if p2.exists():
        return p2

    return p


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def find_trait_input_files(trait: str):
    """
    Preferred: use Q9 summary.
    Fallback: glob inside Output/05_ga_inputs.
    """

    if Q9_SUMMARY_FILE.exists():
        q9 = pd.read_csv(Q9_SUMMARY_FILE)

        if "Trait" not in q9.columns:
            raise ValueError(
                f"Q9 summary found but missing Trait column:\n{Q9_SUMMARY_FILE}\n"
                f"Columns: {q9.columns.tolist()}"
            )

        sub = q9[q9["Trait"].astype(str) == trait].copy()

        if len(sub) == 0:
            raise ValueError(
                f"Trait {trait} not found in Q9 summary:\n{Q9_SUMMARY_FILE}\n"
                f"Traits found: {q9['Trait'].astype(str).unique().tolist()}"
            )

        row = sub.iloc[0]

        possible_target_cols = ["target_file", "y_file", "SAVE_TARGET"]
        possible_xsnp_cols = ["x_snp_file", "X_SNP_FILE", "snp_file"]
        possible_xpca_cols = ["x_pca_file", "X_PCA_FILE", "pca_file"]
        possible_snpmeta_cols = ["snp_metadata_file", "SNP_METADATA_FILE", "snp_meta_file"]

        target_path = None
        xsnp_path = None
        xpca_path = None
        snpmeta_path = None

        for c in possible_target_cols:
            if c in q9.columns:
                target_path = resolve_path_from_summary(row[c])
                break

        for c in possible_xsnp_cols:
            if c in q9.columns:
                xsnp_path = resolve_path_from_summary(row[c])
                break

        for c in possible_xpca_cols:
            if c in q9.columns:
                xpca_path = resolve_path_from_summary(row[c])
                break

        for c in possible_snpmeta_cols:
            if c in q9.columns:
                snpmeta_path = resolve_path_from_summary(row[c])
                break

        if (
            target_path is not None
            and xsnp_path is not None
            and xpca_path is not None
            and snpmeta_path is not None
        ):
            return xsnp_path, xpca_path, target_path, snpmeta_path

        print("[WARNING] Q9 summary found, but could not detect all path columns.")
        print("Columns found:", q9.columns.tolist())
        print("Falling back to glob search.")

    def find_one(patterns, label):
        hits = []

        for pat in patterns:
            hits.extend(list(GA_INPUT_DIR.glob(pat)))

        hits = sorted(set(hits))

        if len(hits) == 0:
            raise FileNotFoundError(
                f"Could not find {label} for trait {trait} in {GA_INPUT_DIR}"
            )

        if len(hits) > 1:
            print(f"[WARNING] Multiple candidates found for {label}, using first:")
            for h in hits[:10]:
                print(" ", h)

        return hits[0]

    xsnp_path = find_one(
        [
            f"**/X_snp_top{TOP_K_REGIONS}_regions_{WINDOW_LABEL}_{trait}.csv",
            f"**/*snp*top{TOP_K_REGIONS}*{WINDOW_LABEL}*{trait}*.csv",
        ],
        "X SNP file"
    )

    xpca_path = find_one(
        [
            f"**/X_pca_20_{trait}.csv",
            f"**/*pca*{trait}*.csv",
        ],
        "X PCA file"
    )

    target_path = find_one(
        [
            f"**/y_mean_adjusted_by_genotype_{trait}.csv",
            f"**/y_mean_by_genotype_{trait}.csv",
            f"**/*target*{trait}*.csv",
            f"**/*y*{trait}*.csv",
        ],
        "target file"
    )

    snpmeta_path = find_one(
        [
            f"**/snp_metadata_top{TOP_K_REGIONS}_regions_{WINDOW_LABEL}_{trait}.csv",
            f"**/*snp_metadata*{trait}*.csv",
        ],
        "SNP metadata file"
    )

    return xsnp_path, xpca_path, target_path, snpmeta_path


def load_ga_hyperparams_from_g2d(trait: str):
    if not USE_G2D_BEST_CONFIG:
        return DEFAULT_CXPB, DEFAULT_MUTPB, "default"

    best_file = (
        G2D_BASE_DIR
        / trait
        / f"G2D_best_config_by_RMSE_{trait}.json"
    )

    if not best_file.exists():
        print(f"[WARNING] G2D best config not found for {trait}:")
        print(best_file)
        print(f"[WARNING] Using default CXPB={DEFAULT_CXPB}, MUTPB={DEFAULT_MUTPB}")
        return DEFAULT_CXPB, DEFAULT_MUTPB, "default_missing_G2D"

    with open(best_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cxpb = float(cfg.get("cxpb", DEFAULT_CXPB))
    mutpb = float(cfg.get("mutpb", DEFAULT_MUTPB))

    print(f"[INFO] Loaded G2D best config for {trait}: CXPB={cxpb}, MUTPB={mutpb}")
    print(f"[INFO] From: {best_file}")

    return cxpb, mutpb, str(best_file)


# =============================================================================
# LOAD INPUTS
# =============================================================================

def load_ga_inputs(trait: str, out_dir: Path):
    print("\n" + "=" * 80)
    print(f"[LOAD] Loading GA inputs for trait: {trait}")
    print("=" * 80)

    x_snp_file, x_pca_file, y_file, snp_metadata_file = find_trait_input_files(trait)

    print(f"[INFO] Loading SNP matrix:\n  {x_snp_file}")
    X_snp_df = safe_read_csv(x_snp_file)

    print(f"[INFO] Loading PCA matrix:\n  {x_pca_file}")
    X_pca_df = safe_read_csv(x_pca_file)

    print(f"[INFO] Loading target:\n  {y_file}")
    y_df = safe_read_csv(y_file)

    print(f"[INFO] Loading SNP metadata:\n  {snp_metadata_file}")
    snp_meta = safe_read_csv(snp_metadata_file)

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

    target_candidates = [
        f"{trait}_adjusted_mean",
        f"{trait}_mean_adjusted",
        f"{trait}_mean",
        f"mean_{trait}",
        f"{trait}_adjusted",
        trait,
        "Harvest_date_adjusted_mean",
        "mean_Harvest_date_adjusted",
        "Harvest_date_mean_adjusted",
        "Harvest_date_adjusted",
        "Harvest_date",
        "target",
        "y",
    ]

    target_col = detect_column(
        y_df,
        target_candidates,
        required=False,
        label="target column"
    )

    if target_col is None:
        numeric_cols = [
            c for c in y_df.columns
            if c != genotype_col_y and pd.api.types.is_numeric_dtype(y_df[c])
        ]

        numeric_cols_no_counts = [
            c for c in numeric_cols
            if "n_env" not in c.lower()
            and "count" not in c.lower()
            and "obs" not in c.lower()
        ]

        if len(numeric_cols_no_counts) == 1:
            target_col = numeric_cols_no_counts[0]
            print(f"[INFO] Target column automatically selected: {target_col}")
        elif len(numeric_cols) == 1:
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

        df[c] = pd.to_numeric(df[c], errors="coerce")

        if df[c].isna().all():
            dropped_rows.append({"snp": c, "reason": "all_missing"})
            continue

        if df[c].nunique(dropna=True) <= 1:
            dropped_rows.append({"snp": c, "reason": "zero_variance"})
            continue

        valid_snp_cols.append(c)

    dropped_df = pd.DataFrame(dropped_rows)
    dropped_df.to_csv(out_dir / f"G2B_dropped_snp_columns_{trait}.csv", index=False)

    for c in valid_snp_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())

    clean_pca_cols = []

    for c in pca_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mean())
            clean_pca_cols.append(c)

    pca_cols = clean_pca_cols

    print("\n[INFO] Input summary")
    print(f"  Trait: {trait}")
    print(f"  Final aligned dataset shape: {df.shape}")
    print(f"  Number of genotypes: {df['Genotype'].nunique()}")
    print(f"  SNP columns original: {len(snp_cols)}")
    print(f"  SNP columns valid: {len(valid_snp_cols)}")
    print(f"  PCA columns: {len(pca_cols)}")
    print(f"  SNP metadata shape: {snp_meta.shape}")
    print(f"  Target mean: {df['y'].mean():.4f}")
    print(f"  Target SD: {df['y'].std(ddof=1):.4f}")

    metadata = {
        "TRAIT": trait,
        "WINDOW_LABEL": WINDOW_LABEL,
        "TOP_K_REGIONS": TOP_K_REGIONS,
        "DATASET_LABEL": DATASET_LABEL,
        "X_SNP_FILE": str(x_snp_file),
        "X_PCA_FILE": str(x_pca_file),
        "Y_FILE": str(y_file),
        "SNP_METADATA_FILE": str(snp_metadata_file),
        "target_col_used": target_col,
        "n_genotypes": int(df["Genotype"].nunique()),
        "n_snp_cols_original": int(len(snp_cols)),
        "n_snp_cols_valid": int(len(valid_snp_cols)),
        "n_pca_cols": int(len(pca_cols)),
        "n_snp_metadata_rows": int(snp_meta.shape[0]),
        "USE_PCA": USE_PCA,
        "FIX_DATA_SPLITS": FIX_DATA_SPLITS,
        "INNER_CV_FOLDS": INNER_CV_FOLDS,
    }

    with open(out_dir / f"G2B_dataset_metadata_{trait}.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return df, valid_snp_cols, pca_cols, snp_meta


# =============================================================================
# SNP METADATA PARSING
# =============================================================================

def build_snp_metadata_maps(snp_meta: pd.DataFrame, out_dir: Path):
    print("\n" + "=" * 80)
    print("[METADATA] Building SNP -> region/gene maps")
    print("=" * 80)

    snp_col = detect_column(
        snp_meta,
        [
            "SNP_ID", "snp_id", "SNP", "snp", "ID", "id",
            "marker", "Marker", "variant", "Variant"
        ],
        required=False,
        label="SNP column in metadata"
    )

    region_col = detect_column(
        snp_meta,
        [
            "region_id", "Region_ID", "region", "Region",
            "window_id", "Window_ID", "region_label", "Region_label",
            "region_name", "Region_name"
        ],
        required=False,
        label="region column in metadata"
    )

    gene_col = detect_column(
        snp_meta,
        [
            "genes_inside",
            "genes_nearby_10kb",
            "genes", "Genes",
            "gene_ids", "Gene_IDs",
            "gene_id", "Gene_ID",
            "nearby_genes", "Nearby_genes",
            "genes_in_region", "Genes_in_region",
            "annotated_genes", "Annotated_genes"
        ],
        required=False,
        label="gene column in metadata"
    )

    if snp_col is None:
        print("[WARNING] Could not detect SNP column in SNP metadata.")
        print(f"[WARNING] Available columns: {list(snp_meta.columns)}")
        return defaultdict(set), defaultdict(set)

    if region_col is None:
        print("[WARNING] Could not detect region column in SNP metadata.")
        print("[WARNING] Region stability will use UNMAPPED_REGION.")

    if gene_col is None:
        print("[WARNING] Could not detect gene column in SNP metadata.")
        print("[WARNING] Gene stability will be skipped unless gene column is available.")

    snp_to_region = defaultdict(set)
    snp_to_genes = defaultdict(set)

    for _, row in snp_meta.iterrows():
        snp = str(row[snp_col]).strip()

        if snp == "" or snp.lower() == "nan":
            continue

        if region_col is not None:
            region = str(row[region_col]).strip()
            if region != "" and region.lower() != "nan":
                snp_to_region[snp].add(region)

        if gene_col is not None:
            genes = split_list_cell(row[gene_col])
            for g in genes:
                snp_to_genes[snp].add(g)

    print(f"[INFO] SNP column detected: {snp_col}")
    print(f"[INFO] Region column detected: {region_col}")
    print(f"[INFO] Gene column detected: {gene_col}")
    print(f"[INFO] SNPs with region info: {sum(len(v) > 0 for v in snp_to_region.values())}")
    print(f"[INFO] SNPs with gene info: {sum(len(v) > 0 for v in snp_to_genes.values())}")

    metadata_cols = {
        "snp_col": snp_col,
        "region_col": region_col,
        "gene_col": gene_col,
    }

    with open(out_dir / "G2B_detected_metadata_columns.json", "w") as f:
        json.dump(metadata_cols, f, indent=4)

    return snp_to_region, snp_to_genes


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

            fitness = metrics["innerCV_RMSE_mean"] + lambda_size * n_selected

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
# ONE-SEED PIPELINE
# =============================================================================

def run_one_seed(df, snp_cols, pca_cols, trait, seed, out_dir, cxpb, mutpb):
    print("\n" + "=" * 80)
    print(f"[{trait}] [SEED {seed}] Starting run")
    print("=" * 80)

    set_all_seeds(seed)

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"[{trait}] [SEED {seed}] Split seed used: {split_seed}")
    print(f"[{trait}] [SEED {seed}] n_trainval = {len(idx_trainval)}")
    print(f"[{trait}] [SEED {seed}] n_test = {len(idx_test)}")

    split_df = pd.DataFrame({
        "Trait": trait,
        "Genotype": genotypes,
        "subset": "unused",
        "seed": seed,
        "split_seed": split_seed,
    })

    split_df.loc[idx_trainval, "subset"] = "trainval"
    split_df.loc[idx_test, "subset"] = "test"

    split_df.to_csv(seed_dir / "data_split_info.csv", index=False)

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

    inner_split_rows = []

    for fold_id, (inner_train_idx, inner_val_idx) in enumerate(inner_cv_splits, start=1):
        for local_idx in inner_train_idx:
            original_idx = idx_trainval[local_idx]
            inner_split_rows.append({
                "Trait": trait,
                "seed": seed,
                "split_seed": split_seed,
                "inner_fold": fold_id,
                "Genotype": genotypes[original_idx],
                "inner_subset": "inner_train",
            })

        for local_idx in inner_val_idx:
            original_idx = idx_trainval[local_idx]
            inner_split_rows.append({
                "Trait": trait,
                "seed": seed,
                "split_seed": split_seed,
                "inner_fold": fold_id,
                "Genotype": genotypes[original_idx],
                "inner_subset": "inner_validation",
            })

    pd.DataFrame(inner_split_rows).to_csv(
        seed_dir / "inner3cv_split_info.csv",
        index=False
    )

    tuning_rows = []
    best = None

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
                f"[{trait}] [SEED {seed}] Tuning "
                f"ridge_alpha={ridge_alpha}, lambda_size={lambda_size}, "
                f"CXPB={cxpb}, MUTPB={mutpb}"
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

            if SAVE_GA_LOGBOOKS:
                log_name = (
                    f"logbook_seed{seed}"
                    f"_ridge{ridge_alpha}"
                    f"_lambda{lambda_size}"
                    f"_cxpb{cxpb}"
                    f"_mutpb{mutpb}.csv"
                )
                logbook_df.to_csv(seed_dir / log_name, index=False)

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
                "Trait": trait,
                "seed": seed,
                "split_seed": split_seed,
                "cxpb": cxpb,
                "mutpb": mutpb,
                "ridge_alpha": ridge_alpha,
                "lambda_size": lambda_size,
                "ga_best_fitness": best_fitness,
                "n_selected": n_selected,

                "innerCV_RMSE_mean": inner_metrics["innerCV_RMSE_mean"],
                "innerCV_RMSE_std": inner_metrics["innerCV_RMSE_std"],
                "innerCV_MAE_mean": inner_metrics["innerCV_MAE_mean"],
                "innerCV_MAE_std": inner_metrics["innerCV_MAE_std"],
                "innerCV_R2_mean": inner_metrics["innerCV_R2_mean"],
                "innerCV_R2_std": inner_metrics["innerCV_R2_std"],
                "innerCV_Pearson_r_mean": inner_metrics["innerCV_Pearson_r_mean"],
                "innerCV_Pearson_r_std": inner_metrics["innerCV_Pearson_r_std"],
            }

            tuning_rows.append(row)

            if best is None or best_fitness < best["ga_best_fitness"]:
                best = row.copy()
                best["selected_mask"] = selected_mask.copy()

            print(
                f"    selected={n_selected}, "
                f"innerCV_RMSE={inner_metrics['innerCV_RMSE_mean']:.3f} "
                f"± {inner_metrics['innerCV_RMSE_std']:.3f}, "
                f"innerCV_R2={inner_metrics['innerCV_R2_mean']:.3f}, "
                f"fitness={best_fitness:.3f}"
            )

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(seed_dir / "tuning_results_inner3cv.csv", index=False)

    best_ridge_alpha = best["ridge_alpha"]
    best_lambda_size = best["lambda_size"]
    selected_mask = best["selected_mask"]

    print(f"\n[{trait}] [SEED {seed}] Best tuning")
    print(f"  ridge_alpha = {best_ridge_alpha}")
    print(f"  lambda_size = {best_lambda_size}")
    print(f"  innerCV_RMSE_mean = {best['innerCV_RMSE_mean']:.3f}")
    print(f"  innerCV_RMSE_std = {best['innerCV_RMSE_std']:.3f}")
    print(f"  innerCV_R2_mean = {best['innerCV_R2_mean']:.3f}")
    print(f"  n_selected = {best['n_selected']}")

    test_metrics, test_pred, final_model = evaluate_subset_final(
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

    selected_snps = [s for s, keep in zip(snp_cols, selected_mask) if keep]

    selected_df = pd.DataFrame({
        "Trait": trait,
        "seed": seed,
        "split_seed": split_seed,
        "snp": selected_snps,
        "selected": 1,
    })

    selected_df.to_csv(seed_dir / "best_selected_snps.csv", index=False)

    if SAVE_PREDICTIONS:
        test_pred_df = pd.DataFrame({
            "Trait": trait,
            "seed": seed,
            "split_seed": split_seed,
            "Genotype": genotypes[idx_test],
            "subset": "test",
            "y_true": y_test,
            "y_pred": test_pred,
            "residual": y_test - test_pred,
        })

        test_pred_df.to_csv(seed_dir / "best_model_test_predictions.csv", index=False)

        trainval_pred_df = pd.DataFrame({
            "Trait": trait,
            "seed": seed,
            "split_seed": split_seed,
            "Genotype": genotypes[idx_trainval],
            "subset": "trainval",
            "y_true": y_trainval,
            "y_pred": trainval_pred,
            "residual": y_trainval - trainval_pred,
        })

        trainval_pred_df.to_csv(seed_dir / "best_model_trainval_predictions.csv", index=False)

    result = {
        "Trait": trait,
        "seed": seed,
        "split_seed": split_seed,
        "cxpb": cxpb,
        "mutpb": mutpb,

        "best_ridge_alpha": best_ridge_alpha,
        "best_lambda_size": best_lambda_size,

        "innerCV_RMSE_mean": best["innerCV_RMSE_mean"],
        "innerCV_RMSE_std": best["innerCV_RMSE_std"],
        "innerCV_MAE_mean": best["innerCV_MAE_mean"],
        "innerCV_MAE_std": best["innerCV_MAE_std"],
        "innerCV_R2_mean": best["innerCV_R2_mean"],
        "innerCV_R2_std": best["innerCV_R2_std"],
        "innerCV_Pearson_r_mean": best["innerCV_Pearson_r_mean"],
        "innerCV_Pearson_r_std": best["innerCV_Pearson_r_std"],

        "trainval_RMSE": trainval_metrics["RMSE"],
        "trainval_MAE": trainval_metrics["MAE"],
        "trainval_R2": trainval_metrics["R2"],
        "trainval_Pearson_r": trainval_metrics["Pearson_r"],

        "test_RMSE": test_metrics["RMSE"],
        "test_MAE": test_metrics["MAE"],
        "test_R2": test_metrics["R2"],
        "test_Pearson_r": test_metrics["Pearson_r"],

        "n_selected_snps": len(selected_snps),

        "n_total": int(len(idx_all)),
        "n_trainval": int(len(idx_trainval)),
        "n_test": int(len(idx_test)),
        "inner_cv_folds": int(INNER_CV_FOLDS),
    }

    with open(seed_dir / "best_model_metrics.json", "w") as f:
        json.dump(result, f, indent=4)

    pd.DataFrame([result]).to_csv(seed_dir / "best_model_metrics.csv", index=False)

    print(f"\n[{trait}] [SEED {seed}] External test")
    print(f"  selected SNPs = {len(selected_snps)}")
    print(f"  innerCV RMSE = {best['innerCV_RMSE_mean']:.3f} ± {best['innerCV_RMSE_std']:.3f}")
    print(f"  test RMSE = {test_metrics['RMSE']:.3f}")
    print(f"  test MAE  = {test_metrics['MAE']:.3f}")
    print(f"  test R2   = {test_metrics['R2']:.3f}")
    print(f"  test r    = {test_metrics['Pearson_r']:.3f}")

    return result, selected_df


# =============================================================================
# FINAL REPORT HELPERS
# =============================================================================

def summarize_metric(metrics_df, cols):
    rows = []

    for col in cols:
        rows.append({
            "metric": col,
            "mean": metrics_df[col].mean(),
            "std": metrics_df[col].std(),
            "min": metrics_df[col].min(),
            "median": metrics_df[col].median(),
            "max": metrics_df[col].max(),
        })

    return pd.DataFrame(rows)


def write_final_report(
    trait,
    out_dir,
    metrics_df,
    summary_df,
    hyperparam_frequency,
    ridge_frequency,
    lambda_frequency,
    n_selected_summary,
    snp_stability,
    region_stability,
    gene_stability,
    cxpb,
    mutpb,
    g2d_source,
):
    report_path = out_dir / f"G2B_final_report_{trait}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"G2B FINAL REPORT - MULTI-SEED GA WITH INNER 3-FOLD CV FITNESS - {trait}\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"TRAIT: {trait}\n")
        f.write(f"DATASET_LABEL: {DATASET_LABEL}\n")
        f.write(f"WINDOW_LABEL: {WINDOW_LABEL}\n")
        f.write(f"TOP_K_REGIONS: {TOP_K_REGIONS}\n")
        f.write(f"SEEDS: {SEEDS}\n")
        f.write(f"TEST_SIZE: {TEST_SIZE}\n")
        f.write(f"INNER_CV_FOLDS: {INNER_CV_FOLDS}\n")
        f.write(f"RIDGE_ALPHA_GRID: {RIDGE_ALPHA_GRID}\n")
        f.write(f"LAMBDA_SIZE_GRID: {LAMBDA_SIZE_GRID}\n")
        f.write(f"POP_SIZE: {POP_SIZE}\n")
        f.write(f"N_GEN: {N_GEN}\n")
        f.write(f"CXPB: {cxpb}\n")
        f.write(f"MUTPB: {mutpb}\n")
        f.write(f"G2D_SOURCE: {g2d_source}\n")
        f.write(f"TOURNSIZE: {TOURNSIZE}\n")
        f.write(f"INIT_PROB_ONE: {INIT_PROB_ONE}\n")
        f.write(f"BIT_FLIP_INDPB: {BIT_FLIP_INDPB}\n")
        f.write(f"USE_PCA: {USE_PCA}\n\n")

        f.write("MAIN TEST PERFORMANCE\n")
        f.write("-" * 80 + "\n")

        for metric in ["test_RMSE", "test_MAE", "test_R2", "test_Pearson_r"]:
            row = summary_df[summary_df["metric"] == metric].iloc[0]
            f.write(
                f"{metric}: "
                f"{row['mean']:.4f} ± {row['std']:.4f} "
                f"(min={row['min']:.4f}, median={row['median']:.4f}, max={row['max']:.4f})\n"
            )

        f.write("\nINNER CV PERFORMANCE USED BY GA FITNESS\n")
        f.write("-" * 80 + "\n")

        for metric in [
            "innerCV_RMSE_mean",
            "innerCV_MAE_mean",
            "innerCV_R2_mean",
            "innerCV_Pearson_r_mean"
        ]:
            row = summary_df[summary_df["metric"] == metric].iloc[0]
            f.write(
                f"{metric}: "
                f"{row['mean']:.4f} ± {row['std']:.4f} "
                f"(min={row['min']:.4f}, median={row['median']:.4f}, max={row['max']:.4f})\n"
            )

        f.write("\nSELECTED SNP SUBSET SIZE\n")
        f.write("-" * 80 + "\n")
        for _, row in n_selected_summary.iterrows():
            f.write(
                f"n_selected_snps: "
                f"{row['mean']:.2f} ± {row['std']:.2f} "
                f"(min={row['min']:.0f}, median={row['median']:.0f}, max={row['max']:.0f})\n"
            )

        f.write("\nBEST HYPERPARAMETER FREQUENCY\n")
        f.write("-" * 80 + "\n")
        f.write(hyperparam_frequency.to_string(index=False))
        f.write("\n\n")

        f.write("RIDGE ALPHA FREQUENCY\n")
        f.write("-" * 80 + "\n")
        f.write(ridge_frequency.to_string(index=False))
        f.write("\n\n")

        f.write("LAMBDA SIZE FREQUENCY\n")
        f.write("-" * 80 + "\n")
        f.write(lambda_frequency.to_string(index=False))
        f.write("\n\n")

        f.write("TOP 20 STABLE SNPs\n")
        f.write("-" * 80 + "\n")
        if snp_stability is not None and len(snp_stability) > 0:
            f.write(snp_stability.head(20).to_string(index=False))
        else:
            f.write("No SNP stability results available.")
        f.write("\n\n")

        f.write("TOP 20 STABLE REGIONS\n")
        f.write("-" * 80 + "\n")
        if region_stability is not None and len(region_stability) > 0:
            f.write(region_stability.head(20).to_string(index=False))
        else:
            f.write("No region stability results available.")
        f.write("\n\n")

        f.write("TOP 20 STABLE GENES\n")
        f.write("-" * 80 + "\n")
        if gene_stability is not None and len(gene_stability) > 0:
            f.write(gene_stability.head(20).to_string(index=False))
        else:
            f.write("No gene stability results available.")
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"[INFO] Final report saved to:\n  {report_path}")


# =============================================================================
# STABILITY AGGREGATION
# =============================================================================

def aggregate_stability(
    trait,
    out_dir,
    all_results,
    all_selected_dfs,
    snp_to_region,
    snp_to_genes,
    cxpb,
    mutpb,
    g2d_source,
):
    print("\n" + "=" * 80)
    print(f"[{trait}] [AGGREGATION] Stability across seeds")
    print("=" * 80)

    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(out_dir / f"G2B_multiseed_metrics_per_seed_{trait}.csv", index=False)

    metric_cols = [
        "innerCV_RMSE_mean",
        "innerCV_RMSE_std",
        "innerCV_MAE_mean",
        "innerCV_MAE_std",
        "innerCV_R2_mean",
        "innerCV_R2_std",
        "innerCV_Pearson_r_mean",
        "innerCV_Pearson_r_std",

        "trainval_RMSE",
        "trainval_MAE",
        "trainval_R2",
        "trainval_Pearson_r",

        "test_RMSE",
        "test_MAE",
        "test_R2",
        "test_Pearson_r",

        "n_selected_snps",
    ]

    summary_df = summarize_metric(metrics_df, metric_cols)
    summary_df.to_csv(out_dir / f"G2B_multiseed_metrics_summary_{trait}.csv", index=False)

    n_selected_summary = summarize_metric(metrics_df, ["n_selected_snps"])
    n_selected_summary.to_csv(out_dir / f"G2B_n_selected_summary_{trait}.csv", index=False)

    hyperparam_frequency = (
        metrics_df
        .groupby(["best_ridge_alpha", "best_lambda_size"])
        .size()
        .reset_index(name="n_seeds")
        .sort_values("n_seeds", ascending=False)
    )
    hyperparam_frequency["frequency"] = hyperparam_frequency["n_seeds"] / len(metrics_df)
    hyperparam_frequency.to_csv(
        out_dir / f"G2B_best_hyperparams_frequency_{trait}.csv",
        index=False
    )

    ridge_frequency = (
        metrics_df
        .groupby("best_ridge_alpha")
        .size()
        .reset_index(name="n_seeds")
        .sort_values("n_seeds", ascending=False)
    )
    ridge_frequency["frequency"] = ridge_frequency["n_seeds"] / len(metrics_df)
    ridge_frequency.to_csv(
        out_dir / f"G2B_ridge_alpha_frequency_{trait}.csv",
        index=False
    )

    lambda_frequency = (
        metrics_df
        .groupby("best_lambda_size")
        .size()
        .reset_index(name="n_seeds")
        .sort_values("n_seeds", ascending=False)
    )
    lambda_frequency["frequency"] = lambda_frequency["n_seeds"] / len(metrics_df)
    lambda_frequency.to_csv(
        out_dir / f"G2B_lambda_size_frequency_{trait}.csv",
        index=False
    )

    selected_all = pd.concat(all_selected_dfs, ignore_index=True)
    selected_all.to_csv(out_dir / f"G2B_selected_snps_all_seeds_long_{trait}.csv", index=False)

    n_seeds = len(sorted(selected_all["seed"].unique()))

    # -------------------------------------------------------------------------
    # SNP stability
    # -------------------------------------------------------------------------
    snp_stability = (
        selected_all
        .groupby("snp")
        .agg(
            n_seeds_selected=("seed", "nunique"),
            seeds=("seed", lambda x: ",".join(map(str, sorted(set(x)))))
        )
        .reset_index()
    )

    snp_stability["selection_frequency"] = (
        snp_stability["n_seeds_selected"] / n_seeds
    )

    snp_stability = snp_stability.sort_values(
        ["n_seeds_selected", "snp"],
        ascending=[False, True]
    )

    snp_stability.to_csv(
        out_dir / f"G2B_snp_stability_across_seeds_{trait}.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Region stability
    # -------------------------------------------------------------------------
    region_records = []

    for _, row in selected_all.iterrows():
        seed = row["seed"]
        split_seed = row["split_seed"]
        snp = row["snp"]

        regions = snp_to_region.get(snp, set())

        if len(regions) == 0:
            region_records.append({
                "Trait": trait,
                "seed": seed,
                "split_seed": split_seed,
                "snp": snp,
                "region": "UNMAPPED_REGION"
            })
        else:
            for region in regions:
                region_records.append({
                    "Trait": trait,
                    "seed": seed,
                    "split_seed": split_seed,
                    "snp": snp,
                    "region": region
                })

    region_long_df = pd.DataFrame(region_records)
    region_long_df.to_csv(
        out_dir / f"G2B_selected_regions_all_seeds_long_{trait}.csv",
        index=False
    )

    region_stability = (
        region_long_df
        .groupby("region")
        .agg(
            n_seeds_selected=("seed", "nunique"),
            n_selected_snp_events=("snp", "count"),
            n_unique_snps_selected=("snp", "nunique"),
            seeds=("seed", lambda x: ",".join(map(str, sorted(set(x)))))
        )
        .reset_index()
    )

    region_stability["selection_frequency"] = (
        region_stability["n_seeds_selected"] / n_seeds
    )

    region_stability = region_stability.sort_values(
        ["n_seeds_selected", "n_unique_snps_selected", "region"],
        ascending=[False, False, True]
    )

    region_stability.to_csv(
        out_dir / f"G2B_region_stability_across_seeds_{trait}.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Gene stability
    # -------------------------------------------------------------------------
    gene_records = []

    for _, row in selected_all.iterrows():
        seed = row["seed"]
        split_seed = row["split_seed"]
        snp = row["snp"]

        genes = snp_to_genes.get(snp, set())

        for gene in genes:
            gene_records.append({
                "Trait": trait,
                "seed": seed,
                "split_seed": split_seed,
                "snp": snp,
                "gene": gene
            })

    gene_stability = None

    if len(gene_records) > 0:
        gene_long_df = pd.DataFrame(gene_records)
        gene_long_df.to_csv(
            out_dir / f"G2B_selected_genes_all_seeds_long_{trait}.csv",
            index=False
        )

        gene_stability = (
            gene_long_df
            .groupby("gene")
            .agg(
                n_seeds_selected=("seed", "nunique"),
                n_selected_snp_events=("snp", "count"),
                n_unique_snps_selected=("snp", "nunique"),
                seeds=("seed", lambda x: ",".join(map(str, sorted(set(x)))))
            )
            .reset_index()
        )

        gene_stability["selection_frequency"] = (
            gene_stability["n_seeds_selected"] / n_seeds
        )

        gene_stability = gene_stability.sort_values(
            ["n_seeds_selected", "n_unique_snps_selected", "gene"],
            ascending=[False, False, True]
        )

        gene_stability.to_csv(
            out_dir / f"G2B_gene_stability_across_seeds_{trait}.csv",
            index=False
        )
    else:
        print(f"[WARNING] {trait}: No gene records found. Gene stability skipped.")

    write_final_report(
        trait=trait,
        out_dir=out_dir,
        metrics_df=metrics_df,
        summary_df=summary_df,
        hyperparam_frequency=hyperparam_frequency,
        ridge_frequency=ridge_frequency,
        lambda_frequency=lambda_frequency,
        n_selected_summary=n_selected_summary,
        snp_stability=snp_stability,
        region_stability=region_stability,
        gene_stability=gene_stability,
        cxpb=cxpb,
        mutpb=mutpb,
        g2d_source=g2d_source,
    )

    print(f"\n[{trait}] [SUMMARY] Test metrics across seeds")
    print(summary_df[summary_df["metric"].str.startswith("test_")])

    print(f"\n[{trait}] [SUMMARY] Inner CV metrics across seeds")
    print(summary_df[summary_df["metric"].str.startswith("innerCV_")])

    print(f"\n[{trait}] [SUMMARY] Selected SNP count")
    print(n_selected_summary)

    print(f"\n[{trait}] [SUMMARY] Best hyperparameter frequency")
    print(hyperparam_frequency)

    print(f"\n[{trait}] [SUMMARY] Top 20 stable SNPs")
    print(snp_stability.head(20))

    print(f"\n[{trait}] [SUMMARY] Top 20 stable regions")
    print(region_stability.head(20))

    if gene_stability is not None:
        print(f"\n[{trait}] [SUMMARY] Top 20 stable genes")
        print(gene_stability.head(20))

    return metrics_df, summary_df, snp_stability, region_stability, gene_stability


# =============================================================================
# RUN ONE TRAIT
# =============================================================================

def run_one_trait(trait: str):
    print("\n" + "#" * 80)
    print(f"G2B MULTI-SEED GA - TRAIT: {trait}")
    print("#" * 80)

    out_dir = BASE_OUT_DIR / trait
    out_dir.mkdir(parents=True, exist_ok=True)

    cxpb, mutpb, g2d_source = load_ga_hyperparams_from_g2d(trait)

    config = {
        "TRAIT": trait,
        "WINDOW_LABEL": WINDOW_LABEL,
        "TOP_K_REGIONS": TOP_K_REGIONS,
        "DATASET_LABEL": DATASET_LABEL,

        "GA_INPUT_DIR": str(GA_INPUT_DIR),
        "Q9_SUMMARY_FILE": str(Q9_SUMMARY_FILE),
        "OUT_DIR": str(out_dir),

        "USE_G2D_BEST_CONFIG": USE_G2D_BEST_CONFIG,
        "G2D_SOURCE": g2d_source,

        "FIX_DATA_SPLITS": FIX_DATA_SPLITS,
        "SPLIT_SEED": SPLIT_SEED,
        "SEEDS": SEEDS,

        "TEST_SIZE": TEST_SIZE,
        "INNER_CV_FOLDS": INNER_CV_FOLDS,

        "RIDGE_ALPHA_GRID": RIDGE_ALPHA_GRID,
        "LAMBDA_SIZE_GRID": LAMBDA_SIZE_GRID,

        "POP_SIZE": POP_SIZE,
        "N_GEN": N_GEN,
        "CXPB": cxpb,
        "MUTPB": mutpb,
        "TOURNSIZE": TOURNSIZE,

        "INIT_PROB_ONE": INIT_PROB_ONE,
        "BIT_FLIP_INDPB": BIT_FLIP_INDPB,
        "MIN_SELECTED_SNPS": MIN_SELECTED_SNPS,
        "MAX_SELECTED_SNPS": MAX_SELECTED_SNPS,

        "USE_PCA": USE_PCA,
        "SAVE_PREDICTIONS": SAVE_PREDICTIONS,
        "SAVE_GA_LOGBOOKS": SAVE_GA_LOGBOOKS,
    }

    with open(out_dir / f"G2B_config_{trait}.json", "w") as f:
        json.dump(config, f, indent=4)

    df, snp_cols, pca_cols, snp_meta = load_ga_inputs(trait, out_dir)
    snp_to_region, snp_to_genes = build_snp_metadata_maps(snp_meta, out_dir)

    progress_metrics_file = out_dir / f"G2B_multiseed_metrics_per_seed_PROGRESS_{trait}.csv"
    progress_selected_file = out_dir / f"G2B_selected_snps_all_seeds_long_PROGRESS_{trait}.csv"

    all_results = []
    all_selected_dfs = []
    completed_seeds = set()

    if progress_metrics_file.exists():
        old_metrics = pd.read_csv(progress_metrics_file)

        if len(old_metrics) > 0:
            all_results = old_metrics.to_dict("records")
            completed_seeds = set(old_metrics["seed"].astype(int).tolist())

            print(f"[RESUME] Existing progress metrics found:")
            print(progress_metrics_file)
            print(f"[RESUME] Completed seeds: {sorted(completed_seeds)}")

    if progress_selected_file.exists():
        old_selected = pd.read_csv(progress_selected_file)

        if len(old_selected) > 0:
            for _, sub in old_selected.groupby("seed"):
                all_selected_dfs.append(sub.copy())

            print(f"[RESUME] Existing selected SNP progress found:")
            print(progress_selected_file)

    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"[SKIP] {trait} seed {seed} already completed.")
            continue

        result, selected_df = run_one_seed(
            df=df,
            snp_cols=snp_cols,
            pca_cols=pca_cols,
            trait=trait,
            seed=seed,
            out_dir=out_dir,
            cxpb=cxpb,
            mutpb=mutpb,
        )

        all_results.append(result)
        all_selected_dfs.append(selected_df)

        pd.DataFrame(all_results).to_csv(
            progress_metrics_file,
            index=False
        )

        pd.concat(all_selected_dfs, ignore_index=True).to_csv(
            progress_selected_file,
            index=False
        )

    outputs = aggregate_stability(
        trait=trait,
        out_dir=out_dir,
        all_results=all_results,
        all_selected_dfs=all_selected_dfs,
        snp_to_region=snp_to_region,
        snp_to_genes=snp_to_genes,
        cxpb=cxpb,
        mutpb=mutpb,
        g2d_source=g2d_source,
    )

    print("\n" + "#" * 80)
    print(f"G2B FINISHED FOR TRAIT: {trait}")
    print("#" * 80)
    print(f"Outputs saved in:\n  {out_dir.resolve()}")

    return outputs


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 80)
    print("G2B MULTI-SEED GA - NEW TRAITS")
    print("#" * 80)

    all_trait_summaries = []

    for trait in TRAITS:
        metrics_df, summary_df, _, _, _ = run_one_trait(trait)

        tmp = summary_df.copy()
        tmp["Trait"] = trait
        all_trait_summaries.append(tmp)

    if len(all_trait_summaries) > 0:
        summary_all = pd.concat(all_trait_summaries, ignore_index=True)
        summary_all_file = BASE_OUT_DIR / "G2B_multiseed_metrics_summary_all_traits.csv"
        summary_all.to_csv(summary_all_file, index=False)

        print("\n" + "=" * 80)
        print("G2B ALL TRAITS COMPLETED")
        print("=" * 80)
        print("Saved all-trait summary:")
        print(summary_all_file)


if __name__ == "__main__":
    main()
