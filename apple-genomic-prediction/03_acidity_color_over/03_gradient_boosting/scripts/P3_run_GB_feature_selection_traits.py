# -*- coding: utf-8 -*-

################################################################################
### P3_run_GB_feature_selection_traits.py
###
### Feature selection with GradientBoostingRegressor for multiple traits:
### - Acidity
### - Color_over
###
### Input expected:
###   Output/Intermediate/GB_feature_selection/all.geno
###   Output/Intermediate/GB_feature_selection/Acidity/CV*_Split*.csv
###   Output/Intermediate/GB_feature_selection/Color_over/CV*_Split*.csv
###
### Output:
###   Output/Intermediate/GB_feature_selection/feature_selection_results_acidity.csv
###   Output/Intermediate/GB_feature_selection/feature_selection_summary_acidity.csv
###   Output/Intermediate/GB_feature_selection/feature_selection_results_color_over.csv
###   Output/Intermediate/GB_feature_selection/feature_selection_summary_color_over.csv
################################################################################

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

GB_WORKING_DIR = Path("Output/Intermediate/GB_feature_selection")

N_ESTIMATORS = 200
RANDOM_STATE = 0

# Se True, salva anche un file per ogni split con tutti gli SNP ordinati per importance.
# Può essere grande, quindi di default lo lascio False.
SAVE_PER_SPLIT_FULL_IMPORTANCE = False


# =============================================================================
# HELPERS
# =============================================================================

def trait_to_file_label(trait: str) -> str:
    """
    Converte il nome trait in etichetta file.
    Harvest_date -> harvest_date
    Color_over   -> color_over
    Acidity      -> acidity
    """
    return trait.lower()


def load_genotype_matrix():
    geno_file = GB_WORKING_DIR / "all.geno"

    if not geno_file.exists():
        raise FileNotFoundError(
            f"File all.geno non trovato:\n{geno_file}\n\n"
            "Devi prima eseguire P2_prepare_GB_inputs_traits.R."
        )

    print("=" * 80)
    print("Loading all.geno ...")
    print("=" * 80)
    print(geno_file)

    geno = pd.read_csv(geno_file, engine="python")

    if geno.shape[1] < 2:
        raise ValueError("all.geno sembra avere meno di 2 colonne.")

    geno_id_col = geno.columns[0]
    geno[geno_id_col] = geno[geno_id_col].astype(str).str.replace("^G_", "", regex=True).str.strip()

    snp_cols = [c for c in geno.columns if c != geno_id_col]

    print(f"Genotype ID column: {geno_id_col}")
    print(f"Number of genotypes in all.geno: {geno.shape[0]}")
    print(f"Number of SNP columns in all.geno: {len(snp_cols)}")

    return geno, geno_id_col, snp_cols


def run_one_trait(trait: str, geno: pd.DataFrame, geno_id_col: str):
    print("\n" + "#" * 80)
    print(f"RUNNING GB FEATURE SELECTION FOR TRAIT: {trait}")
    print("#" * 80)

    trait_dir = GB_WORKING_DIR / trait

    if not trait_dir.exists():
        raise FileNotFoundError(
            f"Cartella del trait non trovata:\n{trait_dir}\n\n"
            "Devi prima eseguire P2_prepare_GB_inputs_traits.R."
        )

    files_splits = sorted([
        f for f in os.listdir(trait_dir)
        if f.endswith(".csv") and f.startswith("CV")
    ])

    if len(files_splits) == 0:
        raise ValueError(f"Nessun file CV*_Split*.csv trovato in {trait_dir}")

    print(f"Number of split files found: {len(files_splits)}")

    importance_rows = []
    summary_rows = []
    failed_rows = []

    per_split_dir = GB_WORKING_DIR / f"{trait}_per_split_importances"
    if SAVE_PER_SPLIT_FULL_IMPORTANCE:
        per_split_dir.mkdir(parents=True, exist_ok=True)

    for file in files_splits:
        split_name = file[:-4]
        split_file = trait_dir / file

        print("\n" + "-" * 80)
        print(f"Processing trait={trait}, split={split_name}")
        print("-" * 80)

        try:
            pheno_split = pd.read_csv(split_file)

            if "Genotype" not in pheno_split.columns:
                raise ValueError(f"Colonna Genotype mancante in {split_file}")

            if trait not in pheno_split.columns:
                # fallback: se P2 ha salvato la colonna come nome generico
                possible_target_cols = [c for c in pheno_split.columns if c != "Genotype"]
                if len(possible_target_cols) == 1:
                    old_col = possible_target_cols[0]
                    print(f"[INFO] Target column '{trait}' not found. Using '{old_col}' as target.")
                    pheno_split = pheno_split.rename(columns={old_col: trait})
                else:
                    raise ValueError(
                        f"Colonna target {trait} mancante in {split_file}. "
                        f"Colonne trovate: {pheno_split.columns.tolist()}"
                    )

            pheno_split["Genotype"] = pheno_split["Genotype"].astype(str).str.replace("^G_", "", regex=True).str.strip()
            pheno_split[trait] = pd.to_numeric(pheno_split[trait], errors="coerce")
            pheno_split = pheno_split.dropna(subset=[trait]).copy()

            if pheno_split.shape[0] < 5:
                raise ValueError(f"Troppi pochi genotipi nel training split: {pheno_split.shape[0]}")

            # Keep only genotypes present in this training split
            mask_geno = geno[geno_id_col].isin(pheno_split["Genotype"])
            geno_split = geno.loc[mask_geno].reset_index(drop=True)

            if geno_split.shape[0] < 5:
                raise ValueError(f"Troppi pochi genotipi genomici matchati: {geno_split.shape[0]}")

            # Set genotype as index to align phenotype and genotype
            pheno_split = pheno_split.set_index("Genotype")
            geno_split = geno_split.set_index(geno_id_col)

            # Reindex phenotype to match genotype order
            pheno_split = pheno_split.reindex(geno_split.index)

            if pheno_split[trait].isna().any():
                n_missing = int(pheno_split[trait].isna().sum())
                raise ValueError(f"Dopo reindex ci sono {n_missing} target missing.")

            y = np.ravel(pheno_split[[trait]].to_numpy(dtype=float))
            X = geno_split.to_numpy(dtype=float)

            print(f"Fitting GradientBoostingRegressor on {X.shape[0]} genotypes and {X.shape[1]} SNPs")

            reg = GradientBoostingRegressor(
                random_state=RANDOM_STATE,
                n_estimators=N_ESTIMATORS
            )

            reg.fit(X, y)

            feature_importance = reg.feature_importances_

            important_mask = feature_importance > 0
            n_important = int(np.sum(important_mask))

            sorted_idx = np.argsort(feature_importance)[::-1]
            important_idx = sorted_idx[:n_important]

            important_snps = geno_split.columns[important_idx].tolist()
            importance_vals = feature_importance[important_idx].tolist()

            print(f"Selected SNPs with importance > 0: {n_important}")

            for snp, imp in zip(important_snps, importance_vals):
                importance_rows.append({
                    "Trait": trait,
                    "Split": split_name,
                    "SNP": snp,
                    "importance": imp,
                    "n_selected": n_important,
                    "n_training_genotypes": X.shape[0],
                    "n_total_snps": X.shape[1],
                })

            summary_rows.append({
                "Trait": trait,
                "Split": split_name,
                "n_selected": n_important,
                "n_training_genotypes": X.shape[0],
                "n_total_snps": X.shape[1],
                "status": "ok",
            })

            if SAVE_PER_SPLIT_FULL_IMPORTANCE:
                full_imp = pd.DataFrame({
                    "Trait": trait,
                    "Split": split_name,
                    "SNP": geno_split.columns.tolist(),
                    "importance": feature_importance,
                }).sort_values("importance", ascending=False)

                full_imp.to_csv(
                    per_split_dir / f"{split_name}_all_importances.csv",
                    index=False
                )

        except Exception as e:
            print(f"[FAILED] trait={trait}, split={split_name}")
            print(e)

            failed_rows.append({
                "Trait": trait,
                "Split": split_name,
                "file": str(split_file),
                "error": str(e),
            })

            summary_rows.append({
                "Trait": trait,
                "Split": split_name,
                "n_selected": np.nan,
                "n_training_genotypes": np.nan,
                "n_total_snps": np.nan,
                "status": "failed",
            })

    label = trait_to_file_label(trait)

    importance_df = pd.DataFrame(importance_rows)
    summary_df = pd.DataFrame(summary_rows)
    failed_df = pd.DataFrame(failed_rows)

    out_file = GB_WORKING_DIR / f"feature_selection_results_{label}.csv"
    summary_file = GB_WORKING_DIR / f"feature_selection_summary_{label}.csv"
    failed_file = GB_WORKING_DIR / f"feature_selection_failed_{label}.csv"
    report_file = GB_WORKING_DIR / f"feature_selection_report_{label}.txt"

    importance_df.to_csv(out_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    if len(failed_df) > 0:
        failed_df.to_csv(failed_file, index=False)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"=== GB FEATURE SELECTION REPORT: {trait} ===\n\n")
        f.write(f"Trait: {trait}\n")
        f.write(f"N_ESTIMATORS: {N_ESTIMATORS}\n")
        f.write(f"RANDOM_STATE: {RANDOM_STATE}\n\n")

        f.write(f"Split files found: {len(files_splits)}\n")
        f.write(f"Completed splits: {int((summary_df['status'] == 'ok').sum())}\n")
        f.write(f"Failed splits: {int((summary_df['status'] == 'failed').sum())}\n\n")

        f.write("Summary by split:\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        if len(importance_df) > 0:
            f.write("Selected SNP count summary:\n")
            f.write(importance_df[["Split", "n_selected"]].drop_duplicates()["n_selected"].describe().to_string())
            f.write("\n\n")

            f.write("Top 20 rows by importance:\n")
            f.write(importance_df.sort_values("importance", ascending=False).head(20).to_string(index=False))
            f.write("\n\n")

        if len(failed_df) > 0:
            f.write("Failed splits:\n")
            f.write(failed_df.to_string(index=False))
            f.write("\n")

    print("\nSaved:")
    print(out_file)
    print(summary_file)
    print(report_file)

    if len(failed_df) > 0:
        print(failed_file)

    return {
        "trait": trait,
        "n_split_files": len(files_splits),
        "n_completed": int((summary_df["status"] == "ok").sum()),
        "n_failed": int((summary_df["status"] == "failed").sum()),
        "n_rows_importance": int(len(importance_df)),
    }


def main():
    print("=" * 80)
    print("P3 - GB FEATURE SELECTION FOR NEW TRAITS")
    print("=" * 80)

    geno, geno_id_col, snp_cols = load_genotype_matrix()

    global_results = []

    for trait in TRAITS:
        res = run_one_trait(trait, geno, geno_id_col)
        global_results.append(res)

    global_df = pd.DataFrame(global_results)
    global_report_file = GB_WORKING_DIR / "P3_GB_feature_selection_global_report.csv"
    global_df.to_csv(global_report_file, index=False)

    config = {
        "TRAITS": TRAITS,
        "GB_WORKING_DIR": str(GB_WORKING_DIR),
        "N_ESTIMATORS": N_ESTIMATORS,
        "RANDOM_STATE": RANDOM_STATE,
        "SAVE_PER_SPLIT_FULL_IMPORTANCE": SAVE_PER_SPLIT_FULL_IMPORTANCE,
    }

    with open(GB_WORKING_DIR / "P3_GB_feature_selection_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("\n" + "=" * 80)
    print("P3 completed.")
    print("=" * 80)
    print(global_df.to_string(index=False))
    print("\nSaved global report:")
    print(global_report_file)


if __name__ == "__main__":
    main()