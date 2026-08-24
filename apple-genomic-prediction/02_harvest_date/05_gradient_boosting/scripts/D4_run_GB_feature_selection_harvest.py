# -*- coding: utf-8 -*-

################################################################################
### Feature Selection using GB - adapted for Harvest_date only
################################################################################

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------
# 1. Paths
# -------------------------------
GB_working_dir = "Output/Intermediate/GB_feature_selection/"
trait = "Harvest_date"
trait_dir = os.path.join(GB_working_dir, trait)

# -------------------------------
# 2. Load common genotype matrix
# -------------------------------
print("Loading all.geno ...")
#geno = pd.read_csv(os.path.join(GB_working_dir, "all.geno"))
geno = pd.read_csv(
    os.path.join(GB_working_dir, "all.geno"),
    engine="python"
)

# first column should be Genotype
geno_id_col = geno.columns[0]

# -------------------------------
# 3. Iterate over split files
# -------------------------------
files_splits = sorted([f for f in os.listdir(trait_dir) if f.endswith(".csv")])

importance_df = pd.DataFrame(columns=["Trait", "Split", "SNP", "importance", "n_selected"])

for file in files_splits:
    split_name = file[:-4]
    print(f"\nProcessing split: {split_name}")

    # Load phenotypic training data for this split
    pheno_split = pd.read_csv(os.path.join(trait_dir, file))

    # Keep only genotypes present in this training split
    mask_geno = geno[geno_id_col].isin(pheno_split["Genotype"])
    geno_split = geno[mask_geno].reset_index(drop=True)

    # Set genotype as index to align phenotype and genotype
    pheno_split = pheno_split.set_index("Genotype")
    geno_split = geno_split.set_index(geno_id_col)

    # Reindex phenotype to match geno order
    pheno_split = pheno_split.reindex(geno_split.index)

    # Convert to numpy
    pheno_np = np.ravel(pheno_split.to_numpy())
    geno_np = geno_split.to_numpy()

    # Fit GB regressor
    print(f"Fitting GradientBoostingRegressor on {geno_np.shape[0]} genotypes and {geno_np.shape[1]} SNPs")
    reg = GradientBoostingRegressor(random_state=0, n_estimators=200)
    reg.fit(geno_np, pheno_np)

    # Extract importances
    feature_importance = reg.feature_importances_

    # Keep SNPs with importance > 0
    important_mask = feature_importance > 0
    n_important = int(np.sum(important_mask))

    sorted_idx = np.argsort(feature_importance)[::-1]
    important_idx = sorted_idx[:n_important]

    important_snps = geno_split.columns[important_idx].tolist()
    importance_vals = feature_importance[important_idx].tolist()

    print(f"Selected SNPs with importance > 0: {n_important}")

    # Save results for this split
    split_df = pd.DataFrame({
        "Trait": trait,
        "Split": split_name,
        "SNP": important_snps,
        "importance": importance_vals,
        "n_selected": n_important
    })

    importance_df = pd.concat([importance_df, split_df], ignore_index=True)

# -------------------------------
# 4. Save final results
# -------------------------------
out_file = os.path.join(GB_working_dir, "feature_selection_results_harvest_date.csv")
importance_df.to_csv(out_file, index=False)

print("\nSaved feature selection results to:")
print(out_file)

# small summary table
summary_df = (
    importance_df[["Split", "n_selected"]]
    .drop_duplicates()
    .sort_values("Split")
    .reset_index(drop=True)
)

summary_file = os.path.join(GB_working_dir, "feature_selection_summary_harvest_date.csv")
summary_df.to_csv(summary_file, index=False)

print("Saved summary to:")
print(summary_file)