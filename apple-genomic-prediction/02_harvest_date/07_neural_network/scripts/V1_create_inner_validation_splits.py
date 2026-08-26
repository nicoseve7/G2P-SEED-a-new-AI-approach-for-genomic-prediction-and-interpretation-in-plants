# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# SETTINGS
# =============================================================================

OUT_DIR = (
    Path("02_harvest_date")
    / "07_neural_network"
    / "output"
    / "datasets"
)

META_FILE = (
    Path("02_harvest_date")
    / "06_deep_learning_baseline"
    / "output"
    / "numpy_arrays_harvest"
    / "sample_metadata_harvest.csv"
)

CV_FILE = (
    Path("data")
    / "raw"
    / "cv"
    / "Harvest_date_CV.csv"
)

TRAIT = "Harvest_date"
VAL_GENOTYPE_FRAC = 0.15
GLOBAL_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================

def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    make_dirs()
    set_all_seeds(GLOBAL_SEED)

    meta = pd.read_csv(META_FILE)
    meta = meta[["Envir", "Genotype", TRAIT]].copy()
    meta["Envir"] = meta["Envir"].astype(str)
    meta["Genotype"] = meta["Genotype"].astype(str)
    meta["ID_key"] = meta["Envir"] + "-" + meta["Genotype"]

    cv = pd.read_csv(CV_FILE)
    cv["Envir"] = cv["Envir"].astype(str)
    cv["Genotype"] = cv["Genotype"].astype(str)
    cv["ID_key"] = cv["Envir"] + "-" + cv["Genotype"]

    cv = cv.set_index("ID_key").loc[meta["ID_key"]].reset_index()

    split_cols = [c for c in cv.columns if c.startswith("CV")]

    out_df = meta[["Envir", "Genotype", "ID_key"]].copy()

    summary_rows = []

    for split_name in split_cols:
        testing = cv[split_name].astype(int).values

        # initialize role
        role = np.array(["UNASSIGNED"] * len(out_df), dtype=object)
        role[testing == 1] = "test"

        train_mask = testing == 0
        train_genotypes = sorted(out_df.loc[train_mask, "Genotype"].unique().tolist())

        # deterministic per split
        split_seed = abs(hash((split_name, GLOBAL_SEED))) % (2**32)

        geno_subtrain, geno_val = train_test_split(
            train_genotypes,
            test_size=VAL_GENOTYPE_FRAC,
            random_state=split_seed,
            shuffle=True
        )

        val_mask = train_mask & out_df["Genotype"].isin(geno_val).values
        subtrain_mask = train_mask & out_df["Genotype"].isin(geno_subtrain).values

        role[val_mask] = "val"
        role[subtrain_mask] = "subtrain"

        out_df[f"{split_name}_role"] = role
        out_df[f"{split_name}_Testing"] = testing
        out_df[f"{split_name}_Validation"] = (role == "val").astype(int)
        out_df[f"{split_name}_Subtrain"] = (role == "subtrain").astype(int)

        summary_rows.append({
            "Split": split_name,
            "n_rows_total": len(out_df),
            "n_rows_test": int((role == "test").sum()),
            "n_rows_val": int((role == "val").sum()),
            "n_rows_subtrain": int((role == "subtrain").sum()),
            "n_genotypes_train_total": len(train_genotypes),
            "n_genotypes_val": len(geno_val),
            "n_genotypes_subtrain": len(geno_subtrain),
        })

    out_file = OUT_DIR / "inner_validation_splits_harvest.csv"
    out_df.to_csv(out_file, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUT_DIR / "inner_validation_splits_harvest_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print("Saved:")
    print(out_file)
    print(summary_file)
    print(summary_df.head())


if __name__ == "__main__":
    main()
