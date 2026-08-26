# Questo script:

# legge SUMMARY_all_snp_SHAP
# legge coordinate SNP dal file info SNP
# crea una tabella base unica
# aggiunge categoria mapped/unmapped

# Assumo che il file coordinate SNP sia quello già usato nel progetto, tipo:

# ../Output/SNP_info_modeling_var_gt0.csv

# -*- coding: utf-8 -*-

################################################################################
### R0_prepare_region_inputs.py
### Prepare unified SNP table for region construction
################################################################################

from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# SETTINGS
# =============================================================================

TRAIT = "Harvest_date"
MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3"

PROJECT_DIR = Path(".")
INPUT_DIR = PROJECT_DIR / "Input"
OUTPUT_DIR = PROJECT_DIR / "Output"

SHAP_DIR = INPUT_DIR / "shap_inputs"
BASE_DIR = INPUT_DIR / "base_files"

REGION_OUT_DIR = OUTPUT_DIR / "00_regioni"
REGION_OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_SNP_SHAP_FILE = SHAP_DIR / f"SUMMARY_all_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
MAPPED_SNP_SHAP_FILE = SHAP_DIR / f"SUMMARY_mapped_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
UNMAPPED_SNP_SHAP_FILE = SHAP_DIR / f"SUMMARY_unmapped_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"

# CAMBIA SOLO SE IL TUO FILE HA NOME DIVERSO
SNP_INFO_FILE = BASE_DIR / "SNP_info_modeling_var_gt0.csv"

SAVE_FILE = REGION_OUT_DIR / f"snp_table_for_regions_{MODEL_NAME}_{TRAIT}.csv"
REPORT_FILE = REGION_OUT_DIR / "R0_prepare_region_inputs_report.txt"


# =============================================================================
# HELPERS
# =============================================================================

def standardize_snp_info_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for c in df.columns:
        cl = c.lower().strip()

        if cl in {"snp", "snp.name", "snp_name", "marker", "marker_name"}:
            rename_map[c] = "SNP"
        elif cl in {"chrom", "chr", "chromosome"}:
            rename_map[c] = "CHROM"
        elif cl in {"pos", "bp", "position"}:
            rename_map[c] = "POS"

    df = df.rename(columns=rename_map)

    required = {"SNP", "CHROM", "POS"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Il file SNP info deve contenere almeno {required}. "
            f"Colonne trovate: {df.columns.tolist()}"
        )

    df["SNP"] = df["SNP"].astype(str)
    df["CHROM"] = df["CHROM"].astype(str)
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")

    df = df.dropna(subset=["POS"]).copy()
    df["POS"] = df["POS"].astype(int)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not ALL_SNP_SHAP_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {ALL_SNP_SHAP_FILE}")
    if not MAPPED_SNP_SHAP_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {MAPPED_SNP_SHAP_FILE}")
    if not UNMAPPED_SNP_SHAP_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {UNMAPPED_SNP_SHAP_FILE}")
    if not SNP_INFO_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {SNP_INFO_FILE}")

    print("Reading SHAP summaries...")
    all_df = pd.read_csv(ALL_SNP_SHAP_FILE)
    mapped_df = pd.read_csv(MAPPED_SNP_SHAP_FILE)
    unmapped_df = pd.read_csv(UNMAPPED_SNP_SHAP_FILE)

    print("Reading SNP coordinates...")
    snp_info = pd.read_csv(SNP_INFO_FILE)
    snp_info = standardize_snp_info_columns(snp_info)

    mapped_set = set(mapped_df["SNP"].astype(str))
    unmapped_set = set(unmapped_df["SNP"].astype(str))

    all_df["SNP"] = all_df["SNP"].astype(str)

    all_df["mapping_category"] = np.where(
        all_df["SNP"].isin(mapped_set), "mapped",
        np.where(all_df["SNP"].isin(unmapped_set), "unmapped", "unknown")
    )

    merged = all_df.merge(
        snp_info[["SNP", "CHROM", "POS"]],
        on="SNP",
        how="left"
    )

    merged["has_coordinates"] = ~merged["POS"].isna()

    n_before = len(merged)
    merged = merged.dropna(subset=["CHROM", "POS"]).copy()
    merged["POS"] = merged["POS"].astype(int)

    merged = merged.sort_values(["CHROM", "POS", "meanSHAP"], ascending=[True, True, False]).reset_index(drop=True)

    merged.to_csv(SAVE_FILE, index=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== R0 PREPARE REGION INPUTS REPORT ===\n\n")
        f.write(f"Input all SNP SHAP rows: {len(all_df)}\n")
        f.write(f"Mapped SNP unique: {len(mapped_set)}\n")
        f.write(f"Unmapped SNP unique: {len(unmapped_set)}\n")
        f.write(f"SNP info rows: {len(snp_info)}\n")
        f.write(f"Rows before coordinate filtering: {n_before}\n")
        f.write(f"Rows after coordinate filtering: {len(merged)}\n\n")

        f.write("Mapping category counts:\n")
        f.write(merged["mapping_category"].value_counts(dropna=False).to_string())
        f.write("\n\n")

        f.write("Chromosome counts:\n")
        f.write(merged["CHROM"].value_counts(dropna=False).to_string())
        f.write("\n")

    print("Saved:")
    print(SAVE_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()