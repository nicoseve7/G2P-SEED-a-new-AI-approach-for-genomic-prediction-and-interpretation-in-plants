# -*- coding: utf-8 -*-

################################################################################
### G0_build_ga_target_and_inputs_no_soil.py
###
### Build GA-ready SNP matrix, PCA matrix, target and metadata
### using top1000 50kb regions ranked from NO-SOIL SHAP.
###
### Run from:
###   dalpaper/regioni_ga_harvest_no_soil/
################################################################################

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

TRAIT = "Harvest_date"
TOP_K = 1000
WINDOW_LABEL = "50kb"

GA_ROOT = (
    Path("02_harvest_date")
    / "09_genetic_algorithm"
)

OUTPUT_DIR = GA_ROOT / "output"

RANK_DIR = (
    OUTPUT_DIR
    / "02_regioni_ranked"
)

GA_DIR = (
    OUTPUT_DIR
    / "03_ga_inputs"
)

GA_DIR.mkdir(parents=True, exist_ok=True)

ALL_GENO_FILE = (
    Path("data")
    / "raw"
    / "genotype"
    / "all.geno"
)

PCA_FILE = (
    Path("01_common_genomic_preprocessing")
    / "output"
    / "genomic_PCs_20_paper_style.csv"
)

ADJ_FILE = (
    Path("02_harvest_date")
    / "02_phenotype_preprocessing"
    / "output"
    / "harvestdate_adjusted_values_genotype.csv"
)

# No-soil top1000 regions created by R3 no-soil
TOP_REGIONS_FILE = RANK_DIR / f"top{TOP_K}_regions_by_region_score_{WINDOW_LABEL}_{TRAIT}.csv"

# Use original static SNP-region membership.
# This file does not depend on full/no-soil SHAP.
REGION_MEMBERSHIP_FILE = Path(
    "../regioni_ga_harvest/Output/00_regioni/"
    f"region_snp_membership_{WINDOW_LABEL}_{TRAIT}.csv"
)

SAVE_TARGET = GA_DIR / f"y_mean_adjusted_by_genotype_{TRAIT}.csv"
SAVE_X_SNP = GA_DIR / f"X_snp_top{TOP_K}_regions_{WINDOW_LABEL}_{TRAIT}.csv"
SAVE_X_PCA = GA_DIR / f"X_pca_20_{TRAIT}.csv"
SAVE_X_FULL = GA_DIR / f"X_snp_plus_pca_top{TOP_K}_regions_{WINDOW_LABEL}_{TRAIT}.csv"
SAVE_SNP_META = GA_DIR / f"snp_metadata_top{TOP_K}_regions_{WINDOW_LABEL}_{TRAIT}.csv"
SAVE_REPORT = GA_DIR / "G0_build_ga_target_and_inputs_no_soil_report.txt"


# =============================================================================
# HELPERS
# =============================================================================

def detect_genotype_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Nessuna colonna Genotype trovata. "
        f"Colonne disponibili: {df.columns.tolist()[:30]}"
    )


def clean_genotype_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("^G_", "", regex=True).str.strip()


def standardize_pca_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    geno_col = detect_genotype_column(df, ["Genotype", "genotype", "Sample", "sample"])
    df = df.rename(columns={geno_col: "Genotype"})
    df["Genotype"] = clean_genotype_series(df["Genotype"])
    return df


def standardize_adj_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    colmap = {}
    for c in df.columns:
        cl = c.lower().strip()

        if cl == "genotype":
            colmap[c] = "Genotype"
        elif "adjusted" in cl:
            colmap[c] = "Harvest_date_adjusted"
        elif cl == "envir":
            colmap[c] = "Envir"

    df = df.rename(columns=colmap)

    needed = {"Genotype", "Harvest_date_adjusted"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"Nel file adjusted mancano colonne richieste {needed}. "
            f"Colonne trovate: {df.columns.tolist()}"
        )

    df["Genotype"] = clean_genotype_series(df["Genotype"])
    df["Harvest_date_adjusted"] = pd.to_numeric(
        df["Harvest_date_adjusted"],
        errors="coerce"
    )

    return df


def standardize_allgeno(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    geno_col = detect_genotype_column(df, ["Genotype", "genotype"])
    df = df.rename(columns={geno_col: "Genotype"})
    df["Genotype"] = clean_genotype_series(df["Genotype"])
    return df


def ensure_required_files():
    files = [
        ALL_GENO_FILE,
        PCA_FILE,
        ADJ_FILE,
        TOP_REGIONS_FILE,
        REGION_MEMBERSHIP_FILE,
    ]

    missing = [f for f in files if not f.exists()]

    if len(missing) > 0:
        msg = "File mancanti:\n" + "\n".join(str(f) for f in missing)
        raise FileNotFoundError(msg)


def get_region_id_column(df: pd.DataFrame) -> str:
    if "region_id" in df.columns:
        return "region_id"
    if "region" in df.columns:
        return "region"
    raise ValueError(
        f"Non trovo region_id/region nel file top regions. "
        f"Colonne trovate: {df.columns.tolist()}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("G0 - BUILD GA INPUTS FROM NO-SOIL TOP REGIONS")
    print("=" * 80)

    ensure_required_files()

    print("Reading top regions no-soil...")
    top_regions = pd.read_csv(TOP_REGIONS_FILE)

    region_col = get_region_id_column(top_regions)

    if region_col != "region_id":
        top_regions = top_regions.rename(columns={region_col: "region_id"})

    top_regions["region_id"] = top_regions["region_id"].astype(str)
    region_ids = set(top_regions["region_id"])

    print(f"Top regions loaded: {len(region_ids)}")

    print("Reading original region membership...")
    membership = pd.read_csv(REGION_MEMBERSHIP_FILE)

    required_membership_cols = {
        "SNP",
        "CHROM",
        "POS",
        "region_start",
        "region_end",
        "region_id",
    }

    missing_membership = required_membership_cols - set(membership.columns)
    if len(missing_membership) > 0:
        raise ValueError(
            f"Nel membership file mancano colonne: {missing_membership}\n"
            f"Colonne trovate: {membership.columns.tolist()}"
        )

    membership["region_id"] = membership["region_id"].astype(str)
    membership["SNP"] = membership["SNP"].astype(str)

    membership_top = membership[membership["region_id"].isin(region_ids)].copy()

    candidate_snps = sorted(membership_top["SNP"].unique().tolist())

    print(f"Candidate SNPs from no-soil top regions: {len(candidate_snps)}")

    print("Reading target adjusted file...")
    adj = pd.read_csv(ADJ_FILE)
    adj = standardize_adj_columns(adj)

    target_df = (
        adj.groupby("Genotype", as_index=False)
        .agg(
            Harvest_date_adjusted_mean=("Harvest_date_adjusted", "mean"),
            n_env_obs=("Harvest_date_adjusted", "size")
        )
    )

    print("Reading PCA file...")
    pca = pd.read_csv(PCA_FILE)
    pca = standardize_pca_columns(pca)

    print("Reading all.geno header...")
    header = pd.read_csv(ALL_GENO_FILE, nrows=0)
    header_cols = header.columns.tolist()

    available_snps = [s for s in candidate_snps if s in header_cols]
    missing_snps = sorted(set(candidate_snps) - set(available_snps))

    print(f"Candidate SNPs present in all.geno: {len(available_snps)}")
    print(f"Missing candidate SNPs in all.geno: {len(missing_snps)}")

    if len(available_snps) == 0:
        raise ValueError("Nessuno SNP candidato è presente in all.geno.")

    usecols = ["Genotype"] + available_snps

    print("Reading filtered all.geno...")
    geno = pd.read_csv(ALL_GENO_FILE, usecols=usecols)
    geno = standardize_allgeno(geno)

    snp_cols = [c for c in geno.columns if c != "Genotype"]
    geno[snp_cols] = geno[snp_cols].apply(pd.to_numeric, errors="coerce")

    print("Building SNP metadata...")

    # Merge membership with top region SHAP/ranking metrics.
    top_region_cols_preferred = [
        "region_id",
        "CHROM",
        "region_start",
        "region_end",
        "region_score",
        "rank_by_region_score",
        "mean_region_SHAP",
        "max_region_SHAP",
        "sum_region_SHAP",
        "mean_n_folds",
        "max_n_folds",
        "mean_top20_count",
        "max_top20_count",
        "mean_top50_count",
        "max_top50_count",
        "n_snps",
        "n_mapped_snps",
        "n_unmapped_snps",
        "n_genes_inside",
        "n_genes_nearby_10kb",
        "genes_inside",
        "genes_nearby_10kb",
    ]

    top_region_cols = [c for c in top_region_cols_preferred if c in top_regions.columns]

    snp_meta = membership_top[membership_top["SNP"].isin(available_snps)].copy()

    # Avoid duplicated CHROM/region_start/region_end columns if already present in membership.
    merge_cols = [
        c for c in top_region_cols
        if c == "region_id" or c not in snp_meta.columns
    ]

    snp_meta = snp_meta.merge(
        top_regions[merge_cols],
        on="region_id",
        how="left"
    ).drop_duplicates().reset_index(drop=True)

    print("Merging genotype, target and PCA...")
    merged = geno.merge(target_df, on="Genotype", how="inner")
    merged = merged.merge(pca, on="Genotype", how="inner")

    print(f"Final merged genotypes: {merged['Genotype'].nunique()}")

    # Target
    y = merged[["Genotype", "Harvest_date_adjusted_mean", "n_env_obs"]].copy()
    y.to_csv(SAVE_TARGET, index=False)

    # SNP only
    X_snp = merged[["Genotype"] + available_snps].copy()
    X_snp.to_csv(SAVE_X_SNP, index=False)

    # PCA only
    pca_cols = [c for c in pca.columns if c != "Genotype"]
    X_pca = merged[["Genotype"] + pca_cols].copy()
    X_pca.to_csv(SAVE_X_PCA, index=False)

    # SNP + PCA
    X_full = merged[["Genotype"] + available_snps + pca_cols].copy()
    X_full.to_csv(SAVE_X_FULL, index=False)

    snp_meta.to_csv(SAVE_SNP_META, index=False)

    print("Writing report...")
    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        f.write("=== G0 BUILD GA TARGET AND INPUTS REPORT - NO SOIL ===\n\n")
        f.write(f"TRAIT: {TRAIT}\n")
        f.write(f"TOP_K: {TOP_K}\n")
        f.write(f"WINDOW_LABEL: {WINDOW_LABEL}\n\n")

        f.write(f"TOP_REGIONS_FILE:\n{TOP_REGIONS_FILE}\n\n")
        f.write(f"REGION_MEMBERSHIP_FILE:\n{REGION_MEMBERSHIP_FILE}\n\n")

        f.write(f"Top regions requested: {len(region_ids)}\n")
        f.write(f"Candidate SNPs from top regions: {len(candidate_snps)}\n")
        f.write(f"Candidate SNPs present in all.geno: {len(available_snps)}\n")
        f.write(f"Missing candidate SNPs in all.geno: {len(missing_snps)}\n")
        f.write(f"Target genotypes available: {target_df['Genotype'].nunique()}\n")
        f.write(f"PCA genotypes available: {pca['Genotype'].nunique()}\n")
        f.write(f"Final merged genotypes: {merged['Genotype'].nunique()}\n")
        f.write(f"Final SNP feature count: {len(available_snps)}\n")
        f.write(f"Final PCA feature count: {len(pca_cols)}\n\n")

        f.write("Target summary:\n")
        f.write(y["Harvest_date_adjusted_mean"].describe().to_string())
        f.write("\n\n")

        f.write("n_env_obs summary:\n")
        f.write(y["n_env_obs"].describe().to_string())
        f.write("\n\n")

        f.write("SNP per region summary:\n")
        f.write(snp_meta.groupby("region_id")["SNP"].nunique().describe().to_string())
        f.write("\n\n")

        if len(missing_snps) > 0:
            f.write("Missing SNPs in all.geno, first 100:\n")
            f.write("\n".join(missing_snps[:100]))
            f.write("\n")

    print("\nSaved:")
    print(SAVE_TARGET)
    print(SAVE_X_SNP)
    print(SAVE_X_PCA)
    print(SAVE_X_FULL)
    print(SAVE_SNP_META)
    print(SAVE_REPORT)

    print("\nSummary:")
    print(f"Top regions: {len(region_ids)}")
    print(f"Candidate SNPs: {len(candidate_snps)}")
    print(f"Available SNPs: {len(available_snps)}")
    print(f"Final genotypes: {merged['Genotype'].nunique()}")


if __name__ == "__main__":
    main()
