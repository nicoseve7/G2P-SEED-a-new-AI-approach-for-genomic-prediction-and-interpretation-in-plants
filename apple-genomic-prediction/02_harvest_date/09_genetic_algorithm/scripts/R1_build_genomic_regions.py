# Costruisce regioni genomiche:

# 25 kb
# 50 kb
# 100 kb

# e produce report comparativo

# -*- coding: utf-8 -*-

################################################################################
### R1_build_genomic_regions.py
### Build fixed genomic regions from SNP SHAP table
################################################################################

from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# SETTINGS
# =============================================================================

TRAIT = "Harvest_date"
MODEL_NAME = (
    "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"
)

GA_ROOT = (
    Path("02_harvest_date")
    / "09_genetic_algorithm"
)

OUTPUT_DIR = GA_ROOT / "output"

REGION_IN_DIR = (
    OUTPUT_DIR
    / "00_region_membership"
)

REGION_OUT_DIR = (
    OUTPUT_DIR
    / "00_region_membership"
)

REGION_OUT_DIR.mkdir(parents=True, exist_ok=True)

SNP_TABLE_FILE = REGION_IN_DIR / f"snp_table_for_regions_{MODEL_NAME}_{TRAIT}.csv"

WINDOW_SIZES = [25000, 50000, 100000]


# =============================================================================
# HELPERS
# =============================================================================

def assign_fixed_regions(df: pd.DataFrame, window_bp: int) -> pd.DataFrame:
    out = df.copy()

    out["region_start"] = ((out["POS"] - 1) // window_bp) * window_bp + 1
    out["region_end"] = out["region_start"] + window_bp - 1
    out["region_id"] = (
        out["CHROM"].astype(str)
        + ":"
        + out["region_start"].astype(str)
        + "-"
        + out["region_end"].astype(str)
    )
    out["window_bp"] = window_bp

    return out


def summarize_regions(region_df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        region_df.groupby(["window_bp", "CHROM", "region_start", "region_end", "region_id"])
        .agg(
            n_snps=("SNP", "nunique"),
            n_mapped_snps=("mapping_category", lambda x: int((x == "mapped").sum())),
            n_unmapped_snps=("mapping_category", lambda x: int((x == "unmapped").sum())),
            mean_region_SHAP=("meanSHAP", "mean"),
            median_region_SHAP=("meanSHAP", "median"),
            max_region_SHAP=("meanSHAP", "max"),
            sum_region_SHAP=("meanSHAP", "sum"),
            mean_n_folds=("n_folds", "mean"),
            max_n_folds=("n_folds", "max"),
            mean_top20_count=("top20_count", "mean"),
            max_top20_count=("top20_count", "max"),
            mean_top50_count=("top50_count", "mean"),
            max_top50_count=("top50_count", "max"),
            best_snp_rank=("bestRank", "min"),
        )
        .reset_index()
    )

    grp = grp.sort_values(
        ["mean_region_SHAP", "max_region_SHAP", "mean_top20_count", "n_snps"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    grp["region_rank"] = np.arange(1, len(grp) + 1)

    return grp


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not SNP_TABLE_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {SNP_TABLE_FILE}")

    snp_df = pd.read_csv(SNP_TABLE_FILE)

    needed = {
        "SNP", "CHROM", "POS", "meanSHAP", "n_folds",
        "top20_count", "top50_count", "bestRank", "mapping_category"
    }
    if not needed.issubset(snp_df.columns):
        raise ValueError(
            f"Il file input regioni deve contenere almeno {needed}. "
            f"Colonne trovate: {snp_df.columns.tolist()}"
        )

    report_lines = []
    report_lines.append("=== R1 BUILD GENOMIC REGIONS REPORT ===\n")

    for window_bp in WINDOW_SIZES:
        print(f"Building regions for {window_bp} bp...")

        tmp = assign_fixed_regions(snp_df, window_bp)
        region_snp_file = REGION_OUT_DIR / f"region_snp_membership_{window_bp//1000}kb_{TRAIT}.csv"
        region_summary_file = REGION_OUT_DIR / f"region_summary_{window_bp//1000}kb_{TRAIT}.csv"

        tmp.to_csv(region_snp_file, index=False)

        region_summary = summarize_regions(tmp)
        region_summary.to_csv(region_summary_file, index=False)

        report_lines.append(f"\n--- WINDOW {window_bp} bp ---\n")
        report_lines.append(f"Rows in membership table: {len(tmp)}\n")
        report_lines.append(f"Unique regions: {region_summary['region_id'].nunique()}\n")
        report_lines.append(f"Mean SNP per region: {region_summary['n_snps'].mean():.3f}\n")
        report_lines.append(f"Median SNP per region: {region_summary['n_snps'].median():.3f}\n")
        report_lines.append(f"Regions with 1 SNP: {(region_summary['n_snps'] == 1).sum()}\n")
        report_lines.append(f"Regions with >=5 SNP: {(region_summary['n_snps'] >= 5).sum()}\n")
        report_lines.append(f"Regions with >=10 SNP: {(region_summary['n_snps'] >= 10).sum()}\n")

        print(f"Saved: {region_snp_file}")
        print(f"Saved: {region_summary_file}")

    report_file = REGION_OUT_DIR / "R1_build_genomic_regions_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print("Saved:")
    print(report_file)


if __name__ == "__main__":
    main()
