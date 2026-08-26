# -*- coding: utf-8 -*-

################################################################################
### R3_rank_regions_from_shap_no_soil.py
###
### Build 50kb/100kb region ranking from NO-SOIL SHAP SNP summary.
###
### This script:
###   1. uses existing SNP -> region membership files from regioni_ga_harvest
###   2. uses no-soil SHAP all-SNP summary from senza_suolo
###   3. recomputes region-level SHAP metrics
###   4. optionally merges static region/gene annotations from original V3 files
###   5. ranks regions using the same composite score:
###
###      region_score =
###        0.40 * mean_region_SHAP_norm
###      + 0.25 * max_region_SHAP_norm
###      + 0.20 * mean_n_folds_norm
###      + 0.15 * max_top20_count_norm
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

GA_ROOT = (
    Path("02_harvest_date")
    / "09_genetic_algorithm"
)

OUTPUT_DIR = GA_ROOT / "output"

RANK_DIR = (
    OUTPUT_DIR
    / "02_regioni_ranked"
)
RANK_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_REGION_DIR = (
    OUTPUT_DIR
    / "00_region_membership"
)

ORIGINAL_ANNOT_DIR = (
    OUTPUT_DIR
    / "01_region_annotations"
)

NO_SOIL_MODEL_NAME = (
    "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"
)

NO_SOIL_AGG_DIR = (
    Path("02_harvest_date")
    / "08_shap"
    / "output"
    / "Interpretation"
    / TRAIT
    / "Aggregated_tables"
)

# No-soil SHAP SNP summary

NO_SOIL_SNP_SHAP_FILE = (
    NO_SOIL_AGG_DIR
    / f"SUMMARY_all_snp_SHAP_{NO_SOIL_MODEL_NAME}_{TRAIT}.csv"
)

WINDOW_SIZES = [50000, 100000]
TOP_K = 1000

# Composite score weights
W_MEAN_SHAP = 0.40
W_MAX_SHAP = 0.25
W_MEAN_NFOLDS = 0.20
W_MAX_TOP20 = 0.15


# =============================================================================
# HELPERS
# =============================================================================

def minmax_norm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    smin = s.min()
    smax = s.max()

    if pd.isna(smin) or pd.isna(smax):
        return pd.Series(np.zeros(len(s)), index=s.index)

    if smax == smin:
        return pd.Series(np.ones(len(s)), index=s.index)

    return (s - smin) / (smax - smin)


def ensure_columns(df: pd.DataFrame, needed_cols, file_label="dataframe"):
    missing = [c for c in needed_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(
            f"Colonne mancanti in {file_label}: {missing}\n"
            f"Colonne trovate: {df.columns.tolist()}"
        )


def load_no_soil_snp_shap() -> pd.DataFrame:
    if not NO_SOIL_SNP_SHAP_FILE.exists():
        raise FileNotFoundError(f"File SHAP no-soil non trovato:\n{NO_SOIL_SNP_SHAP_FILE}")

    df = pd.read_csv(NO_SOIL_SNP_SHAP_FILE)
    ensure_columns(
        df,
        ["SNP", "meanSHAP", "medianSHAP", "sdSHAP", "minSHAP", "maxSHAP",
         "n_folds", "meanRank", "medianRank", "bestRank", "worstRank",
         "top20_count", "top50_count"],
        file_label=str(NO_SOIL_SNP_SHAP_FILE)
    )

    df["SNP"] = df["SNP"].astype(str)

    return df


def load_membership(window_bp: int) -> pd.DataFrame:
    kb = window_bp // 1000
    infile = ORIGINAL_REGION_DIR / f"region_snp_membership_{kb}kb_{TRAIT}.csv"

    if not infile.exists():
        raise FileNotFoundError(f"File membership non trovato:\n{infile}")

    df = pd.read_csv(infile)

    needed = ["SNP", "CHROM", "POS", "region_start", "region_end", "region_id", "window_bp"]
    ensure_columns(df, needed, file_label=str(infile))

    df["SNP"] = df["SNP"].astype(str)
    df["CHROM"] = df["CHROM"].astype(str)
    df["region_id"] = df["region_id"].astype(str)

    return df


def load_static_annotations(window_bp: int) -> pd.DataFrame:
    """
    Loads original annotated region summary only for static information:
    genes_inside, genes_nearby_10kb, n_genes_inside, n_genes_nearby_10kb, etc.

    SHAP-related columns from this file will NOT be used for ranking.
    """
    kb = window_bp // 1000
    infile = ORIGINAL_ANNOT_DIR / f"region_summary_annotated_{kb}kb_{TRAIT}.csv"

    if not infile.exists():
        print(f"[WARNING] Annotated static region file not found:\n{infile}")
        return pd.DataFrame()

    df = pd.read_csv(infile)

    if "region_id" not in df.columns:
        print(f"[WARNING] 'region_id' missing in annotated file:\n{infile}")
        return pd.DataFrame()

    df["region_id"] = df["region_id"].astype(str)

    keep_cols = [
        "region_id",
        "n_genes_inside",
        "genes_inside",
        "n_genes_nearby_10kb",
        "genes_nearby_10kb",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    out = df[keep_cols].drop_duplicates(subset=["region_id"]).copy()

    return out


def clean_gene_lists(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["genes_inside", "genes_nearby_10kb"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)
            out[col] = out[col].str.replace(r"^gene:", "", regex=True)
            out[col] = out[col].str.replace(r";gene:", ";", regex=True)

    return out


def build_region_summary_no_soil(window_bp: int, snp_shap: pd.DataFrame) -> pd.DataFrame:
    membership = load_membership(window_bp)

    # Keep only mapping/static columns from membership.
    membership_cols = [
        "SNP",
        "CHROM",
        "POS",
        "region_start",
        "region_end",
        "region_id",
        "window_bp",
    ]

    membership = membership[membership_cols].copy()

    merged = membership.merge(
        snp_shap,
        on="SNP",
        how="left"
    )

    n_missing = merged["meanSHAP"].isna().sum()
    if n_missing > 0:
        print(f"[WARNING] {window_bp//1000}kb: {n_missing} SNP rows without no-soil SHAP values.")

    merged = merged.dropna(subset=["meanSHAP"]).copy()

    # Basic region-level summary
    region_df = (
        merged.groupby(["window_bp", "CHROM", "region_start", "region_end", "region_id"])
        .agg(
            n_snps=("SNP", "nunique"),
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
            best_snp_rank=("meanRank", "min"),
            snps_in_region=("SNP", lambda x: ";".join(sorted(pd.Series(x).dropna().astype(str).unique()))),
        )
        .reset_index()
    )

    # Recover mapped/unmapped counts if possible using original membership columns.
    # In the membership file you showed there is mapping_category, but we did not keep it above.
    membership_full = load_membership(window_bp)
    if "mapping_category" in membership_full.columns:
        tmp = membership_full[["SNP", "region_id", "mapping_category"]].copy()
        tmp["SNP"] = tmp["SNP"].astype(str)
        tmp["region_id"] = tmp["region_id"].astype(str)

        tmp = tmp[tmp["SNP"].isin(snp_shap["SNP"].astype(str))].copy()

        map_summary = (
            tmp.groupby("region_id")
            .agg(
                n_mapped_snps=("mapping_category", lambda x: (x.astype(str) == "mapped").sum()),
                n_unmapped_snps=("mapping_category", lambda x: (x.astype(str) == "unmapped").sum()),
            )
            .reset_index()
        )

        region_df = region_df.merge(map_summary, on="region_id", how="left")
    else:
        region_df["n_mapped_snps"] = np.nan
        region_df["n_unmapped_snps"] = np.nan

    region_df["n_mapped_snps"] = region_df["n_mapped_snps"].fillna(0).astype(int)
    region_df["n_unmapped_snps"] = region_df["n_unmapped_snps"].fillna(0).astype(int)

    # Merge static annotations from original annotated region summaries.
    annot = load_static_annotations(window_bp)
    if len(annot) > 0:
        region_df = region_df.merge(annot, on="region_id", how="left")

    # Ensure annotation columns exist
    for col in ["n_genes_inside", "n_genes_nearby_10kb"]:
        if col not in region_df.columns:
            region_df[col] = 0
        region_df[col] = pd.to_numeric(region_df[col], errors="coerce").fillna(0).astype(int)

    for col in ["genes_inside", "genes_nearby_10kb"]:
        if col not in region_df.columns:
            region_df[col] = ""

    region_df = clean_gene_lists(region_df)

    return region_df


def build_region_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["mean_region_SHAP_norm"] = minmax_norm(out["mean_region_SHAP"])
    out["max_region_SHAP_norm"] = minmax_norm(out["max_region_SHAP"])
    out["mean_n_folds_norm"] = minmax_norm(out["mean_n_folds"])
    out["max_top20_count_norm"] = minmax_norm(out["max_top20_count"])

    out["region_score"] = (
        W_MEAN_SHAP * out["mean_region_SHAP_norm"] +
        W_MAX_SHAP * out["max_region_SHAP_norm"] +
        W_MEAN_NFOLDS * out["mean_n_folds_norm"] +
        W_MAX_TOP20 * out["max_top20_count_norm"]
    )

    out["rank_by_mean_region_SHAP"] = out["mean_region_SHAP"].rank(
        method="dense",
        ascending=False
    ).astype(int)

    out["rank_by_max_region_SHAP"] = out["max_region_SHAP"].rank(
        method="dense",
        ascending=False
    ).astype(int)

    out["rank_by_region_score"] = out["region_score"].rank(
        method="dense",
        ascending=False
    ).astype(int)

    out = out.sort_values(
        ["rank_by_region_score", "rank_by_mean_region_SHAP", "rank_by_max_region_SHAP"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    report_lines = []
    report_lines.append("=== R3 RANK REGIONS FROM NO-SOIL SHAP REPORT ===\n\n")
    report_lines.append(f"TRAIT = {TRAIT}\n")
    report_lines.append(f"NO_SOIL_MODEL_NAME = {NO_SOIL_MODEL_NAME}\n")
    report_lines.append(f"TOP_K = {TOP_K}\n")
    report_lines.append(
        f"Composite score weights: meanSHAP={W_MEAN_SHAP}, "
        f"maxSHAP={W_MAX_SHAP}, mean_n_folds={W_MEAN_NFOLDS}, max_top20={W_MAX_TOP20}\n\n"
    )
    report_lines.append(f"No-soil SNP SHAP file:\n{NO_SOIL_SNP_SHAP_FILE}\n\n")

    print("=" * 80)
    print("R3 - RANK REGIONS FROM NO-SOIL SHAP")
    print("=" * 80)

    snp_shap = load_no_soil_snp_shap()
    print(f"Loaded no-soil SNP SHAP summary: {snp_shap.shape}")

    for window_bp in WINDOW_SIZES:
        print(f"\nProcessing {window_bp} bp regions...")

        region_summary = build_region_summary_no_soil(window_bp, snp_shap)

        needed_cols = [
            "window_bp", "CHROM", "region_start", "region_end", "region_id",
            "n_snps", "n_mapped_snps", "n_unmapped_snps",
            "mean_region_SHAP", "max_region_SHAP",
            "mean_n_folds", "max_n_folds",
            "mean_top20_count", "max_top20_count",
            "n_genes_inside", "n_genes_nearby_10kb",
        ]
        ensure_columns(region_summary, needed_cols, file_label=f"{window_bp}bp region summary")

        ranked = build_region_score(region_summary)

        # Save full ranked table
        full_out = RANK_DIR / f"ranked_regions_{window_bp//1000}kb_{TRAIT}.csv"
        ranked.to_csv(full_out, index=False)

        # Top by composite score
        top_score = ranked.sort_values("rank_by_region_score").head(TOP_K).copy()
        top_score_out = RANK_DIR / f"top{TOP_K}_regions_by_region_score_{window_bp//1000}kb_{TRAIT}.csv"
        top_score.to_csv(top_score_out, index=False)

        # Top by mean SHAP
        top_mean = ranked.sort_values("rank_by_mean_region_SHAP").head(TOP_K).copy()
        top_mean_out = RANK_DIR / f"top{TOP_K}_regions_by_meanSHAP_{window_bp//1000}kb_{TRAIT}.csv"
        top_mean.to_csv(top_mean_out, index=False)

        # Top by max SHAP
        top_max = ranked.sort_values("rank_by_max_region_SHAP").head(TOP_K).copy()
        top_max_out = RANK_DIR / f"top{TOP_K}_regions_by_maxSHAP_{window_bp//1000}kb_{TRAIT}.csv"
        top_max.to_csv(top_max_out, index=False)

        # Overlap diagnostics
        set_score = set(top_score["region_id"].astype(str))
        set_mean = set(top_mean["region_id"].astype(str))
        set_max = set(top_max["region_id"].astype(str))

        overlap_score_mean = len(set_score & set_mean)
        overlap_score_max = len(set_score & set_max)
        overlap_mean_max = len(set_mean & set_max)

        report_lines.append(f"--- WINDOW {window_bp} bp ---\n")
        report_lines.append(f"Total regions: {len(ranked)}\n")
        report_lines.append(f"Top {TOP_K} by composite score: {len(top_score)}\n")
        report_lines.append(f"Top {TOP_K} by mean_region_SHAP: {len(top_mean)}\n")
        report_lines.append(f"Top {TOP_K} by max_region_SHAP: {len(top_max)}\n")
        report_lines.append(f"Overlap score vs mean: {overlap_score_mean}\n")
        report_lines.append(f"Overlap score vs max: {overlap_score_max}\n")
        report_lines.append(f"Overlap mean vs max: {overlap_mean_max}\n")

        report_lines.append("\nTop 20 regions by composite score:\n")

        cols_for_report = [
            "region_id", "CHROM", "region_start", "region_end",
            "n_snps", "n_mapped_snps", "n_unmapped_snps",
            "mean_region_SHAP", "max_region_SHAP",
            "mean_n_folds", "max_top20_count",
            "n_genes_inside", "n_genes_nearby_10kb",
            "region_score",
            "rank_by_region_score",
        ]
        cols_for_report = [c for c in cols_for_report if c in top_score.columns]

        report_lines.append(top_score[cols_for_report].head(20).to_string(index=False))
        report_lines.append("\n\n")

        print("Saved:")
        print(full_out)
        print(top_score_out)
        print(top_mean_out)
        print(top_max_out)

    report_file = RANK_DIR / "R3_rank_regions_from_no_soil_shap_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print("\nSaved:")
    print(report_file)


if __name__ == "__main__":
    main()
