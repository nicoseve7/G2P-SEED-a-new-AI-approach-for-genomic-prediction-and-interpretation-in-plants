# -*- coding: utf-8 -*-

################################################################################
### Q7_rank_regions_from_shap_no_soil_newtraits.py
###
### Build 50kb/100kb region ranking from NO-SOIL SHAP SNP summary
### for:
###   - Acidity
###   - Color_over
###
### This script:
###   1. reads aggregated all-SNP SHAP summary for each trait
###   2. reads SNP coordinates from PLINK .bim
###   3. assigns each SNP to genomic windows: 50kb and 100kb
###   4. computes region-level SHAP metrics
###   5. ranks regions with the usual composite score:
###
###      region_score =
###        0.40 * mean_region_SHAP_norm
###      + 0.25 * max_region_SHAP_norm
###      + 0.20 * mean_n_folds_norm
###      + 0.15 * max_top20_count_norm
################################################################################

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"

SHAP_OUT_DIR = (
    Path("03_acidity_color_over")
    / "05_shap"
    / "output"
)

GA_OUT_DIR = (
    Path("03_acidity_color_over")
    / "06_genetic_algorithm"
    / "output"
)

OUT_BASE_DIR = GA_OUT_DIR / "01_region_ranking"

# PLINK .bim file:
# columns are normally:
#   CHROM, SNP, CM, POS, A1, A2
BIM_FILE = (
    Path("data")
    / "raw"
    / "genotype"
    / "SNPs_final_2022.bim"
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

def find_existing_file(candidates, label):
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Nessun file trovato per: {label}\n"
        "Path provati:\n" + "\n".join(str(x) for x in candidates)
    )


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


def get_trait_dirs(trait: str):
    agg_dir = SHAP_OUT_DIR / trait / "Aggregated_tables"

    out_dir = OUT_BASE_DIR / trait
    out_dir.mkdir(parents=True, exist_ok=True)

    return agg_dir, out_dir


def get_shap_file(trait: str):
    agg_dir, _ = get_trait_dirs(trait)

    shap_file = (
        agg_dir
        / f"SUMMARY_all_snp_SHAP_{MODEL_NAME}_{trait}.csv"
    )

    if not shap_file.exists():
        raise FileNotFoundError(
            f"File SHAP SNP summary non trovato per {trait}:\n"
            f"{shap_file}\n"
            "Hai eseguito Q5?"
        )

    return shap_file


def load_snp_shap(trait: str) -> pd.DataFrame:
    shap_file = get_shap_file(trait)

    df = pd.read_csv(shap_file)

    ensure_columns(
        df,
        [
            "SNP",
            "meanSHAP",
            "medianSHAP",
            "sdSHAP",
            "minSHAP",
            "maxSHAP",
            "n_folds",
            "meanRank",
            "medianRank",
            "bestRank",
            "worstRank",
            "top20_count",
            "top50_count",
        ],
        file_label=str(shap_file),
    )

    df["SNP"] = df["SNP"].astype(str).str.strip()

    # Numeric conversion
    numeric_cols = [
        "meanSHAP",
        "medianSHAP",
        "sdSHAP",
        "minSHAP",
        "maxSHAP",
        "n_folds",
        "meanRank",
        "medianRank",
        "bestRank",
        "worstRank",
        "top20_count",
        "top50_count",
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SNP", "meanSHAP"]).copy()

    return df


def load_bim_coordinates() -> pd.DataFrame:
    if not BIM_FILE.exists():
    raise FileNotFoundError(
        f"BIM file non trovato:\n{BIM_FILE}"
    )

    bim_file = BIM_FILE

    print(f"[INFO] Uso BIM file: {bim_file}")

    bim = pd.read_csv(
        bim_file,
        sep=r"\s+",
        header=None,
        names=["CHROM", "SNP", "CM", "POS", "A1", "A2"],
        dtype={
            "CHROM": str,
            "SNP": str,
            "CM": str,
            "POS": int,
            "A1": str,
            "A2": str,
        },
    )

    bim["SNP"] = bim["SNP"].astype(str).str.strip()
    bim["CHROM"] = bim["CHROM"].astype(str).str.strip()
    bim["POS"] = pd.to_numeric(bim["POS"], errors="coerce")

    bim = bim.dropna(subset=["SNP", "CHROM", "POS"]).copy()
    bim["POS"] = bim["POS"].astype(int)

    # Remove duplicate SNP IDs if present
    dup_count = bim["SNP"].duplicated().sum()
    if dup_count > 0:
        print(f"[WARNING] BIM contiene {dup_count} SNP duplicati. Tengo la prima occorrenza.")
        bim = bim.drop_duplicates(subset=["SNP"], keep="first").copy()

    return bim[["SNP", "CHROM", "POS"]]


def build_membership_from_bim(snp_shap: pd.DataFrame, bim: pd.DataFrame, window_bp: int):
    """
    Assigns SHAP SNPs to fixed genomic windows.

    region_start is 1-based:
        1, 50001, 100001, ...
    """
    snps = snp_shap[["SNP"]].drop_duplicates().copy()

    membership = snps.merge(
        bim,
        on="SNP",
        how="left",
        validate="one_to_one",
    )

    n_missing_coord = int(membership["POS"].isna().sum())

    if n_missing_coord > 0:
        missing_examples = (
            membership.loc[membership["POS"].isna(), "SNP"]
            .head(20)
            .tolist()
        )

        print(
            f"[WARNING] {n_missing_coord} SNP SHAP non hanno coordinate nel BIM. "
            "Saranno esclusi dal ranking regioni."
        )
        print("Esempi SNP senza coordinate:", missing_examples)

    membership = membership.dropna(subset=["CHROM", "POS"]).copy()
    membership["POS"] = membership["POS"].astype(int)

    membership["window_bp"] = int(window_bp)
    membership["region_start"] = ((membership["POS"] - 1) // window_bp) * window_bp + 1
    membership["region_end"] = membership["region_start"] + window_bp - 1

    membership["region_id"] = (
        "chr" + membership["CHROM"].astype(str)
        + ":"
        + membership["region_start"].astype(str)
        + "-"
        + membership["region_end"].astype(str)
    )

    membership = membership[
        [
            "SNP",
            "CHROM",
            "POS",
            "window_bp",
            "region_start",
            "region_end",
            "region_id",
        ]
    ].copy()

    return membership, n_missing_coord


def build_region_summary(trait: str, window_bp: int, snp_shap: pd.DataFrame, bim: pd.DataFrame):
    membership, n_missing_coord = build_membership_from_bim(
        snp_shap=snp_shap,
        bim=bim,
        window_bp=window_bp,
    )

    merged = membership.merge(
        snp_shap,
        on="SNP",
        how="left",
        validate="one_to_one",
    )

    n_missing_shap = int(merged["meanSHAP"].isna().sum())

    if n_missing_shap > 0:
        print(
            f"[WARNING] {trait} {window_bp//1000}kb: "
            f"{n_missing_shap} SNP con coordinate ma senza SHAP."
        )

    merged = merged.dropna(subset=["meanSHAP"]).copy()

    region_df = (
        merged.groupby(
            [
                "window_bp",
                "CHROM",
                "region_start",
                "region_end",
                "region_id",
            ]
        )
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
            snps_in_region=(
                "SNP",
                lambda x: ";".join(
                    sorted(pd.Series(x).dropna().astype(str).unique())
                ),
            ),
        )
        .reset_index()
    )

    # We do not have stable mapped/unmapped labels here unless the SHAP summary includes them.
    # If the summary contains MappedStatus, use it.
    if "MappedStatus" in snp_shap.columns:
        tmp = merged[["region_id", "SNP", "MappedStatus"]].copy()
        tmp["MappedStatus"] = tmp["MappedStatus"].astype(str)

        map_summary = (
            tmp.groupby("region_id")
            .agg(
                n_mapped_snps=("MappedStatus", lambda x: (x == "mapped").sum()),
                n_unmapped_snps=("MappedStatus", lambda x: (x == "unmapped").sum()),
            )
            .reset_index()
        )

        region_df = region_df.merge(map_summary, on="region_id", how="left")
    else:
        region_df["n_mapped_snps"] = np.nan
        region_df["n_unmapped_snps"] = np.nan

    region_df["n_mapped_snps"] = region_df["n_mapped_snps"].fillna(0).astype(int)
    region_df["n_unmapped_snps"] = region_df["n_unmapped_snps"].fillna(0).astype(int)

    region_df["Trait"] = trait
    region_df["n_missing_coord_snps_total"] = n_missing_coord

    return region_df, membership


def build_region_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["mean_region_SHAP_norm"] = minmax_norm(out["mean_region_SHAP"])
    out["max_region_SHAP_norm"] = minmax_norm(out["max_region_SHAP"])
    out["mean_n_folds_norm"] = minmax_norm(out["mean_n_folds"])
    out["max_top20_count_norm"] = minmax_norm(out["max_top20_count"])

    out["region_score"] = (
        W_MEAN_SHAP * out["mean_region_SHAP_norm"]
        + W_MAX_SHAP * out["max_region_SHAP_norm"]
        + W_MEAN_NFOLDS * out["mean_n_folds_norm"]
        + W_MAX_TOP20 * out["max_top20_count_norm"]
    )

    out["rank_by_mean_region_SHAP"] = out["mean_region_SHAP"].rank(
        method="dense",
        ascending=False,
    ).astype(int)

    out["rank_by_max_region_SHAP"] = out["max_region_SHAP"].rank(
        method="dense",
        ascending=False,
    ).astype(int)

    out["rank_by_region_score"] = out["region_score"].rank(
        method="dense",
        ascending=False,
    ).astype(int)

    out = out.sort_values(
        [
            "rank_by_region_score",
            "rank_by_mean_region_SHAP",
            "rank_by_max_region_SHAP",
        ],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return out


def save_region_outputs(trait: str, window_bp: int, ranked: pd.DataFrame, membership: pd.DataFrame, out_dir: Path):
    kb = window_bp // 1000

    full_out = out_dir / f"ranked_regions_{kb}kb_{trait}.csv"
    top_score_out = out_dir / f"top{TOP_K}_regions_by_region_score_{kb}kb_{trait}.csv"
    top_mean_out = out_dir / f"top{TOP_K}_regions_by_meanSHAP_{kb}kb_{trait}.csv"
    top_max_out = out_dir / f"top{TOP_K}_regions_by_maxSHAP_{kb}kb_{trait}.csv"
    membership_out = out_dir / f"region_snp_membership_{kb}kb_{trait}.csv"

    ranked.to_csv(full_out, index=False)
    ranked.sort_values("rank_by_region_score").head(TOP_K).to_csv(top_score_out, index=False)
    ranked.sort_values("rank_by_mean_region_SHAP").head(TOP_K).to_csv(top_mean_out, index=False)
    ranked.sort_values("rank_by_max_region_SHAP").head(TOP_K).to_csv(top_max_out, index=False)
    membership.to_csv(membership_out, index=False)

    return {
        "full_out": full_out,
        "top_score_out": top_score_out,
        "top_mean_out": top_mean_out,
        "top_max_out": top_max_out,
        "membership_out": membership_out,
    }


def process_one_trait(trait: str, bim: pd.DataFrame):
    print("\n" + "=" * 100)
    print(f"Q7 - RANK REGIONS FROM NO-SOIL SHAP | TRAIT: {trait}")
    print("=" * 100)

    _, out_dir = get_trait_dirs(trait)

    snp_shap = load_snp_shap(trait)

    print(f"Loaded SNP SHAP summary for {trait}: {snp_shap.shape}")
    print(f"Unique SNPs in SHAP summary: {snp_shap['SNP'].nunique()}")

    report_lines = []
    report_lines.append("=" * 100 + "\n")
    report_lines.append(f"Q7 RANK REGIONS FROM NO-SOIL SHAP REPORT: {trait}\n")
    report_lines.append("=" * 100 + "\n\n")
    report_lines.append(f"Trait: {trait}\n")
    report_lines.append(f"Model: {MODEL_NAME}\n")
    report_lines.append(f"TOP_K: {TOP_K}\n")
    report_lines.append(
        "Composite score weights:\n"
        f"  mean_region_SHAP: {W_MEAN_SHAP}\n"
        f"  max_region_SHAP: {W_MAX_SHAP}\n"
        f"  mean_n_folds: {W_MEAN_NFOLDS}\n"
        f"  max_top20_count: {W_MAX_TOP20}\n\n"
    )

    summary_rows = []

    for window_bp in WINDOW_SIZES:
        kb = window_bp // 1000

        print(f"\nProcessing {kb}kb regions...")

        region_summary, membership = build_region_summary(
            trait=trait,
            window_bp=window_bp,
            snp_shap=snp_shap,
            bim=bim,
        )

        needed_cols = [
            "window_bp",
            "CHROM",
            "region_start",
            "region_end",
            "region_id",
            "n_snps",
            "mean_region_SHAP",
            "max_region_SHAP",
            "mean_n_folds",
            "max_n_folds",
            "mean_top20_count",
            "max_top20_count",
        ]

        ensure_columns(region_summary, needed_cols, file_label=f"{trait} {kb}kb region summary")

        ranked = build_region_score(region_summary)

        paths = save_region_outputs(
            trait=trait,
            window_bp=window_bp,
            ranked=ranked,
            membership=membership,
            out_dir=out_dir,
        )

        top_score = ranked.sort_values("rank_by_region_score").head(TOP_K).copy()
        top_mean = ranked.sort_values("rank_by_mean_region_SHAP").head(TOP_K).copy()
        top_max = ranked.sort_values("rank_by_max_region_SHAP").head(TOP_K).copy()

        set_score = set(top_score["region_id"].astype(str))
        set_mean = set(top_mean["region_id"].astype(str))
        set_max = set(top_max["region_id"].astype(str))

        overlap_score_mean = len(set_score & set_mean)
        overlap_score_max = len(set_score & set_max)
        overlap_mean_max = len(set_mean & set_max)

        n_unique_snps_in_top_score = (
            top_score["snps_in_region"]
            .dropna()
            .astype(str)
            .str.split(";")
            .explode()
            .nunique()
        )

        summary_rows.append({
            "Trait": trait,
            "window_kb": kb,
            "n_regions_total": len(ranked),
            "n_membership_rows": len(membership),
            "n_unique_snps_with_coordinates": membership["SNP"].nunique(),
            "n_missing_coord_snps": int(region_summary["n_missing_coord_snps_total"].max()) if len(region_summary) else 0,
            f"n_top{TOP_K}_score_regions": len(top_score),
            f"n_unique_snps_in_top{TOP_K}_score_regions": n_unique_snps_in_top_score,
            "overlap_score_vs_mean": overlap_score_mean,
            "overlap_score_vs_max": overlap_score_max,
            "overlap_mean_vs_max": overlap_mean_max,
            "full_ranked_file": str(paths["full_out"]),
            "top_score_file": str(paths["top_score_out"]),
            "membership_file": str(paths["membership_out"]),
        })

        report_lines.append(f"--- WINDOW {kb}kb ---\n")
        report_lines.append(f"Total ranked regions: {len(ranked)}\n")
        report_lines.append(f"Membership rows: {len(membership)}\n")
        report_lines.append(f"Unique SNPs with coordinates: {membership['SNP'].nunique()}\n")
        report_lines.append(
            f"Missing coordinate SNPs: "
            f"{int(region_summary['n_missing_coord_snps_total'].max()) if len(region_summary) else 0}\n"
        )
        report_lines.append(f"Top {TOP_K} by composite score: {len(top_score)}\n")
        report_lines.append(f"Unique SNPs in top {TOP_K} score regions: {n_unique_snps_in_top_score}\n")
        report_lines.append(f"Overlap score vs mean: {overlap_score_mean}\n")
        report_lines.append(f"Overlap score vs max: {overlap_score_max}\n")
        report_lines.append(f"Overlap mean vs max: {overlap_mean_max}\n\n")

        cols_for_report = [
            "Trait",
            "region_id",
            "CHROM",
            "region_start",
            "region_end",
            "n_snps",
            "n_mapped_snps",
            "n_unmapped_snps",
            "mean_region_SHAP",
            "max_region_SHAP",
            "mean_n_folds",
            "max_top20_count",
            "region_score",
            "rank_by_region_score",
        ]
        cols_for_report = [c for c in cols_for_report if c in top_score.columns]

        report_lines.append("Top 20 regions by composite score:\n")
        report_lines.append(top_score[cols_for_report].head(20).to_string(index=False))
        report_lines.append("\n\n")

        print("Saved:")
        print(paths["full_out"])
        print(paths["top_score_out"])
        print(paths["top_mean_out"])
        print(paths["top_max_out"])
        print(paths["membership_out"])

    report_file = out_dir / f"Q7_rank_regions_from_no_soil_shap_report_{trait}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    trait_summary = pd.DataFrame(summary_rows)
    trait_summary_file = out_dir / f"Q7_region_ranking_summary_{trait}.csv"
    trait_summary.to_csv(trait_summary_file, index=False)

    print("\nSaved report:")
    print(report_file)
    print("Saved trait summary:")
    print(trait_summary_file)

    return trait_summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("Q7 - RANK REGIONS FROM NO-SOIL SHAP NEW TRAITS")
    print("=" * 100)

    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    bim = load_bim_coordinates()

    print(f"BIM coordinates loaded: {bim.shape}")
    print(f"Unique SNPs in BIM: {bim['SNP'].nunique()}")

    all_summaries = []

    for trait in TRAITS:
        trait_summary = process_one_trait(trait, bim)
        all_summaries.append(trait_summary)

    summary_all = pd.concat(all_summaries, ignore_index=True)

    summary_all_file = OUT_BASE_DIR / "Q7_region_ranking_summary_all_traits.csv"
    summary_all.to_csv(summary_all_file, index=False)

    print("\n" + "=" * 100)
    print("Q7 completed.")
    print("=" * 100)
    print(summary_all.to_string(index=False))
    print("\nSaved global summary:")
    print(summary_all_file)


if __name__ == "__main__":
    main()
