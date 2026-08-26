# -*- coding: utf-8 -*-

################################################################################
### H4_compare_full_vs_no_soil_shap_regions.py
###
### Compare SHAP interpretation between:
###   1) full V3 model
###   2) no-soil V3 model
###
### It computes:
###   - SNP overlap: top 20, 50, 100, 200, 500
###   - Gene overlap: top 10, 20, 50, 100, 300, 500
###   - 50kb region rankings using composite score
###   - Region overlap: top 100, 500, 1000
###
### Run from:
###   dalpaper/senza_suolo/
################################################################################

from pathlib import Path
import gzip
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

TRAIT = "Harvest_date"
WINDOW_SIZE = 50_000
WINDOW_LABEL = "50kb"

FULL_MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_interpretation"
NO_SOIL_MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"

FULL_V3_TOP1000_REGION_FILE = Path(
    "../regioni_ga_harvest/Output/02_regioni_ranked/"
    "top1000_regions_by_region_score_50kb_Harvest_date.csv"
)

# Full V3 aggregated SHAP tables
FULL_AGG_DIR = Path(
    "../paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_interpretation/"
    "Output/Interpretation/Harvest_date/Aggregated_tables"
)

# No-soil aggregated SHAP tables
NO_SOIL_AGG_DIR = Path(
    "Output/Interpretation/Harvest_date/Aggregated_tables"
)

# Output
OUT_DIR = Path("Output/Interpretation/Harvest_date/Full_vs_no_soil_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_DIR = OUT_DIR / "region_rankings_50kb"
REGION_DIR.mkdir(parents=True, exist_ok=True)

OVERLAP_DIR = OUT_DIR / "overlap_tables"
OVERLAP_DIR.mkdir(parents=True, exist_ok=True)

# Preferred existing SNP -> region mapping files.
# If one exists, the script uses it directly.
CANDIDATE_SNP_REGION_FILES = [
    Path("../regioni_ga_harvest/Output/00_regioni/region_snp_membership_50kb_Harvest_date.csv"),
    Path("../regioni_ga_harvest/Output/00_region_mapping/snp_to_region_50kb_Harvest_date.csv"),
    Path("../regioni_pc1_ga_harvest/Output/00_region_mapping/snp_to_region_50kb_Harvest_date.csv"),
    Path("Output/00_region_mapping/snp_to_region_50kb_Harvest_date.csv"),
]

# Fallback: if no SNP-region file exists, parse SNP positions from a VCF.
# Set this path to your VCF if needed.
VCF_FILE = Path("../Output/SNPS_final2022.vcf")
# Alternative examples:
# VCF_FILE = Path("../SNPS_final2022.vcf")
# VCF_FILE = Path("../SNPS_final2022.vcf.gz")


SNP_TOP_K = [20, 50, 100, 200, 500]
GENE_TOP_K = [10, 20, 50, 100, 300, 500]
REGION_TOP_K = [100, 500, 1000]

# Composite region score weights.
WEIGHTS = {
    "mean_region_SHAP": 0.40,
    "max_region_SHAP": 0.25,
    "mean_n_folds": 0.20,
    "max_top20_count": 0.15,
}


# =============================================================================
# PATHS
# =============================================================================

FULL_SNP_FILE = FULL_AGG_DIR / f"SUMMARY_all_snp_SHAP_{FULL_MODEL_NAME}_{TRAIT}.csv"
NO_SOIL_SNP_FILE = NO_SOIL_AGG_DIR / f"SUMMARY_all_snp_SHAP_{NO_SOIL_MODEL_NAME}_{TRAIT}.csv"

FULL_GENE_FILE = FULL_AGG_DIR / f"SUMMARY_gene_SHAP_{FULL_MODEL_NAME}_{TRAIT}.csv"
NO_SOIL_GENE_FILE = NO_SOIL_AGG_DIR / f"SUMMARY_gene_SHAP_{NO_SOIL_MODEL_NAME}_{TRAIT}.csv"


# =============================================================================
# BASIC HELPERS
# =============================================================================

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")


def minmax_scale(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    min_v = s.min()
    max_v = s.max()

    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - min_v) / (max_v - min_v)


def parse_region(region: str):
    """
    Parse region string like:
      3:30650001-30700000
    """
    region = str(region)
    chrom, coords = region.split(":")
    start, end = coords.split("-")
    return chrom, int(float(start)), int(float(end))


def chrom_sort_key(chrom):
    c = str(chrom).replace("chr", "").replace("Chr", "").strip()
    try:
        return int(c)
    except Exception:
        return c


def build_region_from_pos(chrom, pos, window_size=WINDOW_SIZE):
    chrom = str(chrom).replace("chr", "").replace("Chr", "").strip()
    pos = int(float(pos))
    start = ((pos - 1) // window_size) * window_size + 1
    end = start + window_size - 1
    return f"{chrom}:{start}-{end}", chrom, start, end

def load_existing_full_v3_top1000_regions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing full V3 top1000 region file:\n{path}")

    df = pd.read_csv(path)

    if "region_id" in df.columns:
        df = df.rename(columns={"region_id": "region"})
    elif "region" not in df.columns:
        raise ValueError(
            f"Expected column 'region_id' or 'region' in {path}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df["region"] = df["region"].astype(str)

    if "region_score" not in df.columns:
        # fallback: if the score column has another name
        possible_score_cols = [
            "composite_score",
            "score",
            "final_score",
            "region_composite_score",
        ]
        found = [c for c in possible_score_cols if c in df.columns]
        if len(found) == 0:
            raise ValueError(
                f"Could not find region_score column in {path}. "
                f"Found columns: {df.columns.tolist()}"
            )
        df = df.rename(columns={found[0]: "region_score"})

    df = df.sort_values("region_score", ascending=False).reset_index(drop=True)
    df["region_rank"] = np.arange(1, len(df) + 1)
    df["model_label"] = "full_v3_existing_top1000"

        # Normalize SNP count column name for compatibility with no-soil ranking
    if "n_snps_in_region_model" not in df.columns:
        if "n_snps" in df.columns:
            df["n_snps_in_region_model"] = df["n_snps"]
        else:
            df["n_snps_in_region_model"] = np.nan

    # Normalize top20 count column if needed
    if "max_top20_count" not in df.columns:
        possible_top20_cols = ["top20_count", "max_region_top20_count", "region_top20_count"]
        found = [c for c in possible_top20_cols if c in df.columns]
        df["max_top20_count"] = df[found[0]] if len(found) > 0 else np.nan

    # Normalize mean_n_folds if needed
    if "mean_n_folds" not in df.columns:
        possible_nfold_cols = ["mean_n_folds", "n_folds", "max_n_folds"]
        found = [c for c in possible_nfold_cols if c in df.columns]
        df["mean_n_folds"] = df[found[0]] if len(found) > 0 else np.nan

    return df
# =============================================================================
# LOAD SNP -> REGION MAPPING
# =============================================================================

# def load_existing_snp_region_mapping(path: Path) -> pd.DataFrame:
#     df = pd.read_csv(path)

#     # Flexible column normalization
#     colmap = {c.lower(): c for c in df.columns}

#     snp_col = None
#     for candidate in ["snp", "id", "marker", "snp_id"]:
#         if candidate in colmap:
#             snp_col = colmap[candidate]
#             break

#     region_col = None
#     for candidate in ["region", "window", "region_id"]:
#         if candidate in colmap:
#             region_col = colmap[candidate]
#             break

#     if snp_col is None or region_col is None:
#         raise ValueError(
#             f"Could not identify SNP and region columns in {path}. "
#             f"Columns found: {df.columns.tolist()}"
#         )

#     out = df[[snp_col, region_col]].copy()
#     out = out.rename(columns={snp_col: "SNP", region_col: "region"})

#     out["SNP"] = out["SNP"].astype(str)
#     out["region"] = out["region"].astype(str)

#     parsed = out["region"].apply(parse_region)
#     out["chromosome"] = [x[0] for x in parsed]
#     out["region_start"] = [x[1] for x in parsed]
#     out["region_end"] = [x[2] for x in parsed]

#     out = out.drop_duplicates(subset=["SNP", "region"]).reset_index(drop=True)

#     return out
def load_existing_snp_region_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Caso specifico dei tuoi file region_snp_membership_50kb_Harvest_date.csv
    if {"SNP", "region_id", "CHROM", "region_start", "region_end"}.issubset(df.columns):
        out = df[["SNP", "region_id", "CHROM", "region_start", "region_end"]].copy()
        out = out.rename(columns={
            "region_id": "region",
            "CHROM": "chromosome",
        })

        out["SNP"] = out["SNP"].astype(str)
        out["region"] = out["region"].astype(str)
        out["chromosome"] = out["chromosome"].astype(str)
        out["region_start"] = pd.to_numeric(out["region_start"], errors="coerce").astype("Int64")
        out["region_end"] = pd.to_numeric(out["region_end"], errors="coerce").astype("Int64")

        out = out.drop_duplicates(subset=["SNP", "region"]).reset_index(drop=True)
        return out

    # Fallback flessibile per eventuali altri file
    colmap = {c.lower(): c for c in df.columns}

    snp_col = None
    for candidate in ["snp", "id", "marker", "snp_id"]:
        if candidate in colmap:
            snp_col = colmap[candidate]
            break

    region_col = None
    for candidate in ["region", "region_id", "window", "window_id"]:
        if candidate in colmap:
            region_col = colmap[candidate]
            break

    if snp_col is None or region_col is None:
        raise ValueError(
            f"Could not identify SNP and region columns in {path}. "
            f"Columns found: {df.columns.tolist()}"
        )

    out = df[[snp_col, region_col]].copy()
    out = out.rename(columns={
        snp_col: "SNP",
        region_col: "region",
    })

    out["SNP"] = out["SNP"].astype(str)
    out["region"] = out["region"].astype(str)

    parsed = out["region"].apply(parse_region)
    out["chromosome"] = [x[0] for x in parsed]
    out["region_start"] = [x[1] for x in parsed]
    out["region_end"] = [x[2] for x in parsed]

    out = out.drop_duplicates(subset=["SNP", "region"]).reset_index(drop=True)

    return out

def parse_vcf_positions(vcf_file: Path) -> pd.DataFrame:
    """
    Parse CHROM, POS, ID from VCF or VCF.GZ.
    This only reads the variant columns, not genotype data.
    """
    if not vcf_file.exists():
        raise FileNotFoundError(f"VCF file not found:\n{vcf_file}")

    print(f"[INFO] Parsing VCF positions from: {vcf_file}")

    rows = []

    opener = gzip.open if str(vcf_file).endswith(".gz") else open

    with opener(vcf_file, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            chrom = parts[0]
            pos = parts[1]
            snp = parts[2]

            if snp == "." or snp == "":
                continue

            region, chrom_clean, start, end = build_region_from_pos(chrom, pos)

            rows.append({
                "SNP": str(snp),
                "chromosome": str(chrom_clean),
                "position": int(float(pos)),
                "region": region,
                "region_start": start,
                "region_end": end,
            })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        raise ValueError(f"No SNP positions parsed from {vcf_file}")

    out = out.drop_duplicates(subset=["SNP"]).reset_index(drop=True)

    return out


def load_or_create_snp_region_mapping() -> pd.DataFrame:
    for f in CANDIDATE_SNP_REGION_FILES:
        if f.exists():
            print(f"[INFO] Using existing SNP-region mapping:")
            print(f)
            mapping = load_existing_snp_region_mapping(f)
            return mapping

    print("[INFO] No existing SNP-region mapping found.")
    print("[INFO] Trying to create mapping from VCF.")

    mapping = parse_vcf_positions(VCF_FILE)

    save_dir = OUT_DIR / "snp_region_mapping"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_file = save_dir / f"snp_to_region_{WINDOW_LABEL}_{TRAIT}.csv"
    mapping.to_csv(save_file, index=False)

    print(f"[INFO] Saved SNP-region mapping:")
    print(save_file)

    return mapping


# =============================================================================
# LOAD SHAP SUMMARIES
# =============================================================================

def load_snp_summary(path: Path, model_label: str) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)
    df["SNP"] = df["SNP"].astype(str)
    df["model_label"] = model_label

    needed = {"SNP", "meanSHAP", "n_folds", "top20_count", "top50_count"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.sort_values(
        ["meanSHAP", "top20_count", "top50_count", "meanRank"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)

    return df


def load_gene_summary(path: Path, model_label: str) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)
    df["Gene"] = df["Gene"].astype(str)
    df["model_label"] = model_label

    needed = {"Gene", "meanSHAP", "n_folds", "top20_count", "top50_count"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.sort_values(
        ["meanSHAP", "top20_count", "top50_count", "meanRank"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)

    return df


# =============================================================================
# REGION RANKING
# =============================================================================

def build_region_ranking(snp_summary: pd.DataFrame, snp_region_map: pd.DataFrame, model_label: str) -> pd.DataFrame:
    df = snp_summary.copy()

    merged = df.merge(
        snp_region_map[["SNP", "region", "chromosome", "region_start", "region_end"]],
        on="SNP",
        how="left"
    )

    n_missing = merged["region"].isna().sum()
    if n_missing > 0:
        print(f"[WARNING] {model_label}: {n_missing} SNPs without region mapping will be dropped.")

    merged = merged.dropna(subset=["region"]).copy()

    merged["region"] = merged["region"].astype(str)
    merged["chromosome"] = merged["chromosome"].astype(str)
    merged["region_start"] = pd.to_numeric(merged["region_start"], errors="coerce")
    merged["region_end"] = pd.to_numeric(merged["region_end"], errors="coerce")

    region_df = (
        merged.groupby(["region", "chromosome", "region_start", "region_end"])
        .agg(
            n_snps_in_region_model=("SNP", "nunique"),
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
            best_snp_rank=("rank", "min"),
            unique_snps=("SNP", lambda x: ";".join(sorted(pd.Series(x).dropna().astype(str).unique()))),
        )
        .reset_index()
    )

    # Normalized components for composite score
    region_df["norm_mean_region_SHAP"] = minmax_scale(region_df["mean_region_SHAP"])
    region_df["norm_max_region_SHAP"] = minmax_scale(region_df["max_region_SHAP"])
    region_df["norm_mean_n_folds"] = minmax_scale(region_df["mean_n_folds"])
    region_df["norm_max_top20_count"] = minmax_scale(region_df["max_top20_count"])

    region_df["region_score"] = (
        WEIGHTS["mean_region_SHAP"] * region_df["norm_mean_region_SHAP"]
        + WEIGHTS["max_region_SHAP"] * region_df["norm_max_region_SHAP"]
        + WEIGHTS["mean_n_folds"] * region_df["norm_mean_n_folds"]
        + WEIGHTS["max_top20_count"] * region_df["norm_max_top20_count"]
    )

    region_df["model_label"] = model_label

    region_df = region_df.sort_values(
        ["region_score", "mean_region_SHAP", "max_region_SHAP", "mean_n_folds", "max_top20_count"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    region_df["region_rank"] = np.arange(1, len(region_df) + 1)

    # Nice column order
    preferred = [
        "model_label",
        "region_rank",
        "region",
        "chromosome",
        "region_start",
        "region_end",
        "region_score",
        "mean_region_SHAP",
        "max_region_SHAP",
        "sum_region_SHAP",
        "mean_n_folds",
        "max_n_folds",
        "max_top20_count",
        "max_top50_count",
        "n_snps_in_region_model",
        "best_snp_rank",
        "unique_snps",
        "norm_mean_region_SHAP",
        "norm_max_region_SHAP",
        "norm_mean_n_folds",
        "norm_max_top20_count",
    ]

    rest = [c for c in region_df.columns if c not in preferred]
    region_df = region_df[[c for c in preferred if c in region_df.columns] + rest]

    return region_df


# =============================================================================
# OVERLAP
# =============================================================================

def compute_overlap(df_a, df_b, id_col, rank_col, score_col, top_k_list, label_a="full", label_b="no_soil"):
    summary_rows = []
    detail_rows = []

    for k in top_k_list:
        top_a = df_a.sort_values(rank_col).head(k).copy()
        top_b = df_b.sort_values(rank_col).head(k).copy()

        set_a = set(top_a[id_col].astype(str))
        set_b = set(top_b[id_col].astype(str))

        inter = sorted(set_a & set_b)
        only_a = sorted(set_a - set_b)
        only_b = sorted(set_b - set_a)
        union = set_a | set_b

        summary_rows.append({
            "entity": id_col,
            "top_k": k,
            f"n_{label_a}": len(set_a),
            f"n_{label_b}": len(set_b),
            "n_overlap": len(inter),
            "n_union": len(union),
            "jaccard": len(inter) / len(union) if len(union) > 0 else np.nan,
            f"overlap_fraction_vs_{label_a}": len(inter) / len(set_a) if len(set_a) > 0 else np.nan,
            f"overlap_fraction_vs_{label_b}": len(inter) / len(set_b) if len(set_b) > 0 else np.nan,
            f"n_only_{label_a}": len(only_a),
            f"n_only_{label_b}": len(only_b),
        })

        a_rank = top_a[[id_col, rank_col, score_col]].copy()
        a_rank = a_rank.rename(columns={
            rank_col: f"rank_{label_a}",
            score_col: f"score_{label_a}",
        })

        b_rank = top_b[[id_col, rank_col, score_col]].copy()
        b_rank = b_rank.rename(columns={
            rank_col: f"rank_{label_b}",
            score_col: f"score_{label_b}",
        })

        detail = pd.DataFrame({id_col: sorted(union)})
        detail = detail.merge(a_rank, on=id_col, how="left")
        detail = detail.merge(b_rank, on=id_col, how="left")

        detail["top_k"] = k
        detail["in_both"] = detail[id_col].isin(inter)
        detail[f"only_{label_a}"] = detail[id_col].isin(only_a)
        detail[f"only_{label_b}"] = detail[id_col].isin(only_b)

        detail_rows.append(detail)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_rows, ignore_index=True)

    return summary, details


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("H4 - FULL V3 vs NO-SOIL SHAP COMPARISON")
    print("=" * 80)

    print("\nLoading SHAP summary tables...")

    full_snp = load_snp_summary(FULL_SNP_FILE, "full_v3")
    nosoil_snp = load_snp_summary(NO_SOIL_SNP_FILE, "no_soil")

    full_gene = load_gene_summary(FULL_GENE_FILE, "full_v3")
    nosoil_gene = load_gene_summary(NO_SOIL_GENE_FILE, "no_soil")

    print(f"Full SNP summary: {full_snp.shape}")
    print(f"No-soil SNP summary: {nosoil_snp.shape}")
    print(f"Full gene summary: {full_gene.shape}")
    print(f"No-soil gene summary: {nosoil_gene.shape}")

    # -------------------------------------------------------------------------
    # SNP overlap
    # -------------------------------------------------------------------------
    print("\nComputing SNP overlaps...")

    snp_overlap_summary, snp_overlap_details = compute_overlap(
        df_a=full_snp,
        df_b=nosoil_snp,
        id_col="SNP",
        rank_col="rank",
        score_col="meanSHAP",
        top_k_list=SNP_TOP_K,
        label_a="full_v3",
        label_b="no_soil",
    )

    snp_overlap_summary.to_csv(OVERLAP_DIR / "overlap_summary_snps.csv", index=False)
    snp_overlap_details.to_csv(OVERLAP_DIR / "overlap_details_snps.csv", index=False)

    # -------------------------------------------------------------------------
    # Gene overlap
    # -------------------------------------------------------------------------
    print("Computing gene overlaps...")

    gene_overlap_summary, gene_overlap_details = compute_overlap(
        df_a=full_gene,
        df_b=nosoil_gene,
        id_col="Gene",
        rank_col="rank",
        score_col="meanSHAP",
        top_k_list=GENE_TOP_K,
        label_a="full_v3",
        label_b="no_soil",
    )

    gene_overlap_summary.to_csv(OVERLAP_DIR / "overlap_summary_genes.csv", index=False)
    gene_overlap_details.to_csv(OVERLAP_DIR / "overlap_details_genes.csv", index=False)

    # -------------------------------------------------------------------------
    # Region ranking
    # -------------------------------------------------------------------------
    print("\nLoading/creating SNP -> 50kb region mapping...")

    snp_region_map = load_or_create_snp_region_mapping()
    snp_region_map.to_csv(REGION_DIR / f"snp_to_region_{WINDOW_LABEL}_used_for_comparison.csv", index=False)

    print(f"SNP-region mapping shape: {snp_region_map.shape}")

    # print("\nBuilding region ranking for full V3...")
    # full_regions = build_region_ranking(full_snp, snp_region_map, "full_v3")

    # print("Building region ranking for no-soil...")
    # nosoil_regions = build_region_ranking(nosoil_snp, snp_region_map, "no_soil")
    print("\nLoading existing full V3 top1000 region ranking...")
    full_regions = load_existing_full_v3_top1000_regions(FULL_V3_TOP1000_REGION_FILE)

    print("Building region ranking for no-soil...")
    nosoil_regions = build_region_ranking(nosoil_snp, snp_region_map, "no_soil")

    full_region_file = REGION_DIR / f"region_ranking_{WINDOW_LABEL}_full_v3.csv"
    nosoil_region_file = REGION_DIR / f"region_ranking_{WINDOW_LABEL}_no_soil.csv"

    full_regions.to_csv(full_region_file, index=False)
    nosoil_regions.to_csv(nosoil_region_file, index=False)

    print(f"Saved full region ranking: {full_region_file}")
    print(f"Saved no-soil region ranking: {nosoil_region_file}")

    # -------------------------------------------------------------------------
    # Region overlap
    # -------------------------------------------------------------------------
    print("\nComputing region overlaps...")

    region_overlap_summary, region_overlap_details = compute_overlap(
        df_a=full_regions,
        df_b=nosoil_regions,
        id_col="region",
        rank_col="region_rank",
        score_col="region_score",
        top_k_list=REGION_TOP_K,
        label_a="full_v3",
        label_b="no_soil",
    )

    region_overlap_summary.to_csv(OVERLAP_DIR / "overlap_summary_regions_50kb.csv", index=False)
    region_overlap_details.to_csv(OVERLAP_DIR / "overlap_details_regions_50kb.csv", index=False)

    # -------------------------------------------------------------------------
    # Combined summary
    # -------------------------------------------------------------------------
    combined_overlap = pd.concat(
        [
            snp_overlap_summary.assign(level="SNP"),
            gene_overlap_summary.assign(level="Gene"),
            region_overlap_summary.assign(level="Region_50kb"),
        ],
        ignore_index=True
    )

    combined_overlap.to_csv(OVERLAP_DIR / "overlap_summary_all_levels.csv", index=False)

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    report_file = OUT_DIR / "H4_full_vs_no_soil_overlap_report.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FULL V3 vs NO-SOIL SHAP COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"TRAIT: {TRAIT}\n")
        f.write(f"WINDOW: {WINDOW_LABEL} ({WINDOW_SIZE} bp)\n\n")

        f.write("Composite region score weights:\n")
        for k, v in WEIGHTS.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("SNP overlap summary:\n")
        f.write(snp_overlap_summary.to_string(index=False))
        f.write("\n\n")

        f.write("Gene overlap summary:\n")
        f.write(gene_overlap_summary.to_string(index=False))
        f.write("\n\n")

        f.write("Region 50kb overlap summary:\n")
        f.write(region_overlap_summary.to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 full V3 SNPs:\n")
        f.write(full_snp.head(20)[["rank", "SNP", "meanSHAP", "n_folds", "top20_count", "top50_count"]].to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 no-soil SNPs:\n")
        f.write(nosoil_snp.head(20)[["rank", "SNP", "meanSHAP", "n_folds", "top20_count", "top50_count"]].to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 full V3 genes:\n")
        f.write(full_gene.head(20)[["rank", "Gene", "meanSHAP", "n_folds", "top20_count", "top50_count"]].to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 no-soil genes:\n")
        f.write(nosoil_gene.head(20)[["rank", "Gene", "meanSHAP", "n_folds", "top20_count", "top50_count"]].to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 full V3 regions:\n")
        f.write(full_regions.head(20)[[
            "region_rank", "region", "region_score", "mean_region_SHAP",
            "max_region_SHAP", "mean_n_folds", "max_top20_count",
            "n_snps_in_region_model"
        ]].to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 no-soil regions:\n")
        f.write(nosoil_regions.head(20)[[
            "region_rank", "region", "region_score", "mean_region_SHAP",
            "max_region_SHAP", "mean_n_folds", "max_top20_count",
            "n_snps_in_region_model"
        ]].to_string(index=False))
        f.write("\n\n")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print("\nMain outputs:")
    print(OVERLAP_DIR / "overlap_summary_all_levels.csv")
    print(OVERLAP_DIR / "overlap_details_snps.csv")
    print(OVERLAP_DIR / "overlap_details_genes.csv")
    print(OVERLAP_DIR / "overlap_details_regions_50kb.csv")
    print(full_region_file)
    print(nosoil_region_file)
    print(report_file)

    print("\nQuick overlap summary:")
    print(combined_overlap.to_string(index=False))


if __name__ == "__main__":
    main()