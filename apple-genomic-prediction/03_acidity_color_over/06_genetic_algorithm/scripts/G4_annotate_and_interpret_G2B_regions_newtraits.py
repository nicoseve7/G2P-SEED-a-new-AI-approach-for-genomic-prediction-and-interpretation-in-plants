# -*- coding: utf-8 -*-

################################################################################
### G4_annotate_and_interpret_G2B_regions_newtraits.py
###
### Region-level biological interpretation after:
###   G2B_multiseed_ga_newtraits.py
###
### For:
###   - Acidity
###   - Color_over
###
### Produces, for each trait:
###   - annotated stable region tables
###   - high-confidence region subsets
###   - stable SNP annotation table
###   - region stability plots
###   - chromosome distribution plots
###   - final interpretation report
###
### Input:
###   Output/06_ga_runs/G2B_multiseed_ga_newtraits/<TRAIT>/
###   Output/04_regioni_annotate/
###   Output/05_ga_inputs/
###
### Output:
###   Output/06_ga_runs/G2B_multiseed_ga_newtraits/<TRAIT>/figures_G4/
###   Output/06_ga_runs/G2B_multiseed_ga_newtraits/<TRAIT>/tables_G4/
################################################################################

from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

WINDOW_LABEL = "50kb"
TOP_K = 1000

BASE_RUN_DIR = Path("Output/06_ga_runs/G2B_multiseed_ga_newtraits")
GA_INPUT_DIR = Path("Output/05_ga_inputs")
ANNOTATION_DIR = Path("Output/04_regioni_annotate")

DPI = 300
TOP_N = 25

REGION_FREQ_HIGH_CONFIDENCE = 0.7
REGION_FREQ_STABLE = 0.5
SNP_FREQ_STABLE = 0.3


# =============================================================================
# BASIC UTILS
# =============================================================================

def savefig(fig_dir: Path, name: str):
    path = fig_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")


def read_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")
    return pd.read_csv(path)


def detect_column(df, candidates, required=False, label="column"):
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


def normalize_chr(x):
    x = str(x).strip()
    x = x.replace("chr", "").replace("Chr", "").replace("CHR", "")
    x = x.replace("GDDH13_", "")
    x = x.replace("gddh13_", "")
    x = x.strip()
    return x


def parse_region(region):
    """
    Parses region strings like:
        chr9:33750001-33800000
        9:33750001-33800000

    Returns:
        chromosome, start, end
    """
    region = str(region).strip()
    match = re.match(r"(.+):(\d+)-(\d+)", region)

    if not match:
        return None, np.nan, np.nan

    chrom = normalize_chr(match.group(1))
    start = int(match.group(2))
    end = int(match.group(3))

    return chrom, start, end


def make_region_string(chrom, start, end):
    return f"chr{normalize_chr(chrom)}:{int(start)}-{int(end)}"


def add_region_components(df, region_col="region"):
    out = df.copy()

    parsed = out[region_col].apply(parse_region)
    out["chromosome"] = [p[0] for p in parsed]
    out["region_start"] = [p[1] for p in parsed]
    out["region_end"] = [p[2] for p in parsed]
    out["region_midpoint"] = (out["region_start"] + out["region_end"]) / 2

    return out


def identify_snp_column(df):
    return detect_column(
        df,
        [
            "snp", "SNP", "snp_id", "SNP_ID", "ID", "id",
            "marker", "Marker", "variant", "Variant"
        ],
        required=False,
        label="SNP column"
    )


def identify_gene_columns(df):
    if df is None:
        return []

    gene_keywords = [
        "gene", "genes", "nearby", "annotation", "swiss", "uniprot",
        "description", "product", "functional"
    ]

    cols = []

    for c in df.columns:
        c_low = c.lower()
        if any(k in c_low for k in gene_keywords):
            cols.append(c)

    return cols


def natural_chr_sort_key(x):
    try:
        return int(str(x))
    except Exception:
        return 999


# =============================================================================
# PATHS PER TRAIT
# =============================================================================

def get_trait_paths(trait: str):
    run_dir = BASE_RUN_DIR / trait
    fig_dir = run_dir / "figures_G4"
    table_dir = run_dir / "tables_G4"

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "run_dir": run_dir,
        "fig_dir": fig_dir,
        "table_dir": table_dir,

        "region_stability_file": run_dir / f"G2B_region_stability_across_seeds_{trait}.csv",
        "snp_stability_file": run_dir / f"G2B_snp_stability_across_seeds_{trait}.csv",
        "gene_stability_file": run_dir / f"G2B_gene_stability_across_seeds_{trait}.csv",

        "selected_regions_long_file": run_dir / f"G2B_selected_regions_all_seeds_long_{trait}.csv",
        "selected_snps_long_file": run_dir / f"G2B_selected_snps_all_seeds_long_{trait}.csv",

        "snp_metadata_file": GA_INPUT_DIR / f"snp_metadata_top{TOP_K}_regions_{WINDOW_LABEL}_{trait}.csv",

        "annotated_region_file": ANNOTATION_DIR / f"top{TOP_K}_regions_by_region_score_annotated_{WINDOW_LABEL}_{trait}.csv",
        "full_annotated_region_file": ANNOTATION_DIR / f"ranked_regions_annotated_{WINDOW_LABEL}_{trait}.csv",
    }

    return paths


# =============================================================================
# ANNOTATION LOADING
# =============================================================================

def find_annotated_region_file(trait: str, paths: dict):
    preferred = paths["annotated_region_file"]

    if preferred.exists():
        return preferred

    full_preferred = paths["full_annotated_region_file"]

    if full_preferred.exists():
        return full_preferred

    candidates = []

    patterns = [
        f"*annotated*{WINDOW_LABEL}*{trait}*.csv",
        f"*{trait}*{WINDOW_LABEL}*annotated*.csv",
        f"*top{TOP_K}*annotated*{WINDOW_LABEL}*{trait}*.csv",
        f"*ranked*annotated*{WINDOW_LABEL}*{trait}*.csv",
    ]

    if ANNOTATION_DIR.exists():
        for pat in patterns:
            candidates.extend(list(ANNOTATION_DIR.rglob(pat)))

    candidates = sorted(set(candidates))

    if len(candidates) == 0:
        print(f"[WARNING] No annotated region file found for {trait}.")
        return None

    print(f"[INFO] Annotated region file candidates for {trait}:")
    for i, c in enumerate(candidates[:10]):
        print(f"  {i}: {c}")

    chosen = candidates[0]
    print(f"[INFO] Using annotated region file for {trait}:\n  {chosen}")

    return chosen


def prepare_annotated_regions(annotation_file):
    if annotation_file is None:
        return None

    annot = pd.read_csv(annotation_file)

    region_col = detect_column(
        annot,
        [
            "region", "Region", "region_id", "Region_ID",
            "window_id", "Window_ID", "region_label", "Region_label"
        ],
        required=False,
        label="region column"
    )

    if region_col is not None:
        annot = annot.rename(columns={region_col: "region"})
        annot["region"] = annot["region"].astype(str).str.strip()
        return annot

    chrom_col = detect_column(
        annot,
        ["chromosome", "Chromosome", "chrom", "CHROM", "chr", "Chr"],
        required=False,
        label="chromosome column"
    )

    start_col = detect_column(
        annot,
        [
            "region_start", "start", "Start", "window_start",
            "Window_start", "start_bp", "Start_bp", "START"
        ],
        required=False,
        label="start column"
    )

    end_col = detect_column(
        annot,
        [
            "region_end", "end", "End", "window_end",
            "Window_end", "end_bp", "End_bp", "END"
        ],
        required=False,
        label="end column"
    )

    if chrom_col is not None and start_col is not None and end_col is not None:
        annot["region"] = [
            make_region_string(c, s, e)
            for c, s, e in zip(annot[chrom_col], annot[start_col], annot[end_col])
        ]
        annot["region"] = annot["region"].astype(str).str.strip()
        return annot

    print("[WARNING] Could not detect or reconstruct region column in annotation file.")
    print(f"[WARNING] Annotation columns: {list(annot.columns)}")
    return annot


# =============================================================================
# LOAD AND MERGE
# =============================================================================

def load_data_for_trait(trait: str, paths: dict):
    print("\n" + "=" * 80)
    print(f"[LOAD] Loading G2B region/SNP stability outputs for {trait}")
    print("=" * 80)

    region_stability = read_required(paths["region_stability_file"])
    snp_stability = read_required(paths["snp_stability_file"])

    selected_regions_long = None
    selected_snps_long = None
    gene_stability = None

    if paths["selected_regions_long_file"].exists():
        selected_regions_long = pd.read_csv(paths["selected_regions_long_file"])

    if paths["selected_snps_long_file"].exists():
        selected_snps_long = pd.read_csv(paths["selected_snps_long_file"])

    if paths["gene_stability_file"].exists():
        gene_stability = pd.read_csv(paths["gene_stability_file"])

    snp_metadata = None

    if paths["snp_metadata_file"].exists():
        snp_metadata = pd.read_csv(paths["snp_metadata_file"])
    else:
        print(f"[WARNING] SNP metadata file not found:\n{paths['snp_metadata_file']}")

    annotation_file = find_annotated_region_file(trait, paths)
    annotated_regions = prepare_annotated_regions(annotation_file)

    print(f"[INFO] Region stability shape: {region_stability.shape}")
    print(f"[INFO] SNP stability shape: {snp_stability.shape}")

    if selected_regions_long is not None:
        print(f"[INFO] Selected regions long shape: {selected_regions_long.shape}")

    if selected_snps_long is not None:
        print(f"[INFO] Selected SNPs long shape: {selected_snps_long.shape}")

    if gene_stability is not None:
        print(f"[INFO] Gene stability shape: {gene_stability.shape}")

    if annotated_regions is not None:
        print(f"[INFO] Annotated regions shape: {annotated_regions.shape}")
    else:
        print("[WARNING] No annotated region table available.")

    if snp_metadata is not None:
        print(f"[INFO] SNP metadata shape: {snp_metadata.shape}")

    return (
        region_stability,
        snp_stability,
        selected_regions_long,
        selected_snps_long,
        gene_stability,
        snp_metadata,
        annotated_regions,
        annotation_file,
    )


def build_annotated_region_stability(region_stability, annotated_regions):
    region_stability = region_stability.copy()

    if "region" not in region_stability.columns:
        raise ValueError(
            "region_stability must contain a 'region' column.\n"
            f"Columns found: {region_stability.columns.tolist()}"
        )

    region_stability["region"] = region_stability["region"].astype(str).str.strip()

    region_stability = add_region_components(region_stability, region_col="region")

    if annotated_regions is None or "region" not in annotated_regions.columns:
        print("[WARNING] Cannot merge region stability with annotations.")
        return region_stability

    annotated_regions = annotated_regions.copy()
    annotated_regions["region"] = annotated_regions["region"].astype(str).str.strip()

    # Avoid duplicate base columns where possible.
    duplicate_cols = [
        c for c in annotated_regions.columns
        if c in region_stability.columns and c != "region"
    ]

    annotated_regions = annotated_regions.drop(columns=duplicate_cols, errors="ignore")

    merged = region_stability.merge(
        annotated_regions,
        on="region",
        how="left",
        suffixes=("", "_annot")
    )

    return merged


def build_annotated_snp_stability(snp_stability, snp_metadata):
    if snp_metadata is None:
        return snp_stability

    snp_stability = snp_stability.copy()

    snp_col_meta = identify_snp_column(snp_metadata)

    if snp_col_meta is None:
        print("[WARNING] Could not detect SNP column in SNP metadata.")
        return snp_stability

    snp_metadata = snp_metadata.rename(columns={snp_col_meta: "snp"})
    snp_metadata["snp"] = snp_metadata["snp"].astype(str).str.strip()

    if "snp" not in snp_stability.columns:
        snp_col_stability = identify_snp_column(snp_stability)
        if snp_col_stability is None:
            raise ValueError(
                "Could not detect SNP column in snp_stability.\n"
                f"Columns: {snp_stability.columns.tolist()}"
            )
        snp_stability = snp_stability.rename(columns={snp_col_stability: "snp"})

    snp_stability["snp"] = snp_stability["snp"].astype(str).str.strip()

    duplicate_cols = [
        c for c in snp_metadata.columns
        if c in snp_stability.columns and c != "snp"
    ]

    snp_metadata = snp_metadata.drop(columns=duplicate_cols, errors="ignore")

    merged = snp_stability.merge(
        snp_metadata,
        on="snp",
        how="left",
        suffixes=("", "_meta")
    )

    return merged


# =============================================================================
# TABLES
# =============================================================================

def save_filtered_tables(region_annot, snp_annot, table_dir: Path, trait: str):
    region_annot = region_annot.copy()
    snp_annot = snp_annot.copy()

    region_annot = region_annot.sort_values(
        ["n_seeds_selected", "n_unique_snps_selected", "n_selected_snp_events"],
        ascending=[False, False, False]
    )

    region_annot.to_csv(
        table_dir / f"G4_region_stability_annotated_all_{trait}.csv",
        index=False
    )

    stable_regions = region_annot[
        region_annot["selection_frequency"] >= REGION_FREQ_STABLE
    ].copy()

    high_conf_regions = region_annot[
        region_annot["selection_frequency"] >= REGION_FREQ_HIGH_CONFIDENCE
    ].copy()

    stable_regions.to_csv(
        table_dir / f"G4_stable_regions_freq_ge_{str(REGION_FREQ_STABLE).replace('.', '')}_{trait}.csv",
        index=False
    )

    high_conf_regions.to_csv(
        table_dir / f"G4_high_confidence_regions_freq_ge_{str(REGION_FREQ_HIGH_CONFIDENCE).replace('.', '')}_{trait}.csv",
        index=False
    )

    if "snp" not in snp_annot.columns:
        snp_col = identify_snp_column(snp_annot)
        if snp_col is not None:
            snp_annot = snp_annot.rename(columns={snp_col: "snp"})

    snp_annot = snp_annot.sort_values(
        ["n_seeds_selected", "snp"],
        ascending=[False, True]
    )

    snp_annot.to_csv(
        table_dir / f"G4_snp_stability_annotated_all_{trait}.csv",
        index=False
    )

    stable_snps = snp_annot[
        snp_annot["selection_frequency"] >= SNP_FREQ_STABLE
    ].copy()

    stable_snps.to_csv(
        table_dir / f"G4_stable_snps_freq_ge_{str(SNP_FREQ_STABLE).replace('.', '')}_{trait}.csv",
        index=False
    )

    # BED-like file for regions
    required_bed_cols = ["chromosome", "region_start", "region_end", "region", "selection_frequency"]

    if all(c in region_annot.columns for c in required_bed_cols):
        bed_df = region_annot[required_bed_cols].copy()
        bed_df = bed_df.dropna(subset=["chromosome", "region_start", "region_end"])

        if len(bed_df) > 0:
            bed_df["region_start_0based"] = bed_df["region_start"].astype(int) - 1

            bed_out = bed_df[
                ["chromosome", "region_start_0based", "region_end", "region", "selection_frequency"]
            ].copy()

            bed_out.to_csv(
                table_dir / f"G4_region_stability_annotated_all_{trait}.bed",
                sep="\t",
                index=False,
                header=False
            )

    return stable_regions, high_conf_regions, stable_snps


# =============================================================================
# PLOTS
# =============================================================================

def plot_top_regions(region_annot, fig_dir: Path, trait: str):
    top = region_annot.head(TOP_N).copy()
    top = top.iloc[::-1]

    if len(top) == 0:
        print(f"[WARNING] {trait}: no regions to plot.")
        return

    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    plt.barh(top["region"], top["selection_frequency"])
    plt.xlabel("Selection frequency")
    plt.ylabel("Region")
    plt.title(f"{trait} - top {TOP_N} stable genomic regions")
    plt.grid(axis="x", alpha=0.3)
    savefig(fig_dir, f"{trait}_top_stable_regions_selection_frequency.png")

    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    plt.barh(top["region"], top["n_unique_snps_selected"])
    plt.xlabel("Number of unique selected SNPs")
    plt.ylabel("Region")
    plt.title(f"{trait} - unique selected SNPs in top {TOP_N} regions")
    plt.grid(axis="x", alpha=0.3)
    savefig(fig_dir, f"{trait}_top_stable_regions_unique_snps.png")

    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    plt.barh(top["region"], top["n_selected_snp_events"])
    plt.xlabel("Selected SNP events across seeds")
    plt.ylabel("Region")
    plt.title(f"{trait} - SNP selection events in top {TOP_N} regions")
    plt.grid(axis="x", alpha=0.3)
    savefig(fig_dir, f"{trait}_top_stable_regions_snp_events.png")


def plot_top_snps(snp_annot, fig_dir: Path, trait: str):
    if "snp" not in snp_annot.columns:
        print(f"[WARNING] {trait}: SNP column missing. Skipping top SNP plot.")
        return

    top = snp_annot.head(TOP_N).copy()
    top = top.iloc[::-1]

    if len(top) == 0:
        print(f"[WARNING] {trait}: no SNPs to plot.")
        return

    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    plt.barh(top["snp"], top["selection_frequency"])
    plt.xlabel("Selection frequency")
    plt.ylabel("SNP")
    plt.title(f"{trait} - top {TOP_N} stable SNPs")
    plt.grid(axis="x", alpha=0.3)
    savefig(fig_dir, f"{trait}_top_stable_snps_selection_frequency.png")


def plot_regions_by_chromosome(region_annot, fig_dir: Path, table_dir: Path, trait: str):
    df = region_annot.copy()
    df = df.dropna(subset=["chromosome"])

    stable = df[df["selection_frequency"] >= REGION_FREQ_STABLE].copy()

    if stable.empty:
        print(f"[WARNING] {trait}: no stable regions for chromosome plot.")
        return

    chrom_counts = (
        stable
        .groupby("chromosome")
        .agg(
            n_stable_regions=("region", "nunique"),
            mean_selection_frequency=("selection_frequency", "mean"),
            max_selection_frequency=("selection_frequency", "max"),
        )
        .reset_index()
    )

    chrom_counts["chr_sort"] = chrom_counts["chromosome"].apply(natural_chr_sort_key)
    chrom_counts = chrom_counts.sort_values("chr_sort")

    chrom_counts.to_csv(
        table_dir / f"G4_stable_regions_by_chromosome_{trait}.csv",
        index=False
    )

    plt.figure(figsize=(9, 5))
    plt.bar(chrom_counts["chromosome"].astype(str), chrom_counts["n_stable_regions"])
    plt.xlabel("Chromosome")
    plt.ylabel(f"Number of stable regions\n(freq ≥ {REGION_FREQ_STABLE})")
    plt.title(f"{trait} - stable genomic regions by chromosome")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_stable_regions_by_chromosome.png")

    plt.figure(figsize=(9, 5))
    plt.bar(chrom_counts["chromosome"].astype(str), chrom_counts["max_selection_frequency"])
    plt.xlabel("Chromosome")
    plt.ylabel("Maximum selection frequency")
    plt.title(f"{trait} - maximum region stability by chromosome")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_max_region_frequency_by_chromosome.png")


def plot_region_genomic_positions(region_annot, fig_dir: Path, trait: str):
    df = region_annot.copy()
    df = df.dropna(subset=["chromosome", "region_midpoint"])
    df = df[df["selection_frequency"] >= REGION_FREQ_STABLE].copy()

    if df.empty:
        print(f"[WARNING] {trait}: no stable regions for genomic position plot.")
        return

    df["chr_sort"] = df["chromosome"].apply(natural_chr_sort_key)
    df = df.sort_values(["chr_sort", "region_midpoint"])

    plt.figure(figsize=(10, 5))
    plt.scatter(df["region_midpoint"], df["selection_frequency"])
    plt.xlabel("Genomic midpoint within chromosome")
    plt.ylabel("Selection frequency")
    plt.title(f"{trait} - stable region positions, frequency ≥ {REGION_FREQ_STABLE}")
    plt.grid(alpha=0.3)
    savefig(fig_dir, f"{trait}_stable_region_positions_all_chromosomes_unfaceted.png")

    for chrom, sub in df.groupby("chromosome"):
        plt.figure(figsize=(8, 4))
        plt.scatter(sub["region_midpoint"], sub["selection_frequency"])

        for _, row in sub.iterrows():
            if row["selection_frequency"] >= REGION_FREQ_HIGH_CONFIDENCE:
                plt.text(
                    row["region_midpoint"],
                    row["selection_frequency"],
                    row["region"],
                    fontsize=7
                )

        plt.xlabel(f"Position on chromosome {chrom}")
        plt.ylabel("Selection frequency")
        plt.title(f"{trait} - stable regions on chromosome {chrom}")
        plt.grid(alpha=0.3)
        savefig(fig_dir, f"{trait}_stable_region_positions_chr_{chrom}.png")


def plot_selection_frequency_distribution(region_annot, snp_annot, fig_dir: Path, trait: str):
    plt.figure(figsize=(7, 5))
    plt.hist(region_annot["selection_frequency"], bins=np.linspace(0, 1, 11))
    plt.xlabel("Region selection frequency")
    plt.ylabel("Number of regions")
    plt.title(f"{trait} - distribution of region selection frequencies")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_region_selection_frequency_distribution.png")

    plt.figure(figsize=(7, 5))
    plt.hist(snp_annot["selection_frequency"], bins=np.linspace(0, 1, 11))
    plt.xlabel("SNP selection frequency")
    plt.ylabel("Number of SNPs")
    plt.title(f"{trait} - distribution of SNP selection frequencies")
    plt.grid(axis="y", alpha=0.3)
    savefig(fig_dir, f"{trait}_snp_selection_frequency_distribution.png")


# =============================================================================
# GENE / ANNOTATION SUMMARY
# =============================================================================

def create_gene_annotation_summary(region_annot, table_dir: Path, trait: str):
    gene_cols = identify_gene_columns(region_annot)

    with open(table_dir / f"G4_detected_gene_annotation_columns_{trait}.json", "w") as f:
        json.dump({"gene_annotation_columns": gene_cols}, f, indent=4)

    if len(gene_cols) == 0:
        print(f"[WARNING] {trait}: no gene/annotation-like columns detected in annotated region table.")
        return None

    stable = region_annot[
        region_annot["selection_frequency"] >= REGION_FREQ_STABLE
    ].copy()

    selected_cols = [
        "region",
        "chromosome",
        "region_start",
        "region_end",
        "n_seeds_selected",
        "selection_frequency",
        "n_unique_snps_selected",
        "n_selected_snp_events",
        "seeds",
    ] + gene_cols

    selected_cols = [c for c in selected_cols if c in stable.columns]

    gene_summary = stable[selected_cols].copy()

    gene_summary.to_csv(
        table_dir / f"G4_stable_regions_with_gene_annotations_{trait}.csv",
        index=False
    )

    high_conf = region_annot[
        region_annot["selection_frequency"] >= REGION_FREQ_HIGH_CONFIDENCE
    ].copy()

    high_conf_summary = high_conf[
        [c for c in selected_cols if c in high_conf.columns]
    ].copy()

    high_conf_summary.to_csv(
        table_dir / f"G4_high_confidence_regions_with_gene_annotations_{trait}.csv",
        index=False
    )

    return gene_summary


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    trait: str,
    run_dir: Path,
    paths: dict,
    region_annot,
    snp_annot,
    stable_regions,
    high_conf_regions,
    stable_snps,
    annotation_file,
):
    report_path = run_dir / f"G4_region_interpretation_report_{trait}.txt"

    gene_cols = identify_gene_columns(region_annot)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"G4 REPORT - G2B REGION INTERPRETATION - {trait}\n")
        f.write("=" * 80 + "\n\n")

        f.write("INPUTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"TRAIT: {trait}\n")
        f.write(f"WINDOW_LABEL: {WINDOW_LABEL}\n")
        f.write(f"RUN_DIR: {run_dir}\n")
        f.write(f"REGION_STABILITY_FILE: {paths['region_stability_file']}\n")
        f.write(f"SNP_STABILITY_FILE: {paths['snp_stability_file']}\n")
        f.write(f"SNP_METADATA_FILE: {paths['snp_metadata_file']}\n")
        f.write(f"ANNOTATED_REGION_FILE: {annotation_file}\n\n")

        f.write("REGION STABILITY SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total selected regions: {region_annot.shape[0]}\n")
        f.write(f"Stable regions, freq >= {REGION_FREQ_STABLE}: {stable_regions.shape[0]}\n")
        f.write(f"High-confidence regions, freq >= {REGION_FREQ_HIGH_CONFIDENCE}: {high_conf_regions.shape[0]}\n")

        if "selection_frequency" in region_annot.columns and len(region_annot) > 0:
            f.write(f"Max region frequency: {region_annot['selection_frequency'].max():.3f}\n\n")
        else:
            f.write("Max region frequency: NA\n\n")

        f.write("SNP STABILITY SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total selected SNPs: {snp_annot.shape[0]}\n")
        f.write(f"Stable SNPs, freq >= {SNP_FREQ_STABLE}: {stable_snps.shape[0]}\n")

        if "selection_frequency" in snp_annot.columns and len(snp_annot) > 0:
            f.write(f"Max SNP frequency: {snp_annot['selection_frequency'].max():.3f}\n\n")
        else:
            f.write("Max SNP frequency: NA\n\n")

        f.write("DETECTED GENE/ANNOTATION COLUMNS\n")
        f.write("-" * 80 + "\n")

        if len(gene_cols) == 0:
            f.write("No gene/annotation-like columns detected.\n\n")
        else:
            for c in gene_cols:
                f.write(f"- {c}\n")
            f.write("\n")

        f.write("TOP 20 STABLE REGIONS\n")
        f.write("-" * 80 + "\n")

        top_cols = [
            "region", "chromosome", "region_start", "region_end",
            "n_seeds_selected", "selection_frequency",
            "n_unique_snps_selected", "n_selected_snp_events", "seeds"
        ]

        top_cols += gene_cols[:8]
        top_cols = [c for c in top_cols if c in region_annot.columns]

        if len(region_annot) > 0 and len(top_cols) > 0:
            f.write(region_annot.head(20)[top_cols].to_string(index=False))
        else:
            f.write("No region rows available.")
        f.write("\n\n")

        f.write("TOP 20 STABLE SNPs\n")
        f.write("-" * 80 + "\n")

        snp_top_cols = [
            "snp", "n_seeds_selected", "selection_frequency", "seeds"
        ]

        extra_snp_cols = [
            "region_id",
            "region",
            "CHROM",
            "POS",
            "genes_inside",
            "genes_nearby_10kb",
            "region_score",
            "rank_by_region_score",
        ]

        snp_top_cols += extra_snp_cols
        snp_top_cols = [c for c in snp_top_cols if c in snp_annot.columns]

        if len(snp_annot) > 0 and len(snp_top_cols) > 0:
            f.write(snp_annot.head(20)[snp_top_cols].to_string(index=False))
        else:
            f.write("No SNP rows available.")
        f.write("\n\n")

        f.write("INTERPRETATION NOTE\n")
        f.write("-" * 80 + "\n")
        f.write(
            "At the single-SNP level, selection can be variable across repeated splits because nearby "
            "markers may be correlated or partially interchangeable. Therefore, the main biological "
            "interpretation should focus on recurrent genomic regions rather than isolated SNPs. "
            "Regions selected in many seeds are more robust candidates for downstream biological inspection.\n"
        )

    print(f"[REPORT] Saved: {report_path}")


# =============================================================================
# ONE TRAIT
# =============================================================================

def run_one_trait(trait: str):
    print("\n" + "#" * 80)
    print(f"G4 - ANNOTATE AND INTERPRET G2B STABLE REGIONS - {trait}")
    print("#" * 80)

    paths = get_trait_paths(trait)

    run_dir = paths["run_dir"]
    fig_dir = paths["fig_dir"]
    table_dir = paths["table_dir"]

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found for {trait}:\n{run_dir}")

    (
        region_stability,
        snp_stability,
        selected_regions_long,
        selected_snps_long,
        gene_stability,
        snp_metadata,
        annotated_regions,
        annotation_file,
    ) = load_data_for_trait(trait, paths)

    region_annot = build_annotated_region_stability(region_stability, annotated_regions)
    snp_annot = build_annotated_snp_stability(snp_stability, snp_metadata)

    region_annot = region_annot.sort_values(
        ["n_seeds_selected", "n_unique_snps_selected", "n_selected_snp_events"],
        ascending=[False, False, False]
    )

    if "snp" not in snp_annot.columns:
        snp_col = identify_snp_column(snp_annot)
        if snp_col is not None:
            snp_annot = snp_annot.rename(columns={snp_col: "snp"})

    snp_annot = snp_annot.sort_values(
        ["n_seeds_selected", "snp"],
        ascending=[False, True]
    )

    stable_regions, high_conf_regions, stable_snps = save_filtered_tables(
        region_annot=region_annot,
        snp_annot=snp_annot,
        table_dir=table_dir,
        trait=trait,
    )

    plot_top_regions(region_annot, fig_dir, trait)
    plot_top_snps(snp_annot, fig_dir, trait)
    plot_regions_by_chromosome(region_annot, fig_dir, table_dir, trait)
    plot_region_genomic_positions(region_annot, fig_dir, trait)
    plot_selection_frequency_distribution(region_annot, snp_annot, fig_dir, trait)

    create_gene_annotation_summary(region_annot, table_dir, trait)

    write_report(
        trait=trait,
        run_dir=run_dir,
        paths=paths,
        region_annot=region_annot,
        snp_annot=snp_annot,
        stable_regions=stable_regions,
        high_conf_regions=high_conf_regions,
        stable_snps=stable_snps,
        annotation_file=annotation_file,
    )

    print("\n" + "#" * 80)
    print(f"G4 FINISHED FOR {trait}")
    print("#" * 80)
    print(f"Figures saved in:\n  {fig_dir}")
    print(f"Tables saved in:\n  {table_dir}")

    summary = {
        "Trait": trait,
        "n_selected_regions": region_annot.shape[0],
        "n_stable_regions_freq_ge_0_5": stable_regions.shape[0],
        "n_high_conf_regions_freq_ge_0_7": high_conf_regions.shape[0],
        "max_region_frequency": region_annot["selection_frequency"].max() if len(region_annot) > 0 else np.nan,
        "n_selected_snps": snp_annot.shape[0],
        "n_stable_snps_freq_ge_0_3": stable_snps.shape[0],
        "max_snp_frequency": snp_annot["selection_frequency"].max() if len(snp_annot) > 0 else np.nan,
        "annotation_file": str(annotation_file),
        "fig_dir": str(fig_dir),
        "table_dir": str(table_dir),
    }

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 80)
    print("G4 - ANNOTATE AND INTERPRET G2B NEW TRAITS")
    print("#" * 80)

    all_summaries = []

    for trait in TRAITS:
        summary = run_one_trait(trait)
        all_summaries.append(summary)

    summary_all = pd.DataFrame(all_summaries)
    summary_file = BASE_RUN_DIR / "G4_region_interpretation_summary_all_traits.csv"
    summary_all.to_csv(summary_file, index=False)

    print("\n" + "=" * 80)
    print("G4 ALL TRAITS FINISHED")
    print("=" * 80)
    print(summary_all.to_string(index=False))
    print("\nSaved all-trait summary:")
    print(summary_file)


if __name__ == "__main__":
    main()