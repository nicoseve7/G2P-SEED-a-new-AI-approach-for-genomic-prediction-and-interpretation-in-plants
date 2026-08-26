# -*- coding: utf-8 -*-

################################################################################
### G4_annotate_and_interpret_G2B_regions.py
###
### Region-level biological interpretation after:
###   G2B_multiseed_ga_variable_split_inner3cv.py
###
### Produces:
###   - annotated stable region tables
###   - high-confidence region subsets
###   - stable SNP annotation table
###   - region stability plots
###   - chromosome distribution plots
###   - final interpretation report
###
### Input:
###   Output/04_ga_runs/G2B_multiseed_variable_split_inner3cv/
###   Output/01_regioni_annotate/
###   Output/03_ga_inputs/snp_metadata_top1000_regions_50kb_Harvest_date.csv
###
### Output:
###   Output/04_ga_runs/G2B_multiseed_variable_split_inner3cv/figures_G4/
###   Output/04_ga_runs/G2B_multiseed_variable_split_inner3cv/tables_G4/
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

TRAIT = "Harvest_date"
WINDOW_LABEL = "50kb"

GA_ROOT = (
    Path("02_harvest_date")
    / "09_genetic_algorithm"
)

RUN_DIR = (
    GA_ROOT
    / "output"
    / "04_ga_runs"
    / "G2B_no_soil_multiseed_variable_split_inner3cv"
)

GA_INPUT_DIR = (
    GA_ROOT
    / "output"
    / "03_ga_inputs"
)

ANNOTATION_DIR = (
    GA_ROOT
    / "output"
    / "01_region_annotations"
)

REGION_STABILITY_FILE = RUN_DIR / "G2B_region_stability_across_seeds.csv"
SNP_STABILITY_FILE = RUN_DIR / "G2B_snp_stability_across_seeds.csv"
SELECTED_REGIONS_LONG_FILE = RUN_DIR / "G2B_selected_regions_all_seeds_long.csv"
SELECTED_SNPS_LONG_FILE = RUN_DIR / "G2B_selected_snps_all_seeds_long.csv"

SNP_METADATA_FILE = GA_INPUT_DIR / "snp_metadata_top1000_regions_50kb_Harvest_date.csv"

# If this exact file does not exist, the script searches automatically.
ANNOTATED_REGION_FILE = ANNOTATION_DIR / "region_summary_annotated_50kb_Harvest_date.csv"

FIG_DIR = RUN_DIR / "figures_G4"
TABLE_DIR = RUN_DIR / "tables_G4"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
TOP_N = 25

REGION_FREQ_HIGH_CONFIDENCE = 0.7
REGION_FREQ_STABLE = 0.5
SNP_FREQ_STABLE = 0.3


# =============================================================================
# UTILS
# =============================================================================

def savefig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")


def read_required(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
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
        3:30650001-30700000
    Returns chromosome, start, end.
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
    return f"{normalize_chr(chrom)}:{int(start)}-{int(end)}"


def find_annotated_region_file():
    if ANNOTATED_REGION_FILE.exists():
        return ANNOTATED_REGION_FILE

    candidates = []

    patterns = [
        f"*annotated*{WINDOW_LABEL}*{TRAIT}*.csv",
        f"*region_summary_annotated*{WINDOW_LABEL}*.csv",
        f"*annotated*50kb*.csv",
    ]

    if ANNOTATION_DIR.exists():
        for pat in patterns:
            candidates.extend(list(ANNOTATION_DIR.rglob(pat)))

    candidates = sorted(set(candidates))

    if len(candidates) == 0:
        print("[WARNING] No annotated region file found automatically.")
        return None

    print("[INFO] Annotated region file candidates:")
    for i, c in enumerate(candidates[:10]):
        print(f"  {i}: {c}")

    chosen = candidates[0]
    print(f"[INFO] Using annotated region file:\n  {chosen}")

    return chosen


def add_region_components(df, region_col="region"):
    out = df.copy()

    parsed = out[region_col].apply(parse_region)
    out["chromosome"] = [p[0] for p in parsed]
    out["region_start"] = [p[1] for p in parsed]
    out["region_end"] = [p[2] for p in parsed]
    out["region_midpoint"] = (out["region_start"] + out["region_end"]) / 2

    return out


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

    # Try to reconstruct region column from chromosome/start/end.
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
        return annot

    print("[WARNING] Could not detect or reconstruct region column in annotation file.")
    print(f"[WARNING] Annotation columns: {list(annot.columns)}")
    return annot


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


# =============================================================================
# LOAD AND MERGE
# =============================================================================

def load_data():
    print("\n" + "=" * 80)
    print("[LOAD] Loading G2B region/SNP stability outputs")
    print("=" * 80)

    region_stability = read_required(REGION_STABILITY_FILE)
    snp_stability = read_required(SNP_STABILITY_FILE)

    selected_regions_long = None
    selected_snps_long = None

    if SELECTED_REGIONS_LONG_FILE.exists():
        selected_regions_long = pd.read_csv(SELECTED_REGIONS_LONG_FILE)

    if SELECTED_SNPS_LONG_FILE.exists():
        selected_snps_long = pd.read_csv(SELECTED_SNPS_LONG_FILE)

    snp_metadata = None
    if SNP_METADATA_FILE.exists():
        snp_metadata = pd.read_csv(SNP_METADATA_FILE)
    else:
        print(f"[WARNING] SNP metadata file not found: {SNP_METADATA_FILE}")

    annotation_file = find_annotated_region_file()
    annotated_regions = prepare_annotated_regions(annotation_file)

    print(f"[INFO] Region stability shape: {region_stability.shape}")
    print(f"[INFO] SNP stability shape: {snp_stability.shape}")

    if annotated_regions is not None:
        print(f"[INFO] Annotated regions shape: {annotated_regions.shape}")
    else:
        print("[WARNING] No annotated region table available.")

    if snp_metadata is not None:
        print(f"[INFO] SNP metadata shape: {snp_metadata.shape}")

    return region_stability, snp_stability, selected_regions_long, selected_snps_long, snp_metadata, annotated_regions, annotation_file


def build_annotated_region_stability(region_stability, annotated_regions):
    region_stability = region_stability.copy()
    region_stability["region"] = region_stability["region"].astype(str).str.strip()

    region_stability = add_region_components(region_stability, region_col="region")

    if annotated_regions is None or "region" not in annotated_regions.columns:
        print("[WARNING] Cannot merge region stability with annotations.")
        return region_stability

    annotated_regions = annotated_regions.copy()
    annotated_regions["region"] = annotated_regions["region"].astype(str).str.strip()

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
    snp_stability["snp"] = snp_stability["snp"].astype(str).str.strip()

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

def save_filtered_tables(region_annot, snp_annot):
    region_annot = region_annot.sort_values(
        ["n_seeds_selected", "n_unique_snps_selected", "n_selected_snp_events"],
        ascending=[False, False, False]
    )

    region_annot.to_csv(TABLE_DIR / "G4_region_stability_annotated_all.csv", index=False)

    stable_regions = region_annot[
        region_annot["selection_frequency"] >= REGION_FREQ_STABLE
    ].copy()

    high_conf_regions = region_annot[
        region_annot["selection_frequency"] >= REGION_FREQ_HIGH_CONFIDENCE
    ].copy()

    stable_regions.to_csv(
        TABLE_DIR / f"G4_stable_regions_freq_ge_{str(REGION_FREQ_STABLE).replace('.', '')}.csv",
        index=False
    )

    high_conf_regions.to_csv(
        TABLE_DIR / f"G4_high_confidence_regions_freq_ge_{str(REGION_FREQ_HIGH_CONFIDENCE).replace('.', '')}.csv",
        index=False
    )

    snp_annot = snp_annot.sort_values(
        ["n_seeds_selected", "snp"],
        ascending=[False, True]
    )

    snp_annot.to_csv(TABLE_DIR / "G4_snp_stability_annotated_all.csv", index=False)

    stable_snps = snp_annot[
        snp_annot["selection_frequency"] >= SNP_FREQ_STABLE
    ].copy()

    stable_snps.to_csv(
        TABLE_DIR / f"G4_stable_snps_freq_ge_{str(SNP_FREQ_STABLE).replace('.', '')}.csv",
        index=False
    )

    # BED-like file for regions
    bed_df = region_annot[["chromosome", "region_start", "region_end", "region", "selection_frequency"]].copy()
    bed_df = bed_df.dropna(subset=["chromosome", "region_start", "region_end"])
    bed_df["region_start_0based"] = bed_df["region_start"].astype(int) - 1

    bed_out = bed_df[
        ["chromosome", "region_start_0based", "region_end", "region", "selection_frequency"]
    ].copy()

    bed_out.to_csv(
        TABLE_DIR / "G4_region_stability_annotated_all.bed",
        sep="\t",
        index=False,
        header=False
    )

    return stable_regions, high_conf_regions, stable_snps


# =============================================================================
# PLOTS
# =============================================================================

def plot_top_regions(region_annot):
    top = region_annot.head(TOP_N).copy()
    top = top.iloc[::-1]

    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    plt.barh(top["region"], top["selection_frequency"])
    plt.xlabel("Selection frequency")
    plt.ylabel("Region")
    plt.title(f"Top {TOP_N} stable genomic regions")
    plt.grid(axis="x", alpha=0.3)
    savefig("top_stable_regions_selection_frequency.png")

    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    plt.barh(top["region"], top["n_unique_snps_selected"])
    plt.xlabel("Number of unique selected SNPs")
    plt.ylabel("Region")
    plt.title(f"Unique selected SNPs in top {TOP_N} regions")
    plt.grid(axis="x", alpha=0.3)
    savefig("top_stable_regions_unique_snps.png")

    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    plt.barh(top["region"], top["n_selected_snp_events"])
    plt.xlabel("Selected SNP events across seeds")
    plt.ylabel("Region")
    plt.title(f"SNP selection events in top {TOP_N} regions")
    plt.grid(axis="x", alpha=0.3)
    savefig("top_stable_regions_snp_events.png")


def plot_top_snps(snp_annot):
    top = snp_annot.head(TOP_N).copy()
    top = top.iloc[::-1]

    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    plt.barh(top["snp"], top["selection_frequency"])
    plt.xlabel("Selection frequency")
    plt.ylabel("SNP")
    plt.title(f"Top {TOP_N} stable SNPs")
    plt.grid(axis="x", alpha=0.3)
    savefig("top_stable_snps_selection_frequency.png")


def plot_regions_by_chromosome(region_annot):
    df = region_annot.copy()
    df = df.dropna(subset=["chromosome"])

    # Keep only stable regions for chromosome summaries
    stable = df[df["selection_frequency"] >= REGION_FREQ_STABLE].copy()

    if stable.empty:
        print("[WARNING] No stable regions for chromosome plot.")
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

    # Natural-ish sorting
    def chr_sort_key(x):
        try:
            return int(x)
        except Exception:
            return 999

    chrom_counts["chr_sort"] = chrom_counts["chromosome"].apply(chr_sort_key)
    chrom_counts = chrom_counts.sort_values("chr_sort")

    chrom_counts.to_csv(TABLE_DIR / "G4_stable_regions_by_chromosome.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(chrom_counts["chromosome"].astype(str), chrom_counts["n_stable_regions"])
    plt.xlabel("Chromosome")
    plt.ylabel(f"Number of stable regions\n(freq ≥ {REGION_FREQ_STABLE})")
    plt.title("Stable genomic regions by chromosome")
    plt.grid(axis="y", alpha=0.3)
    savefig("stable_regions_by_chromosome.png")

    plt.figure(figsize=(9, 5))
    plt.bar(chrom_counts["chromosome"].astype(str), chrom_counts["max_selection_frequency"])
    plt.xlabel("Chromosome")
    plt.ylabel("Maximum selection frequency")
    plt.title("Maximum region stability by chromosome")
    plt.grid(axis="y", alpha=0.3)
    savefig("max_region_frequency_by_chromosome.png")


def plot_region_genomic_positions(region_annot):
    df = region_annot.copy()
    df = df.dropna(subset=["chromosome", "region_midpoint"])
    df = df[df["selection_frequency"] >= REGION_FREQ_STABLE].copy()

    if df.empty:
        print("[WARNING] No stable regions for genomic position plot.")
        return

    def chr_sort_key(x):
        try:
            return int(x)
        except Exception:
            return 999

    df["chr_sort"] = df["chromosome"].apply(chr_sort_key)
    df = df.sort_values(["chr_sort", "region_midpoint"])

    plt.figure(figsize=(10, 5))
    plt.scatter(df["region_midpoint"], df["selection_frequency"])
    plt.xlabel("Genomic midpoint within chromosome")
    plt.ylabel("Selection frequency")
    plt.title(f"Stable region positions, frequency ≥ {REGION_FREQ_STABLE}")
    plt.grid(alpha=0.3)
    savefig("stable_region_positions_all_chromosomes_unfaceted.png")

    # One figure per chromosome for crowded cases
    for chrom, sub in df.groupby("chromosome"):
        plt.figure(figsize=(8, 4))
        plt.scatter(sub["region_midpoint"], sub["selection_frequency"])
        for _, row in sub.iterrows():
            if row["selection_frequency"] >= REGION_FREQ_HIGH_CONFIDENCE:
                plt.text(row["region_midpoint"], row["selection_frequency"], row["region"], fontsize=7)
        plt.xlabel(f"Position on chromosome {chrom}")
        plt.ylabel("Selection frequency")
        plt.title(f"Stable regions on chromosome {chrom}")
        plt.grid(alpha=0.3)
        savefig(f"stable_region_positions_chr_{chrom}.png")


def plot_selection_frequency_distribution(region_annot, snp_annot):
    plt.figure(figsize=(7, 5))
    plt.hist(region_annot["selection_frequency"], bins=np.linspace(0, 1, 11))
    plt.xlabel("Region selection frequency")
    plt.ylabel("Number of regions")
    plt.title("Distribution of region selection frequencies")
    plt.grid(axis="y", alpha=0.3)
    savefig("region_selection_frequency_distribution.png")

    plt.figure(figsize=(7, 5))
    plt.hist(snp_annot["selection_frequency"], bins=np.linspace(0, 1, 11))
    plt.xlabel("SNP selection frequency")
    plt.ylabel("Number of SNPs")
    plt.title("Distribution of SNP selection frequencies")
    plt.grid(axis="y", alpha=0.3)
    savefig("snp_selection_frequency_distribution.png")


# =============================================================================
# GENE / ANNOTATION SUMMARY
# =============================================================================

def create_gene_annotation_summary(region_annot):
    gene_cols = identify_gene_columns(region_annot)

    with open(TABLE_DIR / "G4_detected_gene_annotation_columns.json", "w") as f:
        json.dump({"gene_annotation_columns": gene_cols}, f, indent=4)

    if len(gene_cols) == 0:
        print("[WARNING] No gene/annotation-like columns detected in annotated region table.")
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
    gene_summary.to_csv(TABLE_DIR / "G4_stable_regions_with_gene_annotations.csv", index=False)

    high_conf = region_annot[
        region_annot["selection_frequency"] >= REGION_FREQ_HIGH_CONFIDENCE
    ].copy()

    high_conf_summary = high_conf[[c for c in selected_cols if c in high_conf.columns]].copy()
    high_conf_summary.to_csv(TABLE_DIR / "G4_high_confidence_regions_with_gene_annotations.csv", index=False)

    return gene_summary


# =============================================================================
# REPORT
# =============================================================================

def write_report(region_annot, snp_annot, stable_regions, high_conf_regions, stable_snps, annotation_file):
    report_path = RUN_DIR / "G4_region_interpretation_report.txt"

    gene_cols = identify_gene_columns(region_annot)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("G4 REPORT - G2B NO-SOIL REGION INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("INPUTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"RUN_DIR: {RUN_DIR}\n")
        f.write(f"REGION_STABILITY_FILE: {REGION_STABILITY_FILE}\n")
        f.write(f"SNP_STABILITY_FILE: {SNP_STABILITY_FILE}\n")
        f.write(f"SNP_METADATA_FILE: {SNP_METADATA_FILE}\n")
        f.write(f"ANNOTATED_REGION_FILE: {annotation_file}\n\n")

        f.write("REGION STABILITY SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total selected regions: {region_annot.shape[0]}\n")
        f.write(f"Stable regions, freq >= {REGION_FREQ_STABLE}: {stable_regions.shape[0]}\n")
        f.write(f"High-confidence regions, freq >= {REGION_FREQ_HIGH_CONFIDENCE}: {high_conf_regions.shape[0]}\n")
        f.write(f"Max region frequency: {region_annot['selection_frequency'].max():.3f}\n\n")

        f.write("SNP STABILITY SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total selected SNPs: {snp_annot.shape[0]}\n")
        f.write(f"Stable SNPs, freq >= {SNP_FREQ_STABLE}: {stable_snps.shape[0]}\n")
        f.write(f"Max SNP frequency: {snp_annot['selection_frequency'].max():.3f}\n\n")

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
        top_cols += gene_cols[:5]
        top_cols = [c for c in top_cols if c in region_annot.columns]

        f.write(region_annot.head(20)[top_cols].to_string(index=False))
        f.write("\n\n")

        f.write("TOP 20 STABLE SNPs\n")
        f.write("-" * 80 + "\n")
        snp_top_cols = [
            "snp", "n_seeds_selected", "selection_frequency", "seeds"
        ]
        snp_top_cols = [c for c in snp_top_cols if c in snp_annot.columns]
        f.write(snp_annot.head(20)[snp_top_cols].to_string(index=False))
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
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 80)
    print("G4 - ANNOTATE AND INTERPRET G2B NO-SOIL STABLE REGIONS")
    print("#" * 80)

    (
        region_stability,
        snp_stability,
        selected_regions_long,
        selected_snps_long,
        snp_metadata,
        annotated_regions,
        annotation_file
    ) = load_data()

    region_annot = build_annotated_region_stability(region_stability, annotated_regions)
    snp_annot = build_annotated_snp_stability(snp_stability, snp_metadata)

    region_annot = region_annot.sort_values(
        ["n_seeds_selected", "n_unique_snps_selected", "n_selected_snp_events"],
        ascending=[False, False, False]
    )

    snp_annot = snp_annot.sort_values(
        ["n_seeds_selected", "snp"],
        ascending=[False, True]
    )

    stable_regions, high_conf_regions, stable_snps = save_filtered_tables(region_annot, snp_annot)

    plot_top_regions(region_annot)
    plot_top_snps(snp_annot)
    plot_regions_by_chromosome(region_annot)
    plot_region_genomic_positions(region_annot)
    plot_selection_frequency_distribution(region_annot, snp_annot)

    create_gene_annotation_summary(region_annot)

    write_report(
        region_annot=region_annot,
        snp_annot=snp_annot,
        stable_regions=stable_regions,
        high_conf_regions=high_conf_regions,
        stable_snps=stable_snps,
        annotation_file=annotation_file,
    )

    print("\n" + "#" * 80)
    print("G4 FINISHED")
    print("#" * 80)
    print(f"Figures saved in:\n  {FIG_DIR}")
    print(f"Tables saved in:\n  {TABLE_DIR}")


if __name__ == "__main__":
    main()
