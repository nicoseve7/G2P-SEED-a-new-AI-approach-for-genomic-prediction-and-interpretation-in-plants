# -*- coding: utf-8 -*-

################################################################################
### G6_add_genes_to_region_stability_from_selected_snps.py
###
### Adds gene annotations to:
###   G2B_region_stability_across_seeds_with_unique_snps.csv
###
### For each region row, the script:
###   1. reads unique_snps_selected
###   2. gets SNP coordinates from snp_metadata_top1000_regions_50kb_Harvest_date.csv
###   3. maps each SNP to genes from GFF3 using +/- 10 kb tolerance
###   4. saves a new region stability file with gene columns
###
### Run from:
###   dalpaper/regioni_ga_harvest_no_soil/
################################################################################

from pathlib import Path
import re
import pandas as pd
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

TRAIT = "Harvest_date"
TOLERANCE_BP = 10_000

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

REGION_STABILITY_FILE = RUN_DIR / "G2B_region_stability_across_seeds_with_unique_snps.csv"
SNP_METADATA_FILE = GA_INPUT_DIR / "snp_metadata_top1000_regions_50kb_Harvest_date.csv"

# Cambia questo path se il GFF3 è in un'altra posizione.
# Possibili alternative:
# GFF3_FILE = Path("../Output/gene_models_20170612.gff3")
# GFF3_FILE = Path("../Input/gene_models_20170612.gff3")
# GFF3_FILE = Path("Input/base_files/gene_models_20170612.gff3")
GFF3_FILE = Path("Input/base_files/gene_models_20170612.gff3")

OUT_FILE = RUN_DIR / "G2B_region_stability_across_seeds_with_unique_snps_and_genes_10kb.csv"
SNP_GENE_MAP_OUT = RUN_DIR / "G2B_selected_snp_to_gene_map_10kb.csv"
REPORT_FILE = RUN_DIR / "G6_add_genes_to_region_stability_report.txt"


# =============================================================================
# HELPERS
# =============================================================================

def check_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} non trovato:\n{path}\n"
            f"Controlla il path nella sezione SETTINGS."
        )


def detect_column(df: pd.DataFrame, candidates, required=True, label="column"):
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if required:
        raise ValueError(
            f"Non trovo {label}. Ho provato: {candidates}\n"
            f"Colonne disponibili: {df.columns.tolist()}"
        )

    return None


def normalize_chr(x):
    """
    Normalizza i cromosomi per confrontare VCF/SNP metadata e GFF3.

    Esempi:
      chr9 -> 9
      Chr09 -> 9
      GDDH13_09 -> 9
      09 -> 9
    """
    x = str(x).strip()

    x = x.replace("chr", "")
    x = x.replace("Chr", "")
    x = x.replace("CHR", "")
    x = x.replace("GDDH13_", "")
    x = x.replace("gddh13_", "")

    x = x.strip()

    # Se è numerico, rimuove zeri iniziali: 09 -> 9
    if re.fullmatch(r"\d+", x):
        x = str(int(x))

    return x


def split_snp_list(x):
    """
    Split robusto della colonna unique_snps_selected.
    Nel tuo file gli SNP sono separati da ;.
    """
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return []

    parts = re.split(r"[;,\| ]+", x)
    parts = [p.strip() for p in parts if p.strip() != ""]

    return parts


def parse_gff3_attributes(attr: str):
    """
    Converte la colonna attributes del GFF3 in dizionario.
    Esempio:
      ID=gene:MD01G1000100;Name=...
    """
    out = {}

    if pd.isna(attr):
        return out

    for item in str(attr).split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()

    return out


def clean_gene_id(gene_id: str):
    """
    Pulisce gene:MD... -> MD...
    """
    gene_id = str(gene_id).strip()

    if gene_id.startswith("gene:"):
        gene_id = gene_id.replace("gene:", "", 1)

    return gene_id


# =============================================================================
# LOAD INPUTS
# =============================================================================

def load_region_stability():
    check_file(REGION_STABILITY_FILE, "REGION_STABILITY_FILE")

    df = pd.read_csv(REGION_STABILITY_FILE)

    if "unique_snps_selected" not in df.columns:
        raise ValueError(
            f"Nel file region stability manca la colonna unique_snps_selected.\n"
            f"Colonne trovate: {df.columns.tolist()}"
        )

    return df


def load_snp_metadata():
    check_file(SNP_METADATA_FILE, "SNP_METADATA_FILE")

    df = pd.read_csv(SNP_METADATA_FILE)

    snp_col = detect_column(
        df,
        ["SNP", "snp", "SNP_ID", "snp_id", "marker", "Marker", "id", "ID"],
        required=True,
        label="colonna SNP"
    )

    chrom_col = detect_column(
        df,
        ["CHROM", "chrom", "Chromosome", "chromosome", "chr", "Chr"],
        required=True,
        label="colonna cromosoma SNP"
    )

    pos_col = detect_column(
        df,
        ["POS", "pos", "Position", "position", "bp", "BP"],
        required=True,
        label="colonna posizione SNP"
    )

    out = df[[snp_col, chrom_col, pos_col]].copy()
    out = out.rename(columns={
        snp_col: "SNP",
        chrom_col: "CHROM",
        pos_col: "POS",
    })

    out["SNP"] = out["SNP"].astype(str).str.strip()
    out["CHROM_raw"] = out["CHROM"].astype(str).str.strip()
    out["CHROM_norm"] = out["CHROM_raw"].apply(normalize_chr)
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")

    out = out.dropna(subset=["SNP", "CHROM_norm", "POS"]).copy()
    out["POS"] = out["POS"].astype(int)

    out = out.drop_duplicates(subset=["SNP"]).reset_index(drop=True)

    return out


def load_gff3_genes():
    check_file(GFF3_FILE, "GFF3_FILE")

    cols = [
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    ]

    gff = pd.read_csv(
        GFF3_FILE,
        sep="\t",
        comment="#",
        header=None,
        names=cols,
        dtype=str
    )

    # Usiamo solo feature di tipo gene.
    gff = gff[gff["type"].astype(str).str.lower() == "gene"].copy()

    if gff.empty:
        raise ValueError(
            "Nel GFF3 non ho trovato righe con type == 'gene'. "
            "Controlla se il file usa un altro nome per le feature geniche."
        )

    gff["start"] = pd.to_numeric(gff["start"], errors="coerce")
    gff["end"] = pd.to_numeric(gff["end"], errors="coerce")

    gff = gff.dropna(subset=["seqid", "start", "end"]).copy()
    gff["start"] = gff["start"].astype(int)
    gff["end"] = gff["end"].astype(int)

    gff["CHROM_raw"] = gff["seqid"].astype(str).str.strip()
    gff["CHROM_norm"] = gff["CHROM_raw"].apply(normalize_chr)

    attrs = gff["attributes"].apply(parse_gff3_attributes)

    # Prima prova ID, poi Name.
    gff["Gene"] = attrs.apply(lambda d: d.get("ID", d.get("Name", "")))
    gff["Gene"] = gff["Gene"].apply(clean_gene_id)

    gff = gff[gff["Gene"].astype(str).str.strip() != ""].copy()

    gff = gff[[
        "CHROM_norm",
        "start",
        "end",
        "strand",
        "Gene",
    ]].copy()

    gff = gff.sort_values(["CHROM_norm", "start", "end"]).reset_index(drop=True)

    return gff


# =============================================================================
# SNP -> GENE MAPPING
# =============================================================================

def build_snp_gene_map(selected_snps, snp_meta, genes_gff, tolerance_bp):
    """
    Per ogni SNP selezionato:
      gene_start - tolerance <= SNP_POS <= gene_end + tolerance

    Ritorna una tabella lunga:
      SNP, CHROM, POS, Gene, gene_start, gene_end, distance_to_gene
    """
    selected_snps = sorted(set(str(s).strip() for s in selected_snps if str(s).strip() != ""))

    snp_sub = snp_meta[snp_meta["SNP"].isin(selected_snps)].copy()

    missing_snps = sorted(set(selected_snps) - set(snp_sub["SNP"]))

    rows = []

    genes_by_chr = {
        chrom: sub.copy()
        for chrom, sub in genes_gff.groupby("CHROM_norm")
    }

    for _, snp_row in snp_sub.iterrows():
        snp = snp_row["SNP"]
        chrom = snp_row["CHROM_norm"]
        pos = int(snp_row["POS"])

        genes_chr = genes_by_chr.get(chrom)

        if genes_chr is None or genes_chr.empty:
            continue

        hits = genes_chr[
            (genes_chr["start"] - tolerance_bp <= pos) &
            (genes_chr["end"] + tolerance_bp >= pos)
        ].copy()

        if hits.empty:
            continue

        for _, g in hits.iterrows():
            gene_start = int(g["start"])
            gene_end = int(g["end"])

            if gene_start <= pos <= gene_end:
                relation = "inside_gene"
                distance = 0
            elif pos < gene_start:
                relation = "upstream_or_before_gene"
                distance = gene_start - pos
            else:
                relation = "downstream_or_after_gene"
                distance = pos - gene_end

            rows.append({
                "SNP": snp,
                "CHROM": chrom,
                "POS": pos,
                "Gene": g["Gene"],
                "gene_start": gene_start,
                "gene_end": gene_end,
                "strand": g["strand"],
                "relation": relation,
                "distance_to_gene_bp": distance,
            })

    snp_gene = pd.DataFrame(rows)

    if snp_gene.empty:
        snp_gene = pd.DataFrame(columns=[
            "SNP",
            "CHROM",
            "POS",
            "Gene",
            "gene_start",
            "gene_end",
            "strand",
            "relation",
            "distance_to_gene_bp",
        ])

    return snp_gene, missing_snps


def add_genes_to_region_rows(region_df, snp_gene_map):
    out = region_df.copy()

    snp_to_genes = (
        snp_gene_map
        .groupby("SNP")["Gene"]
        .apply(lambda x: sorted(set(x.astype(str))))
        .to_dict()
    )

    def genes_for_row(snp_cell):
        snps = split_snp_list(snp_cell)

        genes = []
        pairs = []

        for snp in snps:
            gs = snp_to_genes.get(snp, [])

            for g in gs:
                genes.append(g)
                pairs.append(f"{snp}:{g}")

        genes_unique = sorted(set(genes))
        pairs_unique = sorted(set(pairs))

        return pd.Series({
            "n_genes_from_selected_snps_10kb": len(genes_unique),
            "genes_from_selected_snps_10kb": ";".join(genes_unique),
            "snp_gene_pairs_10kb": ";".join(pairs_unique),
        })

    added = out["unique_snps_selected"].apply(genes_for_row)

    out = pd.concat([out, added], axis=1)

    # Mettiamo le nuove colonne vicino a unique_snps_selected.
    preferred = [
        "region",
        "n_seeds_selected",
        "n_selected_snp_events",
        "n_unique_snps_selected",
        "n_unique_snps_selected_recomputed",
        "unique_snps_selected",
        "n_genes_from_selected_snps_10kb",
        "genes_from_selected_snps_10kb",
        "snp_gene_pairs_10kb",
        "seeds",
        "seeds_recomputed",
        "selection_frequency",
    ]

    remaining = [c for c in out.columns if c not in preferred]
    ordered = [c for c in preferred if c in out.columns] + remaining

    out = out[ordered]

    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("G6 - ADD GENES TO REGION STABILITY FROM SELECTED SNPs")
    print("=" * 80)

    print("\nLoading region stability...")
    region_df = load_region_stability()
    print(f"Region stability shape: {region_df.shape}")

    print("\nLoading SNP metadata...")
    snp_meta = load_snp_metadata()
    print(f"SNP metadata shape: {snp_meta.shape}")

    print("\nLoading GFF3 genes...")
    genes_gff = load_gff3_genes()
    print(f"GFF3 genes shape: {genes_gff.shape}")
    print(f"Unique GFF chromosomes: {genes_gff['CHROM_norm'].nunique()}")

    all_selected_snps = []

    for x in region_df["unique_snps_selected"]:
        all_selected_snps.extend(split_snp_list(x))

    all_selected_snps = sorted(set(all_selected_snps))

    print(f"\nUnique selected SNPs in region stability file: {len(all_selected_snps)}")

    print("\nBuilding SNP -> gene map with 10 kb tolerance...")
    snp_gene_map, missing_snps = build_snp_gene_map(
        selected_snps=all_selected_snps,
        snp_meta=snp_meta,
        genes_gff=genes_gff,
        tolerance_bp=TOLERANCE_BP,
    )

    print(f"SNP-gene mapping rows: {snp_gene_map.shape[0]}")
    print(f"SNPs with at least one mapped gene: {snp_gene_map['SNP'].nunique() if len(snp_gene_map) > 0 else 0}")
    print(f"Missing selected SNPs in SNP metadata: {len(missing_snps)}")

    print("\nAdding gene columns to region stability table...")
    out = add_genes_to_region_rows(region_df, snp_gene_map)

    out.to_csv(OUT_FILE, index=False)
    snp_gene_map.to_csv(SNP_GENE_MAP_OUT, index=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== G6 ADD GENES TO REGION STABILITY REPORT ===\n\n")
        f.write(f"TRAIT: {TRAIT}\n")
        f.write(f"TOLERANCE_BP: {TOLERANCE_BP}\n\n")

        f.write("Input files:\n")
        f.write(f"REGION_STABILITY_FILE: {REGION_STABILITY_FILE}\n")
        f.write(f"SNP_METADATA_FILE: {SNP_METADATA_FILE}\n")
        f.write(f"GFF3_FILE: {GFF3_FILE}\n\n")

        f.write("Output files:\n")
        f.write(f"OUT_FILE: {OUT_FILE}\n")
        f.write(f"SNP_GENE_MAP_OUT: {SNP_GENE_MAP_OUT}\n\n")

        f.write("Counts:\n")
        f.write(f"Region rows: {region_df.shape[0]}\n")
        f.write(f"Unique selected SNPs: {len(all_selected_snps)}\n")
        f.write(f"SNP metadata rows: {snp_meta.shape[0]}\n")
        f.write(f"GFF3 genes: {genes_gff.shape[0]}\n")
        f.write(f"SNP-gene mapping rows: {snp_gene_map.shape[0]}\n")
        f.write(f"SNPs with at least one mapped gene: {snp_gene_map['SNP'].nunique() if len(snp_gene_map) > 0 else 0}\n")
        f.write(f"Missing selected SNPs in SNP metadata: {len(missing_snps)}\n\n")

        if len(missing_snps) > 0:
            f.write("Missing selected SNPs in SNP metadata, first 100:\n")
            f.write("\n".join(missing_snps[:100]))
            f.write("\n\n")

        f.write("Gene count per region summary:\n")
        f.write(out["n_genes_from_selected_snps_10kb"].describe().to_string())
        f.write("\n\n")

        f.write("Top 20 rows by n_genes_from_selected_snps_10kb:\n")
        cols = [
            "region",
            "n_seeds_selected",
            "n_unique_snps_selected",
            "n_genes_from_selected_snps_10kb",
            "genes_from_selected_snps_10kb",
            "selection_frequency",
        ]
        cols = [c for c in cols if c in out.columns]
        f.write(
            out.sort_values(
                ["n_genes_from_selected_snps_10kb", "n_seeds_selected"],
                ascending=[False, False]
            )[cols].head(20).to_string(index=False)
        )
        f.write("\n")

    print("\nSaved:")
    print(OUT_FILE)
    print(SNP_GENE_MAP_OUT)
    print(REPORT_FILE)

    print("\nPreview:")
    preview_cols = [
        "region",
        "n_seeds_selected",
        "n_unique_snps_selected",
        "unique_snps_selected",
        "n_genes_from_selected_snps_10kb",
        "genes_from_selected_snps_10kb",
        "selection_frequency",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]
    print(out[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
