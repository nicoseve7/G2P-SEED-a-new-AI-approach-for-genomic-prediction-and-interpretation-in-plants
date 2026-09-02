# -*- coding: utf-8 -*-

################################################################################
### Q8_annotate_ranked_regions_newtraits.py
###
### Annotate Q7 ranked genomic regions with genes from GDDH13 GFF3.
################################################################################

from pathlib import Path
import re
import pandas as pd
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]
WINDOW_LABELS = ["50kb", "100kb"]
TOP_K = 1000
NEARBY_BP = 10000

GA_OUT_DIR = (
    Path("03_acidity_color_over")
    / "06_genetic_algorithm"
    / "output"
)

RANK_BASE_DIR = GA_OUT_DIR / "01_region_ranking"
OUT_BASE_DIR = GA_OUT_DIR / "02_region_annotations"

GFF3_FILE = (
    Path("data")
    / "raw"
    / "annotation"
    / "gene_models_20170612.gff3"
)

# optional: if present, it uses it to add functional annotations
SWISSPROT_FILE = (
    Path("data")
    / "raw"
    / "annotation"
    / "Malus_x_domestica_GDDH13_v1.1_vs_swissprot.xlsx"
)


# =============================================================================
# HELPERS
# =============================================================================

def find_existing_file(candidates, label, required=True):
    for p in candidates:
        if p.exists():
            return p

    if required:
        raise FileNotFoundError(
            f"Nessun file trovato per {label}.\n"
            "Path provati:\n" + "\n".join(str(x) for x in candidates)
        )

    return None


def normalize_chr(x):
    x = str(x).strip()
    x = x.replace("chr", "").replace("Chr", "").replace("CHR", "")
    x = x.replace("GDDH13_", "")
    x = x.replace("gddh13_", "")
    x = x.replace("Md", "")
    x = x.strip()

    # se è tipo "01", diventa "1"
    if x.isdigit():
        x = str(int(x))

    return x


def parse_gff_attributes(attr):
    out = {}
    attr = str(attr)

    for item in attr.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()

    return out


def extract_gene_id(attr):
    d = parse_gff_attributes(attr)

    for key in ["ID", "Name", "gene_id", "GeneID", "locus_tag"]:
        if key in d and str(d[key]).strip() != "":
            val = str(d[key]).strip()
            val = re.sub(r"^gene:", "", val)
            return val

    return ""


def clean_gene_id(g):
    g = str(g).strip()
    g = re.sub(r"^gene:", "", g)
    return g


def load_gff3_genes():
    if not GFF3_FILE.exists():
    raise FileNotFoundError(
        f"GFF3 non trovato:\n{GFF3_FILE}"
    )

    gff3_file = GFF3_FILE

    print(f"[INFO] Uso GFF3: {gff3_file}")

    rows = []

    with open(gff3_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")

            if len(parts) < 9:
                continue

            seqid, source, feature_type, start, end, score, strand, phase, attributes = parts

            if feature_type != "gene":
                continue

            gene_id = extract_gene_id(attributes)

            if gene_id == "":
                continue

            rows.append({
                "gene_id": clean_gene_id(gene_id),
                "CHROM": normalize_chr(seqid),
                "gene_start": int(start),
                "gene_end": int(end),
                "strand": strand,
                "gff3_attributes": attributes,
            })

    genes = pd.DataFrame(rows)

    if genes.empty:
        raise ValueError("Nessun gene trovato nel GFF3. Controlla feature_type o formato file.")

    genes = genes.drop_duplicates(subset=["gene_id", "CHROM", "gene_start", "gene_end"]).copy()

    print(f"[INFO] Geni caricati dal GFF3: {genes.shape[0]}")
    print(f"[INFO] Cromosomi nel GFF3: {sorted(genes['CHROM'].unique().tolist())[:20]}")

    return genes


def load_swissprot_optional():
    swiss_file = SWISSPROT_FILE if SWISSPROT_FILE.exists() else None

    if swiss_file is None:
        print("[INFO] File SwissProt non trovato. Procedo solo con gene_id.")
        return None

    print(f"[INFO] Uso SwissProt opzionale: {swiss_file}")

    try:
        df = pd.read_excel(swiss_file)
    except Exception as e:
        print(f"[WARNING] Impossibile leggere SwissProt: {e}")
        return None

    # Find a possible gene column
    gene_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "gene" in cl or "query" in cl or "md" in cl:
            gene_col = c
            break

    if gene_col is None:
        print("[WARNING] Non trovo colonna gene in SwissProt. Ignoro annotazioni funzionali.")
        return None

    df = df.copy()
    df["gene_id"] = df[gene_col].astype(str).apply(clean_gene_id)

    # Let's keep a few columns readable, if there are any
    candidate_cols = ["gene_id"]
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["swiss", "uniprot", "description", "protein", "annotation", "name", "hit", "subject"]):
            candidate_cols.append(c)

    candidate_cols = list(dict.fromkeys(candidate_cols))
    out = df[candidate_cols].drop_duplicates(subset=["gene_id"]).copy()

    print(f"[INFO] Annotazioni SwissProt caricate: {out.shape}")

    return out


def parse_region_id(region_id):
    """
    Expected:
        chr1:1-50000
        chr10:300001-350000
    """
    txt = str(region_id).strip()
    txt = txt.replace("Chr", "chr").replace("CHR", "chr")

    m = re.match(r"chr(.+):(\d+)-(\d+)", txt)

    if not m:
        m = re.match(r"(.+):(\d+)-(\d+)", txt)

    if not m:
        return None, np.nan, np.nan

    chrom = normalize_chr(m.group(1))
    start = int(m.group(2))
    end = int(m.group(3))

    return chrom, start, end


def annotate_regions(regions: pd.DataFrame, genes: pd.DataFrame, swiss=None):
    out_rows = []

    regions = regions.copy()

    if "region_id" not in regions.columns:
        raise ValueError("Nel file regioni manca la colonna region_id.")

    parsed = regions["region_id"].apply(parse_region_id)
    regions["region_CHROM_norm"] = [p[0] for p in parsed]
    regions["region_start_parsed"] = [p[1] for p in parsed]
    regions["region_end_parsed"] = [p[2] for p in parsed]

    # Use existing columns if they exist; otherwise, use the parsed ones
    if "CHROM" in regions.columns:
        regions["CHROM_norm"] = regions["CHROM"].apply(normalize_chr)
    else:
        regions["CHROM_norm"] = regions["region_CHROM_norm"]

    if "region_start" not in regions.columns:
        regions["region_start"] = regions["region_start_parsed"]

    if "region_end" not in regions.columns:
        regions["region_end"] = regions["region_end_parsed"]

    genes_by_chr = {
        chrom: sub.copy()
        for chrom, sub in genes.groupby("CHROM")
    }

    for _, row in regions.iterrows():
        chrom = normalize_chr(row["CHROM_norm"])
        start = int(row["region_start"])
        end = int(row["region_end"])

        subgenes = genes_by_chr.get(chrom, pd.DataFrame())

        if len(subgenes) == 0:
            genes_inside = []
            genes_nearby = []
        else:
            inside_mask = (
                (subgenes["gene_start"] <= end) &
                (subgenes["gene_end"] >= start)
            )

            nearby_start = start - NEARBY_BP
            nearby_end = end + NEARBY_BP

            nearby_mask = (
                (subgenes["gene_start"] <= nearby_end) &
                (subgenes["gene_end"] >= nearby_start)
            )

            inside_df = subgenes[inside_mask].copy()
            nearby_df = subgenes[nearby_mask & (~inside_mask)].copy()

            genes_inside = sorted(inside_df["gene_id"].astype(str).unique().tolist())
            genes_nearby = sorted(nearby_df["gene_id"].astype(str).unique().tolist())

        new_row = row.to_dict()
        new_row["n_genes_inside"] = len(genes_inside)
        new_row["genes_inside"] = ";".join(genes_inside)
        new_row[f"n_genes_nearby_{NEARBY_BP//1000}kb"] = len(genes_nearby)
        new_row[f"genes_nearby_{NEARBY_BP//1000}kb"] = ";".join(genes_nearby)

        out_rows.append(new_row)

    annotated = pd.DataFrame(out_rows)

    # Optional addition of aggregated SwissProt descriptions
    if swiss is not None and "genes_inside" in annotated.columns:
        swiss_cols = [c for c in swiss.columns if c != "gene_id"]

        swiss_map = swiss.set_index("gene_id").to_dict(orient="index")

        def summarize_swiss(gene_list_str, max_items=10):
            genes_list = [
                clean_gene_id(x)
                for x in str(gene_list_str).split(";")
                if str(x).strip() != ""
            ]

            descs = []

            for g in genes_list:
                if g in swiss_map:
                    vals = swiss_map[g]
                    txt_parts = []
                    for c in swiss_cols[:3]:
                        val = vals.get(c, "")
                        if pd.notna(val) and str(val).strip() != "":
                            txt_parts.append(f"{c}={val}")
                    if txt_parts:
                        descs.append(g + " [" + " | ".join(txt_parts) + "]")

            return "; ".join(descs[:max_items])

        annotated["genes_inside_swissprot_summary"] = annotated["genes_inside"].apply(summarize_swiss)
        annotated[f"genes_nearby_{NEARBY_BP//1000}kb_swissprot_summary"] = annotated[
            f"genes_nearby_{NEARBY_BP//1000}kb"
        ].apply(summarize_swiss)

    # Remove the technical columns if you want to keep it clean
    technical_cols = ["region_CHROM_norm", "region_start_parsed", "region_end_parsed", "CHROM_norm"]
    annotated = annotated.drop(columns=[c for c in technical_cols if c in annotated.columns])

    return annotated


def process_one_file(infile: Path, outfile: Path, genes: pd.DataFrame, swiss=None):
    if not infile.exists():
        print(f"[WARNING] File non trovato, skip: {infile}")
        return None

    df = pd.read_csv(infile)

    annotated = annotate_regions(df, genes, swiss=swiss)
    annotated.to_csv(outfile, index=False)

    return {
        "input_file": str(infile),
        "output_file": str(outfile),
        "n_rows": annotated.shape[0],
        "n_regions": annotated["region_id"].nunique() if "region_id" in annotated.columns else annotated.shape[0],
        "regions_with_genes_inside": int((annotated["n_genes_inside"] > 0).sum()),
        f"regions_with_genes_nearby_{NEARBY_BP//1000}kb": int((annotated[f"n_genes_nearby_{NEARBY_BP//1000}kb"] > 0).sum()),
    }


def process_trait(trait: str, genes: pd.DataFrame, swiss=None):
    print("\n" + "=" * 100)
    print(f"Q8 - ANNOTATE RANKED REGIONS | TRAIT: {trait}")
    print("=" * 100)

    rank_dir = RANK_BASE_DIR / trait
    out_dir = OUT_BASE_DIR / trait
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for window_label in WINDOW_LABELS:
        kb = window_label.replace("kb", "")

        files_to_process = [
            (
                rank_dir / f"ranked_regions_{window_label}_{trait}.csv",
                out_dir / f"ranked_regions_annotated_{window_label}_{trait}.csv",
            ),
            (
                rank_dir / f"top{TOP_K}_regions_by_region_score_{window_label}_{trait}.csv",
                out_dir / f"top{TOP_K}_regions_by_region_score_annotated_{window_label}_{trait}.csv",
            ),
            (
                rank_dir / f"top{TOP_K}_regions_by_meanSHAP_{window_label}_{trait}.csv",
                out_dir / f"top{TOP_K}_regions_by_meanSHAP_annotated_{window_label}_{trait}.csv",
            ),
            (
                rank_dir / f"top{TOP_K}_regions_by_maxSHAP_{window_label}_{trait}.csv",
                out_dir / f"top{TOP_K}_regions_by_maxSHAP_annotated_{window_label}_{trait}.csv",
            ),
        ]

        for infile, outfile in files_to_process:
            print(f"\nAnnotating:\n  {infile}")
            res = process_one_file(infile, outfile, genes, swiss=swiss)

            if res is not None:
                res["Trait"] = trait
                res["window_label"] = window_label
                rows.append(res)

                print(f"Saved:\n  {outfile}")
                print(
                    f"  rows={res['n_rows']} | "
                    f"with genes inside={res['regions_with_genes_inside']} | "
                    f"with genes nearby={res[f'regions_with_genes_nearby_{NEARBY_BP//1000}kb']}"
                )

    summary = pd.DataFrame(rows)

    summary_file = out_dir / f"Q8_annotation_summary_{trait}.csv"
    summary.to_csv(summary_file, index=False)

    report_file = out_dir / f"Q8_annotation_report_{trait}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"Q8 ANNOTATION REPORT: {trait}\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"NEARBY_BP: {NEARBY_BP}\n")
        f.write(f"TOP_K: {TOP_K}\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    print("\nSaved summary:")
    print(summary_file)
    print("Saved report:")
    print(report_file)

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("Q8 - ANNOTATE RANKED REGIONS NEW TRAITS")
    print("=" * 100)

    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    genes = load_gff3_genes()
    swiss = load_swissprot_optional()

    all_summaries = []

    for trait in TRAITS:
        trait_summary = process_trait(trait, genes, swiss=swiss)
        all_summaries.append(trait_summary)

    all_summary = pd.concat(all_summaries, ignore_index=True)

    all_summary_file = OUT_BASE_DIR / "Q8_annotation_summary_all_traits.csv"
    all_summary.to_csv(all_summary_file, index=False)

    print("\n" + "=" * 100)
    print("Q8 completed.")
    print("=" * 100)
    print(all_summary.to_string(index=False))
    print("\nSaved global summary:")
    print(all_summary_file)


if __name__ == "__main__":
    main()
