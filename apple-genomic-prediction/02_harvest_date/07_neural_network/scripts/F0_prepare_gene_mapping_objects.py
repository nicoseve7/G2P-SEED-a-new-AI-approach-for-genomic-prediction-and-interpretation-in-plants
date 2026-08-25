# -*- coding: utf-8 -*-

################################################################################
### F0_prepare_gene_mapping_objects.py
### Prepare SNP->gene mapping objects for bio_geni_relu_mappedhidden_fusionhidden
################################################################################

from pathlib import Path
import pandas as pd
import numpy as np


TRAIT = "Harvest_date"
MAP_TOLERANCE_BP = 3000

GENO_DIR = Path(f"../Output/Intermediate/geno_files/{TRAIT}")
VCF_FILE = Path("Input/SNPS_final2022.vcf")

GFF3_FILE = Path("Input/annotation/gene_models_20170612.gff3")
SWISSPROT_FILE = Path("Input/annotation/Malus_x_domestica_GDDH13_v1.1_vs_swissprot.xlsx")

OUT_DIR = Path("Output")
BIO_DIR = OUT_DIR / "biologic_objects"

SNP_GENE_DIR = BIO_DIR / "snp_gene_mapping"
REPORT_DIR = BIO_DIR / "reports"

SNP_GENE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_chrom_name(chrom):
    chrom = str(chrom).strip()
    chrom = chrom.replace("chromosome", "")
    chrom = chrom.replace("Chromosome", "")
    chrom = chrom.replace("chr", "")
    chrom = chrom.replace("Chr", "")
    chrom = chrom.strip()
    return f"Chr{chrom}"


def parse_attributes(attr_string: str) -> dict:
    attrs = {}
    for item in str(attr_string).split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            attrs[k] = v
    return attrs


def extract_gene_name_from_gff_attrs(attr_string: str) -> str:
    attrs = parse_attributes(attr_string)

    if "Name" in attrs and str(attrs["Name"]).startswith("MD"):
        return str(attrs["Name"])

    if "ID" in attrs:
        gene_id = str(attrs["ID"]).replace("gene:", "")
        if gene_id.startswith("MD"):
            return gene_id

    return None


def load_split_snps(geno_dir: Path):
    files = sorted(geno_dir.glob("geno_CV*_Split*.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No split-specific geno files found in: {geno_dir}")

    split_to_snps = {}
    for f in files:
        split_name = f.stem.replace("geno_", "")
        tmp = pd.read_csv(f, nrows=1)
        cols = tmp.columns.tolist()
        snps = cols[1:]
        split_to_snps[split_name] = snps

    return split_to_snps


def load_gene_models(gff3_file: Path):
    rows = []

    with open(gff3_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue

            chrom, source, feature_type, start, end, score, strand, phase, attrs = parts

            if feature_type != "gene":
                continue

            gene_name = extract_gene_name_from_gff_attrs(attrs)
            if gene_name is None:
                continue

            rows.append({
                "Gene": gene_name,
                "Chrom": normalize_chrom_name(chrom),
                "start": int(start),
                "end": int(end),
                "strand": strand,
            })

    genes = pd.DataFrame(rows)

    if len(genes) == 0:
        raise ValueError("No genes parsed from GFF3. Check gene_models_20170612.gff3.")

    genes = genes.drop_duplicates().reset_index(drop=True)
    return genes


def load_swissprot_annotations(excel_file: Path):
    df = pd.read_excel(excel_file)

    gene_col = None
    for c in df.columns:
        if "Query" in str(c):
            gene_col = c
            break
    if gene_col is None:
        raise ValueError("Could not find Query column in SwissProt file.")

    match_col = None
    for c in df.columns:
        if "Match" in str(c):
            match_col = c
            break

    desc_col = None
    for c in df.columns:
        if "Description" in str(c):
            desc_col = c
            break

    score_col = None
    for c in df.columns:
        if str(c).lower() == "score":
            score_col = c
            break

    out = df[[gene_col]].copy()
    out.columns = ["Gene"]

    out["SwissProt_Match"] = df[match_col] if match_col is not None else np.nan
    out["Description"] = df[desc_col] if desc_col is not None else np.nan
    out["Score"] = df[score_col] if score_col is not None else np.nan

    out["Gene"] = out["Gene"].astype(str)
    out = out[out["Gene"].str.startswith("MD")].copy()
    out = out.sort_values(["Gene", "Score"], ascending=[True, False])
    out = out.drop_duplicates(subset=["Gene"], keep="first").reset_index(drop=True)

    return out


def load_vcf_positions(vcf_file: Path, target_snps: set):
    rows = []

    with open(vcf_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue

            chrom, pos, snp_id, ref, alt = parts[:5]

            if snp_id in target_snps:
                rows.append({
                    "SNP": snp_id,
                    "Chrom": normalize_chrom_name(chrom),
                    "POS": int(pos),
                    "REF": ref,
                    "ALT": alt,
                })

    snp_pos = pd.DataFrame(rows)

    if len(snp_pos) == 0:
        raise ValueError("No target SNP positions found in VCF.")

    snp_pos = snp_pos.drop_duplicates(subset=["SNP"]).reset_index(drop=True)
    return snp_pos


def map_snps_to_genes(snp_pos_df: pd.DataFrame, genes_df: pd.DataFrame, tol_bp: int):
    edges = []

    genes_by_chr = {
        chrom: sub.reset_index(drop=True)
        for chrom, sub in genes_df.groupby("Chrom")
    }

    for _, snp_row in snp_pos_df.iterrows():
        snp = snp_row["SNP"]
        chrom = snp_row["Chrom"]
        pos = snp_row["POS"]

        if chrom not in genes_by_chr:
            continue

        gsub = genes_by_chr[chrom]
        mask = (gsub["start"] - tol_bp <= pos) & (pos <= gsub["end"] + tol_bp)
        matched = gsub.loc[mask, ["Gene", "Chrom", "start", "end"]]

        for _, grow in matched.iterrows():
            edges.append({
                "SNP": snp,
                "Gene": grow["Gene"],
                "Chrom": chrom,
                "POS": pos,
            })

    if len(edges) == 0:
        return pd.DataFrame(columns=["SNP", "Gene", "Chrom", "POS"])

    return pd.DataFrame(edges).drop_duplicates().reset_index(drop=True)


def main():
    print("Loading split-specific SNP lists...")
    split_to_snps = load_split_snps(GENO_DIR)

    all_snps = sorted(set(sum(split_to_snps.values(), [])))
    print(f"Number of unique SNPs across all splits: {len(all_snps)}")

    print("Loading gene models...")
    genes = load_gene_models(GFF3_FILE)
    print(f"Number of genes parsed from GFF3: {len(genes)}")

    print("Loading SwissProt annotations...")
    swiss = load_swissprot_annotations(SWISSPROT_FILE)
    print(f"SwissProt annotations loaded for {len(swiss)} genes")

    print("Loading SNP positions from VCF...")
    snp_pos = load_vcf_positions(VCF_FILE, set(all_snps))
    print(f"SNPs with coordinates found in VCF: {len(snp_pos)} / {len(all_snps)}")

    print("Example chromosome names in genes:")
    print(sorted(genes["Chrom"].dropna().astype(str).unique())[:10])

    print("Example chromosome names in SNP positions:")
    print(sorted(snp_pos["Chrom"].dropna().astype(str).unique())[:10])

    print("Mapping SNPs to genes...")
    snp_gene_edges = map_snps_to_genes(snp_pos, genes, MAP_TOLERANCE_BP)
    print(f"SNP-gene edges created: {len(snp_gene_edges)}")

    if len(snp_gene_edges) == 0:
        raise ValueError(
            "SNP-gene mapping returned 0 edges. Most likely chromosome naming mismatch between VCF and GFF3."
        )

    snp_pos.to_csv(SNP_GENE_DIR / "global_snp_positions.csv", index=False)
    snp_gene_edges.to_csv(SNP_GENE_DIR / "global_snp_gene_edges.csv", index=False)

    global_gene_table = (
        genes.merge(swiss, on="Gene", how="left")
        .sort_values(["Chrom", "start", "Gene"])
        .reset_index(drop=True)
    )
    global_gene_table.to_csv(REPORT_DIR / "global_gene_annotation_table.csv", index=False)
    swiss.to_csv(REPORT_DIR / "global_gene_swissprot_annotation.csv", index=False)

    snp_map_summary = (
        snp_gene_edges.groupby("SNP")
        .agg(n_genes=("Gene", "nunique"))
        .reset_index()
    )

    mapped_snps = set(snp_gene_edges["SNP"].unique())
    unmapped_snps = sorted(set(all_snps) - mapped_snps)

    snp_map_summary.to_csv(REPORT_DIR / "snp_gene_mapping_summary_per_snp.csv", index=False)
    pd.DataFrame({"SNP": unmapped_snps}).to_csv(REPORT_DIR / "unmapped_snps.csv", index=False)

    split_rows = []
    split_snp_counts = []

    for split_name, snps in split_to_snps.items():
        print(f"\nPreparing gene objects for {split_name}")

        split_edges = snp_gene_edges[snp_gene_edges["SNP"].isin(snps)].copy()
        split_edges.to_csv(SNP_GENE_DIR / f"{split_name}_snp_gene_edges.csv", index=False)

        mapped_snps_split = sorted(split_edges["SNP"].unique().tolist())
        n_unique_mapped_snps = len(mapped_snps_split)
        fraction_mapped_snps = n_unique_mapped_snps / len(snps) if len(snps) > 0 else np.nan

        split_genes = sorted(split_edges["Gene"].unique().tolist())
        split_gene_df = global_gene_table[global_gene_table["Gene"].isin(split_genes)].copy()
        split_gene_df.to_csv(SNP_GENE_DIR / f"{split_name}_gene_annotation.csv", index=False)

        split_rows.append({
            "Split": split_name,
            "n_snps_in_split": len(snps),
            "n_unique_mapped_snps": n_unique_mapped_snps,
            "fraction_mapped_snps": fraction_mapped_snps,
            "n_mapped_snp_gene_edges": len(split_edges),
            "n_unique_genes": len(split_genes),
        })

        split_snp_counts.append({
            "Split": split_name,
            "n_snps_in_split": len(snps),
        })

    pd.DataFrame(split_rows).to_csv(REPORT_DIR / "split_gene_object_summary.csv", index=False)
    pd.DataFrame(split_snp_counts).to_csv(REPORT_DIR / "split_snp_counts.csv", index=False)

    print("\nSaved global and split-specific gene-mapping objects.")
    print("Key files:")
    print(SNP_GENE_DIR / "global_snp_positions.csv")
    print(SNP_GENE_DIR / "global_snp_gene_edges.csv")
    print(REPORT_DIR / "global_gene_annotation_table.csv")
    print(REPORT_DIR / "split_gene_object_summary.csv")


if __name__ == "__main__":
    main()