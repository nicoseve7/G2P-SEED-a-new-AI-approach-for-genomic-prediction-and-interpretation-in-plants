import pandas as pd
from pathlib import Path

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ROOT_DIR = BASE_DIR.parent

BIM_FILE = (
    Path("data")
    / "raw"
    / "genotype"
    / "SNPs_final_2022.bim"
)
GFF3_FILE = (
    Path("data")
    / "raw"
    / "annotation"
    / "gene_models_20170612.gff3"
)

OUT_DIR = ROOT_DIR / "Output/biologic_objects/snp_gene_mapping_newtraits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_FLANK_BP = 3000


def normalize_chr(x):
    x = str(x).strip()
    x = x.replace("Chr", "").replace("chr", "")
    try:
        return str(int(float(x)))
    except ValueError:
        return x


# ============================================================
# BIM
# ============================================================

bim = pd.read_csv(
    BIM_FILE,
    sep=r"\s+",
    header=None,
    names=["CHROM", "SNP", "CM", "POS", "A1", "A2"]
)

bim["SNP"] = bim["SNP"].astype(str).str.strip()
bim["CHROM"] = bim["CHROM"].apply(normalize_chr)

positions = bim[["SNP", "CHROM", "POS"]].copy()

positions.to_csv(
    OUT_DIR / "global_snp_positions_newtraits.csv",
    index=False
)


# ============================================================
# GFF3
# ============================================================

gff = pd.read_csv(
    GFF_FILE,
    sep="\t",
    comment="#",
    header=None,
    names=[
        "seqid", "source", "type", "start", "end",
        "score", "strand", "phase", "attributes"
    ]
)

genes = gff[
    gff["type"].astype(str).str.lower() == "gene"
].copy()


def extract_gene_id(attributes):
    for item in str(attributes).split(";"):
        item = item.strip()

        if item.startswith("Name="):
            value = item.split("=", 1)[1]
            if value.startswith("MD"):
                return value

    for item in str(attributes).split(";"):
        item = item.strip()

        if item.startswith("ID="):
            value = item.split("=", 1)[1]
            value = value.replace("gene:", "")
            return value

    return None


genes["Gene"] = genes["attributes"].apply(extract_gene_id)

genes = genes.dropna(subset=["Gene"]).copy()

genes["CHROM"] = genes["seqid"].apply(normalize_chr)

genes["window_start"] = (
    genes["start"] - GENE_FLANK_BP
).clip(lower=1)

genes["window_end"] = (
    genes["end"] + GENE_FLANK_BP
)


# ============================================================
# SNP -> GENE
# ============================================================

genes_by_chr = {
    chrom: df.copy()
    for chrom, df in genes.groupby("CHROM")
}

rows = []

for chrom, snps_chr in positions.groupby("CHROM"):

    if chrom not in genes_by_chr:
        continue

    genes_chr = genes_by_chr[chrom]

    for snp_row in snps_chr.itertuples(index=False):

        hits = genes_chr[
            (genes_chr["window_start"] <= snp_row.POS)
            &
            (genes_chr["window_end"] >= snp_row.POS)
        ]

        for gene_row in hits.itertuples(index=False):
            rows.append({
                "SNP": snp_row.SNP,
                "Gene": gene_row.Gene,
                "CHROM": snp_row.CHROM,
                "POS": snp_row.POS,
            })


edges = pd.DataFrame(rows)

edges = (
    edges
    .drop_duplicates(subset=["SNP", "Gene"])
    .reset_index(drop=True)
)

edges.to_csv(
    OUT_DIR / "global_snp_gene_edges_newtraits.csv",
    index=False
)

print("Total SNPs in BIM:", positions["SNP"].nunique())
print("Mapped SNPs:", edges["SNP"].nunique())
print("Genes:", edges["Gene"].nunique())
print("Edges:", len(edges))
