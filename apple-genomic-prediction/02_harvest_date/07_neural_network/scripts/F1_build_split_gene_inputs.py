# -*- coding: utf-8 -*-

################################################################################
### F1_build_split_gene_inputs.py
### Build split-specific inputs for bio_geni_relu_mappedhidden_fusionhidden
################################################################################

from pathlib import Path
import pandas as pd
import numpy as np

TRAIT = "Harvest_date"

OUT_DIR = (
    Path("02_harvest_date")
    / "07_neural_network"
    / "output"
)

BIO_DIR = OUT_DIR / "biologic_objects"

SNP_GENE_DIR = (
    BIO_DIR
    / "snp_gene_mapping"
)

SPLIT_INPUT_DIR = (
    BIO_DIR
    / "split_inputs"
)

SPLIT_INPUT_DIR.mkdir(parents=True, exist_ok=True)

GENO_DIR = (
    Path("02_harvest_date")
    / "05_gradient_boosting"
    / "output"
    / "geno_files"
    / TRAIT
)

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


def save_list_csv(items, colname, out_file: Path):
    pd.DataFrame({colname: list(items)}).to_csv(out_file, index=False)


def main():
    split_to_snps = load_split_snps(GENO_DIR)

    summary_rows = []

    for split_name, split_snps in split_to_snps.items():
        print(f"\nPreparing split-specific gene inputs for {split_name}")

        split_dir = SPLIT_INPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_edges_file = SNP_GENE_DIR / f"{split_name}_snp_gene_edges.csv"
        split_gene_annot_file = SNP_GENE_DIR / f"{split_name}_gene_annotation.csv"

        if not split_edges_file.exists():
            raise FileNotFoundError(f"Missing SNP-gene edge file: {split_edges_file}")
        if not split_gene_annot_file.exists():
            raise FileNotFoundError(f"Missing gene annotation file: {split_gene_annot_file}")

        split_edges = pd.read_csv(split_edges_file)
        gene_annot_df = pd.read_csv(split_gene_annot_file)

        all_split_snps = list(map(str, split_snps))
        mapped_snps = sorted(split_edges["SNP"].astype(str).unique().tolist())
        unmapped_snps = sorted(set(all_split_snps) - set(mapped_snps))
        all_genes = sorted(gene_annot_df["Gene"].astype(str).unique().tolist())

        snp_gene_edges = split_edges[["SNP", "Gene"]].drop_duplicates().reset_index(drop=True)

        save_list_csv(all_split_snps, "SNP", split_dir / "all_split_snps.csv")
        save_list_csv(mapped_snps, "SNP", split_dir / "mapped_snps.csv")
        save_list_csv(unmapped_snps, "SNP", split_dir / "unmapped_snps.csv")
        save_list_csv(all_genes, "Gene", split_dir / "all_genes.csv")

        gene_annot_df.to_csv(split_dir / "gene_annotation.csv", index=False)
        snp_gene_edges.to_csv(split_dir / "snp_to_gene_edges.csv", index=False)

        mapped_snp_index = pd.DataFrame({
            "SNP": mapped_snps,
            "mapped_snp_idx": np.arange(len(mapped_snps))
        })

        unmapped_snp_index = pd.DataFrame({
            "SNP": unmapped_snps,
            "unmapped_snp_idx": np.arange(len(unmapped_snps))
        })

        gene_index = pd.DataFrame({
            "Gene": all_genes,
            "gene_idx": np.arange(len(all_genes))
        })

        mapped_snp_index.to_csv(split_dir / "mapped_snp_index.csv", index=False)
        unmapped_snp_index.to_csv(split_dir / "unmapped_snp_index.csv", index=False)
        gene_index.to_csv(split_dir / "gene_index.csv", index=False)

        snp_gene_idx_edges = (
            snp_gene_edges
            .merge(mapped_snp_index, on="SNP", how="left")
            .merge(gene_index, on="Gene", how="left")
            [["mapped_snp_idx", "gene_idx", "SNP", "Gene"]]
            .drop_duplicates()
            .sort_values(["mapped_snp_idx", "gene_idx"])
            .reset_index(drop=True)
        )
        snp_gene_idx_edges.to_csv(split_dir / "snp_to_gene_edges_indexed.csv", index=False)

        summary_rows.append({
            "Split": split_name,
            "n_all_split_snps": len(all_split_snps),
            "n_mapped_snps": len(mapped_snps),
            "n_unmapped_snps": len(unmapped_snps),
            "fraction_mapped_snps": len(mapped_snps) / len(all_split_snps) if len(all_split_snps) > 0 else np.nan,
            "n_genes": len(all_genes),
            "n_snp_gene_edges": len(snp_gene_edges),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SPLIT_INPUT_DIR / "split_input_summary.csv", index=False)

    print("\nSaved split-specific gene inputs.")
    print("Key file:")
    print(SPLIT_INPUT_DIR / "split_input_summary.csv")


if __name__ == "__main__":
    main()
