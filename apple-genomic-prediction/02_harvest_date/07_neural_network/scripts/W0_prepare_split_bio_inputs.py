# -*- coding: utf-8 -*-

################################################################################
### W0_prepare_split_bio_inputs.py
################################################################################

from pathlib import Path
import pandas as pd

TRAIT = "Harvest_date"

GENO_DIR = Path("../Output/Intermediate/geno_files") / TRAIT
GLOBAL_SNP_GENE_FILE = Path("../bio_geni_relu_concathidden/Output/biologic_objects/snp_gene_mapping/global_snp_gene_edges.csv")

OUT_BASE_DIR = Path("Output/biologic_objects/split_inputs")
REPORTS_DIR = Path("Output/biologic_objects/reports")

OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_split_snp_names(geno_file: Path):
    df = pd.read_csv(geno_file, nrows=1)
    first_col = df.columns[0]
    snp_cols = [c for c in df.columns if c != first_col]
    return snp_cols

def main():
    print("Loading global SNP->gene mapping...")
    snp_gene = pd.read_csv(GLOBAL_SNP_GENE_FILE, dtype=str)
    snp_gene["SNP"] = snp_gene["SNP"].astype(str).str.strip()
    snp_gene["Gene"] = snp_gene["Gene"].astype(str).str.strip()

    geno_files = sorted(GENO_DIR.glob("geno_CV*_Split*.csv"))
    if len(geno_files) == 0:
        raise ValueError(f"No split-specific geno files found in {GENO_DIR}")

    summary_rows = []

    for geno_file in geno_files:
        split_name = geno_file.stem.replace("geno_", "")
        print(f"\nProcessing {split_name}...")

        out_dir = OUT_BASE_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        split_snps = load_split_snp_names(geno_file)
        split_snps_set = set(split_snps)

        mapped_edges = (
            snp_gene[snp_gene["SNP"].isin(split_snps_set)]
            .drop_duplicates(subset=["SNP", "Gene"])
            .reset_index(drop=True)
        )

        mapped_snps = sorted(mapped_edges["SNP"].astype(str).unique().tolist())
        all_genes = sorted(mapped_edges["Gene"].astype(str).unique().tolist())
        unmapped_snps = sorted(split_snps_set - set(mapped_snps))

        pd.DataFrame({"SNP": mapped_snps}).to_csv(out_dir / "mapped_snps.csv", index=False)
        pd.DataFrame({"SNP": unmapped_snps}).to_csv(out_dir / "unmapped_snps.csv", index=False)
        pd.DataFrame({"Gene": all_genes}).to_csv(out_dir / "all_genes.csv", index=False)
        mapped_edges.to_csv(out_dir / "snp_to_gene_edges.csv", index=False)

        summary_rows.append({
            "Split": split_name,
            "n_snps_total": len(split_snps),
            "n_mapped_snps": len(mapped_snps),
            "n_unmapped_snps": len(unmapped_snps),
            "fraction_mapped_snps": round(len(mapped_snps) / max(len(split_snps), 1), 6),
            "n_genes": len(all_genes),
            "n_snp_gene_edges": len(mapped_edges),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORTS_DIR / "split_bio_object_summary.csv", index=False)

    print("\nSaved:")
    print(REPORTS_DIR / "split_bio_object_summary.csv")

if __name__ == "__main__":
    main()