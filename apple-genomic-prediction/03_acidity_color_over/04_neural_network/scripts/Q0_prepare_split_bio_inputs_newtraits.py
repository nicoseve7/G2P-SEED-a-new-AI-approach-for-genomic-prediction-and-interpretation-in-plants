# -*- coding: utf-8 -*-

################################################################################
### Q0_prepare_split_bio_inputs_newtraits.py
###
### Prepara gli oggetti biologici split-specific per la rete no-soil
### sui nuovi tratti:
###   - Acidity
###   - Color_over
###
### Per ogni trait e per ogni split:
###   - legge geno_CV*_Split*.csv
###   - prende la lista degli SNP nello split
###   - usa il mapping globale SNP -> Gene già costruito
###   - separa SNP mapped e unmapped
###   - salva:
###       mapped_snps.csv
###       unmapped_snps.csv
###       all_genes.csv
###       snp_to_gene_edges.csv
###
### Output:
###   Output/biologic_objects/<Trait>/split_inputs/<Split>/
################################################################################

from pathlib import Path
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

GENO_BASE_DIR = Path("Output/Intermediate/geno_files")

GLOBAL_SNP_GENE_FILE = Path(
    "../bio_geni_relu_concathidden/Output/biologic_objects/"
    "snp_gene_mapping/global_snp_gene_edges.csv"
)

OUT_BIO_BASE_DIR = Path("Output/biologic_objects")
REPORTS_BASE_DIR = OUT_BIO_BASE_DIR / "reports"

REPORTS_BASE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File non trovato:\n{path}")


def load_split_snp_names(geno_file: Path):
    """
    Legge solo header + prima riga del file geno split-specific.
    La prima colonna è Genotype / rownames, tutte le altre sono SNP.
    """
    df = pd.read_csv(geno_file, nrows=1)
    first_col = df.columns[0]
    snp_cols = [c for c in df.columns if c != first_col]
    return snp_cols


def process_one_trait(trait: str, snp_gene: pd.DataFrame):
    print("\n" + "=" * 80)
    print(f"Processing trait: {trait}")
    print("=" * 80)

    geno_dir = GENO_BASE_DIR / trait

    if not geno_dir.exists():
        raise FileNotFoundError(f"Cartella geno non trovata per {trait}:\n{geno_dir}")

    out_base_dir = OUT_BIO_BASE_DIR / trait / "split_inputs"
    reports_dir = OUT_BIO_BASE_DIR / trait / "reports"

    out_base_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    geno_files = sorted(geno_dir.glob("geno_CV*_Split*.csv"))

    if len(geno_files) == 0:
        raise ValueError(f"No split-specific geno files found in {geno_dir}")

    print(f"Found geno files: {len(geno_files)}")

    summary_rows = []

    for geno_file in geno_files:
        split_name = geno_file.stem.replace("geno_", "")
        print(f"  Processing {split_name}...")

        out_dir = out_base_dir / split_name
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

        pd.DataFrame({"SNP": mapped_snps}).to_csv(
            out_dir / "mapped_snps.csv",
            index=False
        )

        pd.DataFrame({"SNP": unmapped_snps}).to_csv(
            out_dir / "unmapped_snps.csv",
            index=False
        )

        pd.DataFrame({"Gene": all_genes}).to_csv(
            out_dir / "all_genes.csv",
            index=False
        )

        mapped_edges.to_csv(
            out_dir / "snp_to_gene_edges.csv",
            index=False
        )

        summary_rows.append({
            "Trait": trait,
            "Split": split_name,
            "n_snps_total": len(split_snps),
            "n_mapped_snps": len(mapped_snps),
            "n_unmapped_snps": len(unmapped_snps),
            "fraction_mapped_snps": round(len(mapped_snps) / max(len(split_snps), 1), 6),
            "n_genes": len(all_genes),
            "n_snp_gene_edges": len(mapped_edges),
            "geno_file": str(geno_file),
            "out_dir": str(out_dir),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_file = reports_dir / f"split_bio_object_summary_{trait}.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"Saved summary: {summary_file}")

    return summary_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Q0 - PREPARE SPLIT BIO INPUTS FOR NEW TRAITS")
    print("=" * 80)

    print("Loading global SNP->gene mapping...")
    check_file(GLOBAL_SNP_GENE_FILE)

    snp_gene = pd.read_csv(GLOBAL_SNP_GENE_FILE, dtype=str)
    snp_gene["SNP"] = snp_gene["SNP"].astype(str).str.strip()
    snp_gene["Gene"] = snp_gene["Gene"].astype(str).str.strip()

    print(f"Global SNP-gene mapping shape: {snp_gene.shape}")
    print(f"Unique mapped SNPs globally: {snp_gene['SNP'].nunique()}")
    print(f"Unique genes globally: {snp_gene['Gene'].nunique()}")

    all_summaries = []

    for trait in TRAITS:
        summary = process_one_trait(trait, snp_gene)
        all_summaries.append(summary)

    global_summary = pd.concat(all_summaries, ignore_index=True)

    global_summary_file = REPORTS_BASE_DIR / "split_bio_object_summary_all_traits.csv"
    global_summary.to_csv(global_summary_file, index=False)

    print("\n" + "=" * 80)
    print("Q0 completed.")
    print("=" * 80)

    print("Saved global summary:")
    print(global_summary_file)

    print("\nSummary preview:")
    print(
        global_summary
        .groupby("Trait")
        .agg(
            n_splits=("Split", "nunique"),
            mean_total_snps=("n_snps_total", "mean"),
            mean_mapped_snps=("n_mapped_snps", "mean"),
            mean_unmapped_snps=("n_unmapped_snps", "mean"),
            mean_genes=("n_genes", "mean"),
            mean_edges=("n_snp_gene_edges", "mean"),
        )
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()