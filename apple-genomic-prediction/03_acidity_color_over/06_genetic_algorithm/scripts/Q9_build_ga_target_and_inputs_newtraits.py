################################################################################
### Q9_build_ga_target_and_inputs_newtraits.py
###
### Build GA-ready SNP matrix, PCA matrix, target and metadata
### using top1000 50kb regions ranked from NO-SOIL SHAP for:
###   - Acidity
###   - Color_over
################################################################################

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

TOP_K = 1000
WINDOW_LABEL = "50kb"

GA_OUT_DIR = (
    Path("03_acidity_color_over")
    / "06_genetic_algorithm"
    / "output"
)

ANNOT_BASE_DIR = GA_OUT_DIR / "02_region_annotations"
RANK_BASE_DIR = GA_OUT_DIR / "01_region_ranking"
GA_BASE_DIR = GA_OUT_DIR / "03_ga_inputs"

GA_BASE_DIR.mkdir(parents=True, exist_ok=True)

ALL_GENO_FILE = (
    Path("03_acidity_color_over")
    / "03_gradient_boosting"
    / "output"
    / "all.geno"
)

PCA_FILE = (
    Path("01_common_genomic_preprocessing")
    / "output"
    / "genomic_PCs_20_paper_style.csv"
)

PHENO_BASE_DIR = (
    Path("03_acidity_color_over")
    / "02_phenotype_preprocessing"
    / "output"
)


# =============================================================================
# HELPERS
# =============================================================================

def check_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} non trovato:\n{path}")


def clean_genotype_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("^G_", "", regex=True)
        .str.strip()
    )


def detect_genotype_column(df: pd.DataFrame):
    candidates = ["Genotype", "genotype", "Sample", "sample", "IID", "iid"]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "Nessuna colonna Genotype trovata.\n"
        f"Colonne disponibili: {df.columns.tolist()[:50]}"
    )


def load_pca() -> pd.DataFrame:
    check_file(PCA_FILE, "PCA file")

    pca = pd.read_csv(PCA_FILE)

    geno_col = detect_genotype_column(pca)
    pca = pca.rename(columns={geno_col: "Genotype"})
    pca["Genotype"] = clean_genotype_series(pca["Genotype"])

    pc_cols = [c for c in pca.columns if c.startswith("PC")]

    if len(pc_cols) == 0:
        raise ValueError(
            f"Nessuna colonna PC trovata in {PCA_FILE}.\n"
            f"Colonne: {pca.columns.tolist()}"
        )

    for c in pc_cols:
        pca[c] = pd.to_numeric(pca[c], errors="coerce")

    if pca["Genotype"].duplicated().any():
        dup = pca.loc[pca["Genotype"].duplicated(), "Genotype"].head(20).tolist()
        raise ValueError(f"Genotipi duplicati nel file PCA. Esempi: {dup}")

    return pca[["Genotype"] + pc_cols].copy()


def load_trait_pheno(trait: str) -> pd.DataFrame:
    pheno_file = PHENO_BASE_DIR / trait / f"{trait}_processed_final.csv"
    check_file(pheno_file, f"Phenotype processed file for {trait}")

    pheno = pd.read_csv(pheno_file)

    needed = {"Genotype", "Envir", trait}
    missing = needed - set(pheno.columns)

    if missing:
        raise ValueError(
            f"Nel file {pheno_file} mancano colonne: {missing}\n"
            f"Colonne trovate: {pheno.columns.tolist()}"
        )

    pheno = pheno[["Genotype", "Envir", trait]].copy()
    pheno["Genotype"] = clean_genotype_series(pheno["Genotype"])
    pheno["Envir"] = pheno["Envir"].astype(str).str.strip()
    pheno[trait] = pd.to_numeric(pheno[trait], errors="coerce")

    pheno = pheno.dropna(subset=[trait]).reset_index(drop=True)

    if pheno.empty:
        raise ValueError(f"Nessuna riga valida per {trait}.")

    return pheno


def build_genotype_level_target(pheno: pd.DataFrame, trait: str) -> pd.DataFrame:
    """
    Per il GA facciamo un target genotype-level:
    media del trait processato sulle environment disponibili per quel genotipo.

    Per Acidity avremo probabilmente 533 genotipi.
    Per Color_over probabilmente 534 genotipi.
    Questo è normale, perché alcuni tratti possono avere un genotipo mancante
    dopo preprocessing/filtro fenotipico.
    """
    target = (
        pheno.groupby("Genotype", as_index=False)
        .agg(
            trait_mean=(trait, "mean"),
            trait_median=(trait, "median"),
            trait_sd=(trait, "std"),
            n_env_obs=(trait, "size"),
            environments=("Envir", lambda x: ";".join(sorted(pd.Series(x).astype(str).unique()))),
        )
    )

    target = target.rename(columns={"trait_mean": f"{trait}_mean"})
    target["trait_sd"] = target["trait_sd"].fillna(0.0)

    return target


def get_top_regions_file(trait: str) -> Path:
    path = (
        ANNOT_BASE_DIR
        / trait
        / f"top{TOP_K}_regions_by_region_score_annotated_{WINDOW_LABEL}_{trait}.csv"
    )

    check_file(path, f"Top annotated regions file for {trait}")
    return path


def get_membership_file(trait: str) -> Path:
    """
    Prova vari nomi possibili, perché il file membership può essere stato salvato
    in modo leggermente diverso da Q7.
    """

    candidates = [
        RANK_BASE_DIR / trait / f"region_snp_membership_{WINDOW_LABEL}_{trait}.csv",
        RANK_BASE_DIR / trait / f"snp_region_membership_{WINDOW_LABEL}_{trait}.csv",
        RANK_BASE_DIR / trait / f"membership_regions_{WINDOW_LABEL}_{trait}.csv",
        RANK_BASE_DIR / f"region_snp_membership_{WINDOW_LABEL}_{trait}.csv",
        RANK_BASE_DIR / f"snp_region_membership_{WINDOW_LABEL}_{trait}.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    # fallback: search for any 50 KB “membership” file in the trait
    search_patterns = [
        f"*membership*{WINDOW_LABEL}*{trait}*.csv",
        f"*region*snp*{WINDOW_LABEL}*{trait}*.csv",
        f"*snp*region*{WINDOW_LABEL}*{trait}*.csv",
    ]

    found = []
    for pat in search_patterns:
        found.extend(list(RANK_BASE_DIR.rglob(pat)))

    found = sorted(set(found))

    if len(found) > 0:
        print(f"[INFO] Membership file trovato automaticamente per {trait}:")
        print(found[0])
        return found[0]

    raise FileNotFoundError(
        f"Membership file non trovato per {trait}.\n"
        f"Ho provato questi path:\n" +
        "\n".join(str(x) for x in candidates)
    )


def standardize_allgeno_header():
    check_file(ALL_GENO_FILE, "all.geno")

    header = pd.read_csv(ALL_GENO_FILE, nrows=0)
    cols = header.columns.tolist()

    geno_col = None
    for c in ["Genotype", "genotype"]:
        if c in cols:
            geno_col = c
            break

    if geno_col is None:
        raise ValueError(
            f"Nessuna colonna Genotype trovata in all.geno.\n"
            f"Prime colonne: {cols[:20]}"
        )

    snp_cols = [c for c in cols if c != geno_col]

    return geno_col, snp_cols


def load_filtered_allgeno(available_snps):
    geno_col, _ = standardize_allgeno_header()

    usecols = [geno_col] + available_snps

    geno = pd.read_csv(ALL_GENO_FILE, usecols=usecols)
    geno = geno.rename(columns={geno_col: "Genotype"})
    geno["Genotype"] = clean_genotype_series(geno["Genotype"])

    snp_cols = [c for c in geno.columns if c != "Genotype"]
    geno[snp_cols] = geno[snp_cols].apply(pd.to_numeric, errors="coerce")

    return geno


def ensure_region_id(df: pd.DataFrame, file_label: str) -> pd.DataFrame:
    df = df.copy()

    if "region_id" in df.columns:
        df["region_id"] = df["region_id"].astype(str)
        return df

    if "region" in df.columns:
        df = df.rename(columns={"region": "region_id"})
        df["region_id"] = df["region_id"].astype(str)
        return df

    raise ValueError(
        f"Nel file {file_label} non trovo né region_id né region.\n"
        f"Colonne disponibili: {df.columns.tolist()}"
    )


def load_top_regions(trait: str) -> pd.DataFrame:
    f = get_top_regions_file(trait)
    top = pd.read_csv(f)
    top = ensure_region_id(top, str(f))

    if top["region_id"].nunique() != TOP_K:
        print(
            f"[WARNING] {trait}: top regions unique = {top['region_id'].nunique()}, "
            f"atteso {TOP_K}"
        )

    return top


def load_membership(trait: str) -> pd.DataFrame:
    f = get_membership_file(trait)
    membership = pd.read_csv(f)

    needed = {"SNP", "region_id"}

    if not needed.issubset(membership.columns):
        if "region" in membership.columns and "region_id" not in membership.columns:
            membership = membership.rename(columns={"region": "region_id"})

    needed = {"SNP", "region_id"}
    missing = needed - set(membership.columns)

    if missing:
        raise ValueError(
            f"Nel membership file mancano colonne: {missing}\n"
            f"File: {f}\n"
            f"Colonne trovate: {membership.columns.tolist()}"
        )

    membership["SNP"] = membership["SNP"].astype(str)
    membership["region_id"] = membership["region_id"].astype(str)

    return membership


def process_trait(trait: str, pca: pd.DataFrame):
    print("\n" + "=" * 100)
    print(f"Q9 - BUILD GA INPUTS | TRAIT: {trait}")
    print("=" * 100)

    trait_ga_dir = GA_BASE_DIR / trait
    trait_ga_dir.mkdir(parents=True, exist_ok=True)

    save_target = trait_ga_dir / f"y_mean_by_genotype_{trait}.csv"
    save_x_snp = trait_ga_dir / f"X_snp_top{TOP_K}_regions_{WINDOW_LABEL}_{trait}.csv"
    save_x_pca = trait_ga_dir / f"X_pca_20_{trait}.csv"
    save_x_full = trait_ga_dir / f"X_snp_plus_pca_top{TOP_K}_regions_{WINDOW_LABEL}_{trait}.csv"
    save_snp_meta = trait_ga_dir / f"snp_metadata_top{TOP_K}_regions_{WINDOW_LABEL}_{trait}.csv"
    save_region_meta = trait_ga_dir / f"region_metadata_top{TOP_K}_{WINDOW_LABEL}_{trait}.csv"
    save_report = trait_ga_dir / f"Q9_build_ga_inputs_report_{trait}.txt"

    # -------------------------------------------------------------------------
    # Top regions + membership
    # -------------------------------------------------------------------------
    print("Loading top annotated regions...")
    top_regions = load_top_regions(trait)
    top_region_ids = set(top_regions["region_id"].astype(str))

    print(f"Top regions unique: {len(top_region_ids)}")

    print("Loading SNP-region membership...")
    membership = load_membership(trait)

    membership_top = membership[membership["region_id"].isin(top_region_ids)].copy()

    if membership_top.empty:
        raise ValueError(
            f"{trait}: membership_top è vuoto. "
            "Probabile mismatch tra region_id del ranking e region_id del membership."
        )

    candidate_snps = sorted(membership_top["SNP"].astype(str).unique().tolist())

    print(f"Candidate SNPs from top regions: {len(candidate_snps)}")

    # -------------------------------------------------------------------------
    # all.geno available SNPs
    # -------------------------------------------------------------------------
    print("Checking SNPs available in all.geno...")
    geno_col, allgeno_snps = standardize_allgeno_header()
    allgeno_snp_set = set(allgeno_snps)

    available_snps = [s for s in candidate_snps if s in allgeno_snp_set]
    missing_snps = sorted(set(candidate_snps) - set(available_snps))

    print(f"Available SNPs in all.geno: {len(available_snps)}")
    print(f"Missing SNPs in all.geno: {len(missing_snps)}")

    if len(available_snps) == 0:
        raise ValueError(f"{trait}: nessuno SNP candidato è presente in all.geno.")

    print("Reading filtered all.geno...")
    geno = load_filtered_allgeno(available_snps)

    # -------------------------------------------------------------------------
    # Target genotype-level
    # -------------------------------------------------------------------------
    print("Building genotype-level target...")
    pheno = load_trait_pheno(trait)
    target = build_genotype_level_target(pheno, trait)

    print(f"Phenotype rows: {pheno.shape[0]}")
    print(f"Phenotype unique genotypes: {pheno['Genotype'].nunique()}")
    print(f"Target genotype rows: {target.shape[0]}")

    # -------------------------------------------------------------------------
    # Merge genotype + target + PCA
    # -------------------------------------------------------------------------
    print("Merging genotype, target and PCA...")

    merged = geno.merge(target, on="Genotype", how="inner")
    merged = merged.merge(pca, on="Genotype", how="inner")

    pc_cols = [c for c in pca.columns if c != "Genotype"]

    print(f"Final merged genotypes: {merged['Genotype'].nunique()}")
    print(f"Final SNP feature count: {len(available_snps)}")
    print(f"Final PCA feature count: {len(pc_cols)}")

    if merged.empty:
        raise ValueError(f"{trait}: merge finale vuoto.")

    # -------------------------------------------------------------------------
    # SNP metadata
    # -------------------------------------------------------------------------
    print("Building SNP metadata...")

    top_region_cols_preferred = [
        "region_id",
        "window_bp",
        "CHROM",
        "region_start",
        "region_end",
        "region_score",
        "rank_by_region_score",
        "mean_region_SHAP",
        "median_region_SHAP",
        "max_region_SHAP",
        "sum_region_SHAP",
        "mean_n_folds",
        "max_n_folds",
        "mean_top20_count",
        "max_top20_count",
        "mean_top50_count",
        "max_top50_count",
        "n_snps",
        "n_mapped_snps",
        "n_unmapped_snps",
        "n_genes_inside",
        "genes_inside",
        "n_genes_nearby_10kb",
        "genes_nearby_10kb",
    ]

    top_region_cols = [c for c in top_region_cols_preferred if c in top_regions.columns]

    snp_meta = membership_top[membership_top["SNP"].isin(available_snps)].copy()

    merge_cols = [
        c for c in top_region_cols
        if c == "region_id" or c not in snp_meta.columns
    ]

    snp_meta = snp_meta.merge(
        top_regions[merge_cols].drop_duplicates(subset=["region_id"]),
        on="region_id",
        how="left"
    )

    snp_meta = snp_meta.drop_duplicates().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    print("Saving GA input files...")

    y = merged[["Genotype", f"{trait}_mean", "trait_median", "trait_sd", "n_env_obs", "environments"]].copy()
    y.to_csv(save_target, index=False)

    X_snp = merged[["Genotype"] + available_snps].copy()
    X_snp.to_csv(save_x_snp, index=False)

    X_pca = merged[["Genotype"] + pc_cols].copy()
    X_pca.to_csv(save_x_pca, index=False)

    X_full = merged[["Genotype"] + available_snps + pc_cols].copy()
    X_full.to_csv(save_x_full, index=False)

    snp_meta.to_csv(save_snp_meta, index=False)
    top_regions.to_csv(save_region_meta, index=False)

    with open(save_report, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"Q9 BUILD GA INPUTS REPORT: {trait}\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"TRAIT: {trait}\n")
        f.write(f"TOP_K: {TOP_K}\n")
        f.write(f"WINDOW_LABEL: {WINDOW_LABEL}\n\n")

        f.write(f"Top regions unique: {len(top_region_ids)}\n")
        f.write(f"Membership rows total: {membership.shape[0]}\n")
        f.write(f"Membership rows in top regions: {membership_top.shape[0]}\n")
        f.write(f"Candidate SNPs from top regions: {len(candidate_snps)}\n")
        f.write(f"Candidate SNPs present in all.geno: {len(available_snps)}\n")
        f.write(f"Missing candidate SNPs in all.geno: {len(missing_snps)}\n\n")

        f.write(f"Phenotype rows: {pheno.shape[0]}\n")
        f.write(f"Phenotype unique genotypes: {pheno['Genotype'].nunique()}\n")
        f.write(f"Target genotype rows: {target.shape[0]}\n")
        f.write(f"PCA genotype rows: {pca.shape[0]}\n")
        f.write(f"all.geno genotype rows: {geno.shape[0]}\n")
        f.write(f"Final merged genotypes: {merged['Genotype'].nunique()}\n\n")

        f.write(f"Final SNP feature count: {len(available_snps)}\n")
        f.write(f"Final PCA feature count: {len(pc_cols)}\n\n")

        f.write("Target summary:\n")
        f.write(y[f"{trait}_mean"].describe().to_string())
        f.write("\n\n")

        f.write("n_env_obs summary:\n")
        f.write(y["n_env_obs"].describe().to_string())
        f.write("\n\n")

        f.write("SNP per region summary:\n")
        f.write(snp_meta.groupby("region_id")["SNP"].nunique().describe().to_string())
        f.write("\n\n")

        if len(missing_snps) > 0:
            f.write("Missing SNPs in all.geno, first 100:\n")
            f.write("\n".join(missing_snps[:100]))
            f.write("\n\n")

        f.write("Output files:\n")
        f.write(f"- {save_target}\n")
        f.write(f"- {save_x_snp}\n")
        f.write(f"- {save_x_pca}\n")
        f.write(f"- {save_x_full}\n")
        f.write(f"- {save_snp_meta}\n")
        f.write(f"- {save_region_meta}\n")

    print("Saved:")
    print(save_target)
    print(save_x_snp)
    print(save_x_pca)
    print(save_x_full)
    print(save_snp_meta)
    print(save_region_meta)
    print(save_report)

    return {
        "Trait": trait,
        "n_top_regions": len(top_region_ids),
        "n_candidate_snps": len(candidate_snps),
        "n_available_snps": len(available_snps),
        "n_missing_snps": len(missing_snps),
        "n_pheno_rows": pheno.shape[0],
        "n_target_genotypes": target.shape[0],
        "n_final_genotypes": merged["Genotype"].nunique(),
        "n_pca_features": len(pc_cols),
        "target_file": str(save_target),
        "x_snp_file": str(save_x_snp),
        "x_pca_file": str(save_x_pca),
        "x_full_file": str(save_x_full),
        "snp_metadata_file": str(save_snp_meta),
        "region_metadata_file": str(save_region_meta),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("Q9 - BUILD GA INPUTS FOR NEW TRAITS")
    print("=" * 100)

    check_file(ALL_GENO_FILE, "all.geno")
    check_file(PCA_FILE, "PCA file")

    pca = load_pca()

    print(f"[INFO] PCA loaded: {pca.shape}")

    rows = []

    for trait in TRAITS:
        rows.append(process_trait(trait, pca))

    summary = pd.DataFrame(rows)

    summary_file = GA_BASE_DIR / "Q9_ga_inputs_summary_all_traits.csv"
    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 100)
    print("Q9 completed.")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("\nSaved global summary:")
    print(summary_file)


if __name__ == "__main__":
    main()
