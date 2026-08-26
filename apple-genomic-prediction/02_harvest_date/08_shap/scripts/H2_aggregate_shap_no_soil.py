# -*- coding: utf-8 -*-

################################################################################
### H2_aggregate_shap_no_soil.py
### Aggregate SHAP results for V3 no-soil model
################################################################################

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

TRAIT = "Harvest_date"
MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"

OUT_DIR = (
    Path("02_harvest_date")
    / "08_shap"
    / "output"
)

NN_OUT_DIR = (
    Path("02_harvest_date")
    / "07_neural_network"
    / "output"
)

SPLITWISE_DIR = (
    OUT_DIR
    / "Interpretation"
    / TRAIT
    / "Splitwise_tables"
)

DICT_DIR = (
    OUT_DIR
    / "Interpretation"
    / TRAIT
    / "Feature_dictionaries"
)

AGG_DIR = (
    OUT_DIR
    / "Interpretation"
    / TRAIT
    / "Aggregated_tables"
)

AGG_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_INPUTS_DIR = (
    NN_OUT_DIR
    / "biologic_objects"
    / "split_inputs"
)

WEATHER_DICT_FILE = DICT_DIR / "weather_v3_feature_dictionary.csv"

# Master files
MASTER_WEATHER = AGG_DIR / f"MASTER_weather_SHAP_{MODEL_NAME}_{TRAIT}.csv"
MASTER_PCA = AGG_DIR / f"MASTER_pca_SHAP_{MODEL_NAME}_{TRAIT}.csv"
MASTER_MAPPED = AGG_DIR / f"MASTER_mapped_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
MASTER_UNMAPPED = AGG_DIR / f"MASTER_unmapped_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
MASTER_ALLSNP = AGG_DIR / f"MASTER_all_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
MASTER_BRANCH = AGG_DIR / f"MASTER_branch_importance_{MODEL_NAME}_{TRAIT}.csv"
MASTER_GENE = AGG_DIR / f"MASTER_gene_SHAP_{MODEL_NAME}_{TRAIT}.csv"

# Summary files
SUMMARY_WEATHER = AGG_DIR / f"SUMMARY_weather_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_PCA = AGG_DIR / f"SUMMARY_pca_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_MAPPED = AGG_DIR / f"SUMMARY_mapped_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_UNMAPPED = AGG_DIR / f"SUMMARY_unmapped_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_ALLSNP = AGG_DIR / f"SUMMARY_all_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_BRANCH = AGG_DIR / f"SUMMARY_branch_importance_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_BRANCH_GENOMIC = AGG_DIR / f"SUMMARY_branch_importance_with_snp_total_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_MAPPED_UNMAPPED = AGG_DIR / f"SUMMARY_mapped_vs_unmapped_{MODEL_NAME}_{TRAIT}.csv"

SUMMARY_GENE = AGG_DIR / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_GENE_STABLE5 = AGG_DIR / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{TRAIT}_stable_nfolds_ge5.csv"
SUMMARY_GENE_STABLE10 = AGG_DIR / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{TRAIT}_stable_nfolds_ge10.csv"

SUMMARY_WEATHER_PERIOD = AGG_DIR / f"SUMMARY_weather_by_period_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_WEATHER_VARIABLE = AGG_DIR / f"SUMMARY_weather_by_variable_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_WEATHER_STAT = AGG_DIR / f"SUMMARY_weather_by_stat_{MODEL_NAME}_{TRAIT}.csv"

REPORT_FILE = AGG_DIR / f"aggregation_report_{MODEL_NAME}_{TRAIT}.txt"


# =============================================================================
# HELPERS
# =============================================================================

def load_csvs(pattern: str) -> pd.DataFrame:
    files = sorted(SPLITWISE_DIR.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            f"Nessun file trovato con pattern: {pattern}\nIn: {SPLITWISE_DIR}"
        )

    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def build_summary_feature_level(master_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = master_df.copy()

    if len(df) == 0:
        return pd.DataFrame(columns=[
            id_col,
            "meanSHAP",
            "medianSHAP",
            "sdSHAP",
            "minSHAP",
            "maxSHAP",
            "n_folds",
            "meanRank",
            "medianRank",
            "bestRank",
            "worstRank",
            "top20_count",
            "top50_count",
        ])

    top20_df = df[df["Rank"] <= 20].copy()
    top50_df = df[df["Rank"] <= 50].copy()

    top20_counts = top20_df.groupby(id_col).size().rename("top20_count").reset_index()
    top50_counts = top50_df.groupby(id_col).size().rename("top50_count").reset_index()

    summary_df = (
        df.groupby(id_col)
        .agg(
            meanSHAP=("mean_abs_SHAP", "mean"),
            medianSHAP=("mean_abs_SHAP", "median"),
            sdSHAP=("mean_abs_SHAP", "std"),
            minSHAP=("mean_abs_SHAP", "min"),
            maxSHAP=("mean_abs_SHAP", "max"),
            n_folds=("mean_abs_SHAP", "size"),
            meanRank=("Rank", "mean"),
            medianRank=("Rank", "median"),
            bestRank=("Rank", "min"),
            worstRank=("Rank", "max"),
        )
        .reset_index()
    )

    summary_df = summary_df.merge(top20_counts, on=id_col, how="left")
    summary_df = summary_df.merge(top50_counts, on=id_col, how="left")

    summary_df["top20_count"] = summary_df["top20_count"].fillna(0).astype(int)
    summary_df["top50_count"] = summary_df["top50_count"].fillna(0).astype(int)
    summary_df["sdSHAP"] = summary_df["sdSHAP"].fillna(0.0)

    summary_df = summary_df.sort_values(
        ["meanSHAP", "top20_count", "top50_count", "meanRank"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    return summary_df


def build_gene_master_from_mapped(master_mapped: pd.DataFrame) -> pd.DataFrame:
    gene_rows = []

    split_dirs = sorted(SPLIT_INPUTS_DIR.glob("CV*_Split*"))

    if len(split_dirs) == 0:
        print(f"ATTENZIONE: nessuna cartella split trovata in {SPLIT_INPUTS_DIR}")

    for split_dir in split_dirs:
        split_name = split_dir.name
        edge_file = split_dir / "snp_to_gene_edges.csv"

        if not edge_file.exists():
            print(f"ATTENZIONE: edge file mancante per {split_name}: {edge_file}")
            continue

        edges = pd.read_csv(edge_file, dtype=str)

        if not {"SNP", "Gene"}.issubset(edges.columns):
            raise ValueError(f"Il file {edge_file} deve contenere almeno le colonne SNP e Gene")

        edges["SNP"] = edges["SNP"].astype(str)
        edges["Gene"] = edges["Gene"].astype(str)

        split_shap = master_mapped[master_mapped["Split"] == split_name].copy()

        if len(split_shap) == 0:
            continue

        split_shap["SNP"] = split_shap["SNP"].astype(str)

        merged = split_shap.merge(edges, on="SNP", how="left")
        merged = merged.dropna(subset=["Gene"]).copy()

        if len(merged) == 0:
            continue

        gene_split = (
            merged.groupby(["Split", "Gene"])
            .agg(
                gene_mean_abs_SHAP=("mean_abs_SHAP", "sum"),
                n_snps_in_gene_in_split=("SNP", "nunique")
            )
            .reset_index()
        )

        gene_split = gene_split.sort_values(
            "gene_mean_abs_SHAP",
            ascending=False
        ).reset_index(drop=True)

        gene_split["Rank"] = np.arange(1, len(gene_split) + 1)

        gene_rows.append(gene_split)

    if len(gene_rows) == 0:
        return pd.DataFrame(columns=[
            "Split",
            "Gene",
            "gene_mean_abs_SHAP",
            "n_snps_in_gene_in_split",
            "Rank",
        ])

    return pd.concat(gene_rows, ignore_index=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("H2 - AGGREGATE SHAP NO-SOIL")
    print("=" * 80)

    print("Caricamento tabelle split-wise...")

    master_weather = load_csvs("*_weather_mean_abs_SHAP.csv")
    master_pca = load_csvs("*_pca_mean_abs_SHAP.csv")
    master_mapped = load_csvs("*_mapped_snp_mean_abs_SHAP.csv")
    master_unmapped = load_csvs("*_unmapped_snp_mean_abs_SHAP.csv")
    master_allsnp = load_csvs("*_all_snp_mean_abs_SHAP.csv")
    master_branch = load_csvs("*_branch_importance.csv")

    print("Caricate:")
    print("  weather:", master_weather.shape)
    print("  pca:", master_pca.shape)
    print("  mapped:", master_mapped.shape)
    print("  unmapped:", master_unmapped.shape)
    print("  all snp:", master_allsnp.shape)
    print("  branch:", master_branch.shape)

    # -------------------------------------------------------------------------
    # Weather dictionary
    # -------------------------------------------------------------------------
    if not WEATHER_DICT_FILE.exists():
        raise FileNotFoundError(
            f"Dizionario weather non trovato:\n{WEATHER_DICT_FILE}\n"
            "Esegui prima H1b_build_weather_feature_dictionary.py"
        )

    weather_dict = pd.read_csv(WEATHER_DICT_FILE)

    required_weather_cols = {
        "Feature_Code",
        "Original_Feature_Name",
        "Period",
        "Variable",
        "Statistic",
    }

    if not required_weather_cols.issubset(weather_dict.columns):
        raise ValueError(
            f"Il dizionario weather deve contenere {sorted(required_weather_cols)}.\n"
            f"Colonne trovate: {weather_dict.columns.tolist()}"
        )

    master_weather = master_weather.merge(
        weather_dict,
        left_on="Feature",
        right_on="Feature_Code",
        how="left"
    )

    master_weather["Original_Feature_Name"] = master_weather["Original_Feature_Name"].fillna(
        master_weather["Feature"]
    )
    master_weather["Period"] = master_weather["Period"].fillna("Unknown")
    master_weather["Variable"] = master_weather["Variable"].fillna("Unknown")
    master_weather["Statistic"] = master_weather["Statistic"].fillna("Unknown")

    # -------------------------------------------------------------------------
    # Save master files
    # -------------------------------------------------------------------------
    print("Salvataggio master files...")

    master_weather.to_csv(MASTER_WEATHER, index=False)
    master_pca.to_csv(MASTER_PCA, index=False)
    master_mapped.to_csv(MASTER_MAPPED, index=False)
    master_unmapped.to_csv(MASTER_UNMAPPED, index=False)
    master_allsnp.to_csv(MASTER_ALLSNP, index=False)
    master_branch.to_csv(MASTER_BRANCH, index=False)

    # -------------------------------------------------------------------------
    # Summary feature-level
    # -------------------------------------------------------------------------
    print("Costruzione summary feature-level...")

    summary_weather = build_summary_feature_level(master_weather, "Original_Feature_Name")
    summary_weather = summary_weather.rename(columns={"Original_Feature_Name": "Feature"})

    weather_meta = (
        master_weather[["Original_Feature_Name", "Period", "Variable", "Statistic"]]
        .drop_duplicates()
        .rename(columns={"Original_Feature_Name": "Feature"})
    )

    summary_weather = summary_weather.merge(weather_meta, on="Feature", how="left")

    summary_pca = build_summary_feature_level(master_pca, "Feature")
    summary_mapped = build_summary_feature_level(master_mapped, "SNP")
    summary_unmapped = build_summary_feature_level(master_unmapped, "SNP")
    summary_allsnp = build_summary_feature_level(master_allsnp, "SNP")

    summary_weather["Trait"] = TRAIT
    summary_pca["Trait"] = TRAIT
    summary_mapped["Trait"] = TRAIT
    summary_unmapped["Trait"] = TRAIT
    summary_allsnp["Trait"] = TRAIT

    summary_weather.to_csv(SUMMARY_WEATHER, index=False)
    summary_pca.to_csv(SUMMARY_PCA, index=False)
    summary_mapped.to_csv(SUMMARY_MAPPED, index=False)
    summary_unmapped.to_csv(SUMMARY_UNMAPPED, index=False)
    summary_allsnp.to_csv(SUMMARY_ALLSNP, index=False)

    # -------------------------------------------------------------------------
    # Branch importance
    # -------------------------------------------------------------------------
    print("Summary branch importance...")

    summary_branch = (
        master_branch.groupby("Branch")
        .agg(
            meanSHAP=("mean_abs_SHAP", "mean"),
            medianSHAP=("mean_abs_SHAP", "median"),
            sdSHAP=("mean_abs_SHAP", "std"),
            minSHAP=("mean_abs_SHAP", "min"),
            maxSHAP=("mean_abs_SHAP", "max"),
            n_splits=("mean_abs_SHAP", "size"),
        )
        .reset_index()
        .sort_values("meanSHAP", ascending=False)
        .reset_index(drop=True)
    )

    summary_branch["sdSHAP"] = summary_branch["sdSHAP"].fillna(0.0)
    summary_branch["Trait"] = TRAIT
    summary_branch.to_csv(SUMMARY_BRANCH, index=False)

    mapped_mask = summary_branch["Branch"] == "Mapped_SNP"
    unmapped_mask = summary_branch["Branch"] == "Unmapped_SNP"

    mapped_branch_mean = summary_branch.loc[mapped_mask, "meanSHAP"].iloc[0] if mapped_mask.any() else 0.0
    unmapped_branch_mean = summary_branch.loc[unmapped_mask, "meanSHAP"].iloc[0] if unmapped_mask.any() else 0.0

    summary_branch_genomic = summary_branch.copy()

    extra = pd.DataFrame([{
        "Branch": "SNP_total",
        "meanSHAP": mapped_branch_mean + unmapped_branch_mean,
        "medianSHAP": np.nan,
        "sdSHAP": np.nan,
        "minSHAP": np.nan,
        "maxSHAP": np.nan,
        "n_splits": summary_branch["n_splits"].max() if len(summary_branch) > 0 else np.nan,
        "Trait": TRAIT,
    }])

    summary_branch_genomic = pd.concat(
        [summary_branch_genomic, extra],
        ignore_index=True
    )

    summary_branch_genomic.to_csv(SUMMARY_BRANCH_GENOMIC, index=False)

    # -------------------------------------------------------------------------
    # Mapped vs unmapped
    # -------------------------------------------------------------------------
    print("Summary mapped vs unmapped...")

    mapped_vs_unmapped = pd.DataFrame({
        "Trait": [TRAIT, TRAIT],
        "Category": ["mapped", "unmapped"],
        "meanSHAP": [
            master_mapped["mean_abs_SHAP"].mean() if len(master_mapped) > 0 else np.nan,
            master_unmapped["mean_abs_SHAP"].mean() if len(master_unmapped) > 0 else np.nan,
        ],
        "medianSHAP": [
            master_mapped["mean_abs_SHAP"].median() if len(master_mapped) > 0 else np.nan,
            master_unmapped["mean_abs_SHAP"].median() if len(master_unmapped) > 0 else np.nan,
        ],
        "sdSHAP": [
            master_mapped["mean_abs_SHAP"].std(ddof=1) if len(master_mapped) > 1 else 0.0,
            master_unmapped["mean_abs_SHAP"].std(ddof=1) if len(master_unmapped) > 1 else 0.0,
        ],
        "n_rows": [
            len(master_mapped),
            len(master_unmapped),
        ],
        "n_unique_features": [
            master_mapped["SNP"].nunique() if "SNP" in master_mapped.columns else 0,
            master_unmapped["SNP"].nunique() if "SNP" in master_unmapped.columns else 0,
        ],
    })

    mapped_vs_unmapped.to_csv(SUMMARY_MAPPED_UNMAPPED, index=False)

    # -------------------------------------------------------------------------
    # Gene-level
    # -------------------------------------------------------------------------
    print("Summary gene-level...")

    master_gene = build_gene_master_from_mapped(master_mapped)

    if len(master_gene) > 0:
        master_gene["Trait"] = TRAIT
    else:
        master_gene = pd.DataFrame(columns=[
            "Split",
            "Gene",
            "gene_mean_abs_SHAP",
            "n_snps_in_gene_in_split",
            "Rank",
            "Trait",
        ])

    master_gene.to_csv(MASTER_GENE, index=False)

    if len(master_gene) > 0:
        top20_df = master_gene[master_gene["Rank"] <= 20].copy()
        top50_df = master_gene[master_gene["Rank"] <= 50].copy()

        top20_counts = top20_df.groupby("Gene").size().rename("top20_count").reset_index()
        top50_counts = top50_df.groupby("Gene").size().rename("top50_count").reset_index()

        summary_gene = (
            master_gene.groupby("Gene")
            .agg(
                meanSHAP=("gene_mean_abs_SHAP", "mean"),
                medianSHAP=("gene_mean_abs_SHAP", "median"),
                sdSHAP=("gene_mean_abs_SHAP", "std"),
                n_folds=("gene_mean_abs_SHAP", "size"),
                meanRank=("Rank", "mean"),
                medianRank=("Rank", "median"),
                bestRank=("Rank", "min"),
                mean_n_snps_in_gene=("n_snps_in_gene_in_split", "mean"),
            )
            .reset_index()
        )

        summary_gene = summary_gene.merge(top20_counts, on="Gene", how="left")
        summary_gene = summary_gene.merge(top50_counts, on="Gene", how="left")

        summary_gene["top20_count"] = summary_gene["top20_count"].fillna(0).astype(int)
        summary_gene["top50_count"] = summary_gene["top50_count"].fillna(0).astype(int)
        summary_gene["sdSHAP"] = summary_gene["sdSHAP"].fillna(0.0)
        summary_gene["Trait"] = TRAIT

        summary_gene = summary_gene.sort_values(
            ["meanSHAP", "top20_count", "top50_count", "meanRank"],
            ascending=[False, False, False, True]
        ).reset_index(drop=True)
    else:
        summary_gene = pd.DataFrame(columns=[
            "Gene",
            "meanSHAP",
            "medianSHAP",
            "sdSHAP",
            "n_folds",
            "meanRank",
            "medianRank",
            "bestRank",
            "mean_n_snps_in_gene",
            "top20_count",
            "top50_count",
            "Trait",
        ])

    summary_gene.to_csv(SUMMARY_GENE, index=False)
    summary_gene[summary_gene["n_folds"] >= 5].to_csv(SUMMARY_GENE_STABLE5, index=False)
    summary_gene[summary_gene["n_folds"] >= 10].to_csv(SUMMARY_GENE_STABLE10, index=False)

    # -------------------------------------------------------------------------
    # Weather by period / variable / stat
    # -------------------------------------------------------------------------
    print("Summary weather by period / variable / stat...")

    weather_aux = summary_weather.copy()

    weather_period_summary = (
        weather_aux.groupby("Period")
        .agg(
            meanSHAP=("meanSHAP", "mean"),
            medianSHAP=("meanSHAP", "median"),
            n_features=("Feature", "size"),
        )
        .reset_index()
        .sort_values("meanSHAP", ascending=False)
        .reset_index(drop=True)
    )
    weather_period_summary["Trait"] = TRAIT
    weather_period_summary.to_csv(SUMMARY_WEATHER_PERIOD, index=False)

    weather_variable_summary = (
        weather_aux.groupby("Variable")
        .agg(
            meanSHAP=("meanSHAP", "mean"),
            medianSHAP=("meanSHAP", "median"),
            n_features=("Feature", "size"),
        )
        .reset_index()
        .sort_values("meanSHAP", ascending=False)
        .reset_index(drop=True)
    )
    weather_variable_summary["Trait"] = TRAIT
    weather_variable_summary.to_csv(SUMMARY_WEATHER_VARIABLE, index=False)

    weather_stat_summary = (
        weather_aux.groupby("Statistic")
        .agg(
            meanSHAP=("meanSHAP", "mean"),
            medianSHAP=("meanSHAP", "median"),
            n_features=("Feature", "size"),
        )
        .reset_index()
        .sort_values("meanSHAP", ascending=False)
        .reset_index(drop=True)
    )
    weather_stat_summary["Trait"] = TRAIT
    weather_stat_summary.to_csv(SUMMARY_WEATHER_STAT, index=False)

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    print("Scrittura report...")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== SHAP AGGREGATION REPORT FOR V3 NO-SOIL ===\n\n")

        f.write(f"MODEL_NAME: {MODEL_NAME}\n")
        f.write(f"TRAIT: {TRAIT}\n")
        f.write("Soil branch: REMOVED\n\n")

        f.write("Branch importance summary:\n")
        f.write(summary_branch.to_string(index=False))
        f.write("\n\n")

        f.write("Branch importance with SNP_total:\n")
        f.write(summary_branch_genomic.to_string(index=False))
        f.write("\n\n")

        f.write("Mapped vs unmapped summary:\n")
        f.write(mapped_vs_unmapped.to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 SNPs overall:\n")
        f.write(summary_allsnp.head(20).to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 mapped SNPs overall:\n")
        f.write(summary_mapped.head(20).to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 unmapped SNPs overall:\n")
        f.write(summary_unmapped.head(20).to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 genes overall:\n")
        f.write(summary_gene.head(20).to_string(index=False))
        f.write("\n\n")

        f.write("Top stable genes (n_folds >= 5):\n")
        f.write(summary_gene[summary_gene["n_folds"] >= 5].head(20).to_string(index=False))
        f.write("\n\n")

        f.write("Weather by period:\n")
        f.write(weather_period_summary.to_string(index=False))
        f.write("\n\n")

        f.write("Weather by variable:\n")
        f.write(weather_variable_summary.to_string(index=False))
        f.write("\n\n")

        f.write("Weather by statistic:\n")
        f.write(weather_stat_summary.to_string(index=False))
        f.write("\n\n")

    print("Salvato tutto in:")
    print(AGG_DIR)


if __name__ == "__main__":
    main()
