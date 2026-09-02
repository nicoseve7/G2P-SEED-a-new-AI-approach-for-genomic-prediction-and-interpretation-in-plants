# -*- coding: utf-8 -*-

################################################################################
### Q5_aggregate_shap_no_soil_newtraits.py
###
### Aggregate SHAP results for V3 no-soil model for:
###   - Acidity
###   - Color_over
###
### Input:
###   Output/02_no_soil_model/<TRAIT>/Interpretation/Splitwise_tables/
###
### Output:
###   Output/02_no_soil_model/<TRAIT>/Interpretation/Aggregated_tables/
################################################################################

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"

BASE_MODEL_DIR = Path("Output/02_no_soil_model")
BIO_BASE_DIR = Path("Output/biologic_objects")


# =============================================================================
# HELPERS
# =============================================================================

def natural_split_sort_key(split_name: str):
    try:
        cv_part, split_part = str(split_name).split("_")
        cv_num = int(cv_part.replace("CV", ""))
        split_num = int(split_part.replace("Split", ""))
        return cv_num, split_num
    except Exception:
        return 999, 999


def check_dir(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cartella non trovata:\n{path}")


def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File non trovato:\n{path}")


def load_csvs(splitwise_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(splitwise_dir.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            f"Nessun file trovato con pattern: {pattern}\n"
            f"In: {splitwise_dir}"
        )

    dfs = []

    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    if "Split" in out.columns:
        out["Split"] = out["Split"].astype(str)

    return out


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

    df["mean_abs_SHAP"] = pd.to_numeric(df["mean_abs_SHAP"], errors="coerce")
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")

    top20_df = df[df["Rank"] <= 20].copy()
    top50_df = df[df["Rank"] <= 50].copy()

    top20_counts = (
        top20_df.groupby(id_col)
        .size()
        .rename("top20_count")
        .reset_index()
    )

    top50_counts = (
        top50_df.groupby(id_col)
        .size()
        .rename("top50_count")
        .reset_index()
    )

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

    summary_df["GlobalRank"] = np.arange(1, len(summary_df) + 1)

    return summary_df


def attach_weather_dictionary(master_weather: pd.DataFrame, weather_dict_file: Path):
    check_file(weather_dict_file)

    weather_dict = pd.read_csv(weather_dict_file)

    required = {
        "Feature_Code",
        "Original_Feature_Name",
        "Period",
        "Variable",
        "Statistic",
    }

    missing = required - set(weather_dict.columns)

    if missing:
        raise ValueError(
            f"Il dizionario weather non contiene le colonne richieste: {missing}\n"
            f"File: {weather_dict_file}\n"
            f"Colonne trovate: {weather_dict.columns.tolist()}"
        )

    mw = master_weather.copy()
    mw["Feature"] = mw["Feature"].astype(str)

    weather_dict["Feature_Code"] = weather_dict["Feature_Code"].astype(str)
    weather_dict["Original_Feature_Name"] = weather_dict["Original_Feature_Name"].astype(str)

    # Caso 1: Feature = WeatherV3_1, WeatherV3_2, ...
    merged_code = mw.merge(
        weather_dict,
        left_on="Feature",
        right_on="Feature_Code",
        how="left"
    )

    # Caso 2: Feature = Temperature_Dmean_sum_P1, ecc.
    missing_mask = merged_code["Original_Feature_Name"].isna()

    if missing_mask.any():
        mw_missing = merged_code.loc[missing_mask, mw.columns].copy()

        merged_original = mw_missing.merge(
            weather_dict,
            left_on="Feature",
            right_on="Original_Feature_Name",
            how="left"
        )

        merged_code = pd.concat(
            [
                merged_code.loc[~missing_mask].copy(),
                merged_original.copy()
            ],
            ignore_index=True
        )

    merged_code["Original_Feature_Name"] = merged_code["Original_Feature_Name"].fillna(
        merged_code["Feature"]
    )
    merged_code["Period"] = merged_code["Period"].fillna("Unknown")
    merged_code["Variable"] = merged_code["Variable"].fillna("Unknown")
    merged_code["Statistic"] = merged_code["Statistic"].fillna("Unknown")

    return merged_code


def build_gene_master_from_mapped(
    trait: str,
    master_mapped: pd.DataFrame,
    split_inputs_dir: Path,
) -> pd.DataFrame:
    gene_rows = []

    split_dirs = sorted(
        split_inputs_dir.glob("CV*_Split*"),
        key=lambda p: natural_split_sort_key(p.name)
    )

    if len(split_dirs) == 0:
        print(f"ATTENZIONE: nessuna cartella split trovata in {split_inputs_dir}")

    for split_dir in split_dirs:
        split_name = split_dir.name
        edge_file = split_dir / "snp_to_gene_edges.csv"

        if not edge_file.exists():
            print(f"ATTENZIONE: edge file mancante per {split_name}: {edge_file}")
            continue

        edges = pd.read_csv(edge_file, dtype=str)

        if not {"SNP", "Gene"}.issubset(edges.columns):
            raise ValueError(
                f"Il file {edge_file} deve contenere almeno le colonne SNP e Gene"
            )

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
            merged.groupby(["Trait", "Split", "Gene"])
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
            "Trait",
            "Split",
            "Gene",
            "gene_mean_abs_SHAP",
            "n_snps_in_gene_in_split",
            "Rank",
        ])

    out = pd.concat(gene_rows, ignore_index=True)
    out["Trait"] = trait

    return out


def build_gene_summary(master_gene: pd.DataFrame, trait: str) -> pd.DataFrame:
    if len(master_gene) == 0:
        return pd.DataFrame(columns=[
            "Trait",
            "Gene",
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
            "mean_n_snps_in_gene",
            "top20_count",
            "top50_count",
            "GlobalRank",
        ])

    top20_df = master_gene[master_gene["Rank"] <= 20].copy()
    top50_df = master_gene[master_gene["Rank"] <= 50].copy()

    top20_counts = (
        top20_df.groupby("Gene")
        .size()
        .rename("top20_count")
        .reset_index()
    )

    top50_counts = (
        top50_df.groupby("Gene")
        .size()
        .rename("top50_count")
        .reset_index()
    )

    summary_gene = (
        master_gene.groupby("Gene")
        .agg(
            meanSHAP=("gene_mean_abs_SHAP", "mean"),
            medianSHAP=("gene_mean_abs_SHAP", "median"),
            sdSHAP=("gene_mean_abs_SHAP", "std"),
            minSHAP=("gene_mean_abs_SHAP", "min"),
            maxSHAP=("gene_mean_abs_SHAP", "max"),
            n_folds=("gene_mean_abs_SHAP", "size"),
            meanRank=("Rank", "mean"),
            medianRank=("Rank", "median"),
            bestRank=("Rank", "min"),
            worstRank=("Rank", "max"),
            mean_n_snps_in_gene=("n_snps_in_gene_in_split", "mean"),
        )
        .reset_index()
    )

    summary_gene = summary_gene.merge(top20_counts, on="Gene", how="left")
    summary_gene = summary_gene.merge(top50_counts, on="Gene", how="left")

    summary_gene["top20_count"] = summary_gene["top20_count"].fillna(0).astype(int)
    summary_gene["top50_count"] = summary_gene["top50_count"].fillna(0).astype(int)
    summary_gene["sdSHAP"] = summary_gene["sdSHAP"].fillna(0.0)
    summary_gene["Trait"] = trait

    summary_gene = summary_gene.sort_values(
        ["meanSHAP", "top20_count", "top50_count", "meanRank"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    summary_gene["GlobalRank"] = np.arange(1, len(summary_gene) + 1)

    summary_gene = summary_gene[
        [
            "Trait",
            "Gene",
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
            "mean_n_snps_in_gene",
            "top20_count",
            "top50_count",
            "GlobalRank",
        ]
    ]

    return summary_gene


# =============================================================================
# MAIN PER TRAIT
# =============================================================================

def process_one_trait(trait: str):
    print("\n" + "=" * 100)
    print(f"Q5 - AGGREGATE SHAP NO-SOIL FOR TRAIT: {trait}")
    print("=" * 100)

    trait_model_dir = BASE_MODEL_DIR / trait
    splitwise_dir = trait_model_dir / "Interpretation" / "Splitwise_tables"
    dict_dir = trait_model_dir / "Interpretation" / "Feature_dictionaries"
    agg_dir = trait_model_dir / "Interpretation" / "Aggregated_tables"
    reports_dir = trait_model_dir / "Interpretation" / "Reports"

    split_inputs_dir = BIO_BASE_DIR / trait / "split_inputs"
    weather_dict_file = dict_dir / "weather_v3_feature_dictionary.csv"

    check_dir(splitwise_dir)
    check_dir(split_inputs_dir)
    check_file(weather_dict_file)

    agg_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Caricamento tabelle split-wise...")

    master_weather = load_csvs(splitwise_dir, f"{trait}_*_weather_mean_abs_SHAP.csv")
    master_pca = load_csvs(splitwise_dir, f"{trait}_*_pca_mean_abs_SHAP.csv")
    master_mapped = load_csvs(splitwise_dir, f"{trait}_*_mapped_snp_mean_abs_SHAP.csv")
    master_unmapped = load_csvs(splitwise_dir, f"{trait}_*_unmapped_snp_mean_abs_SHAP.csv")
    master_allsnp = load_csvs(splitwise_dir, f"{trait}_*_all_snp_mean_abs_SHAP.csv")
    master_branch = load_csvs(splitwise_dir, f"{trait}_*_branch_importance.csv")

    # Normalizza colonne Trait.
    for df in [
        master_weather,
        master_pca,
        master_mapped,
        master_unmapped,
        master_allsnp,
        master_branch,
    ]:
        df["Trait"] = trait

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
    master_weather = attach_weather_dictionary(master_weather, weather_dict_file)

    # -------------------------------------------------------------------------
    # Save master files
    # -------------------------------------------------------------------------
    print("Salvataggio master files...")

    master_weather_file = agg_dir / f"MASTER_weather_SHAP_{MODEL_NAME}_{trait}.csv"
    master_pca_file = agg_dir / f"MASTER_pca_SHAP_{MODEL_NAME}_{trait}.csv"
    master_mapped_file = agg_dir / f"MASTER_mapped_snp_SHAP_{MODEL_NAME}_{trait}.csv"
    master_unmapped_file = agg_dir / f"MASTER_unmapped_snp_SHAP_{MODEL_NAME}_{trait}.csv"
    master_allsnp_file = agg_dir / f"MASTER_all_snp_SHAP_{MODEL_NAME}_{trait}.csv"
    master_branch_file = agg_dir / f"MASTER_branch_importance_{MODEL_NAME}_{trait}.csv"
    master_gene_file = agg_dir / f"MASTER_gene_SHAP_{MODEL_NAME}_{trait}.csv"

    master_weather.to_csv(master_weather_file, index=False)
    master_pca.to_csv(master_pca_file, index=False)
    master_mapped.to_csv(master_mapped_file, index=False)
    master_unmapped.to_csv(master_unmapped_file, index=False)
    master_allsnp.to_csv(master_allsnp_file, index=False)
    master_branch.to_csv(master_branch_file, index=False)

    # -------------------------------------------------------------------------
    # Summary feature-level
    # -------------------------------------------------------------------------
    print("Costruzione summary feature-level...")

    summary_weather = build_summary_feature_level(
        master_weather,
        "Original_Feature_Name"
    )

    summary_weather = summary_weather.rename(
        columns={"Original_Feature_Name": "Feature"}
    )

    weather_meta = (
        master_weather[["Original_Feature_Name", "Period", "Variable", "Statistic"]]
        .drop_duplicates()
        .rename(columns={"Original_Feature_Name": "Feature"})
    )

    summary_weather = summary_weather.merge(weather_meta, on="Feature", how="left")
    summary_weather["Trait"] = trait

    summary_pca = build_summary_feature_level(master_pca, "Feature")
    summary_mapped = build_summary_feature_level(master_mapped, "SNP")
    summary_unmapped = build_summary_feature_level(master_unmapped, "SNP")
    summary_allsnp = build_summary_feature_level(master_allsnp, "SNP")

    for df in [summary_pca, summary_mapped, summary_unmapped, summary_allsnp]:
        df["Trait"] = trait

    # Aggiungi MappedStatus al summary all SNP, se disponibile.
    if "MappedStatus" in master_allsnp.columns:
        status_meta = (
            master_allsnp[["SNP", "MappedStatus"]]
            .drop_duplicates()
        )

        # Se per qualche motivo uno SNP appare sia mapped sia unmapped in split diversi,
        # teniamo una versione combinata.
        status_meta = (
            status_meta.groupby("SNP")["MappedStatus"]
            .apply(lambda x: ";".join(sorted(set(x.astype(str)))))
            .reset_index()
        )

        summary_allsnp = summary_allsnp.merge(status_meta, on="SNP", how="left")

    summary_weather_file = agg_dir / f"SUMMARY_weather_SHAP_{MODEL_NAME}_{trait}.csv"
    summary_pca_file = agg_dir / f"SUMMARY_pca_SHAP_{MODEL_NAME}_{trait}.csv"
    summary_mapped_file = agg_dir / f"SUMMARY_mapped_snp_SHAP_{MODEL_NAME}_{trait}.csv"
    summary_unmapped_file = agg_dir / f"SUMMARY_unmapped_snp_SHAP_{MODEL_NAME}_{trait}.csv"
    summary_allsnp_file = agg_dir / f"SUMMARY_all_snp_SHAP_{MODEL_NAME}_{trait}.csv"

    summary_weather.to_csv(summary_weather_file, index=False)
    summary_pca.to_csv(summary_pca_file, index=False)
    summary_mapped.to_csv(summary_mapped_file, index=False)
    summary_unmapped.to_csv(summary_unmapped_file, index=False)
    summary_allsnp.to_csv(summary_allsnp_file, index=False)

    # -------------------------------------------------------------------------
    # Branch importance
    # -------------------------------------------------------------------------
    print("Summary branch importance...")

    master_branch["mean_abs_SHAP"] = pd.to_numeric(
        master_branch["mean_abs_SHAP"],
        errors="coerce"
    )

    summary_branch = (
        master_branch.groupby("Branch")
        .agg(
            meanSHAP=("mean_abs_SHAP", "mean"),
            medianSHAP=("mean_abs_SHAP", "median"),
            sdSHAP=("mean_abs_SHAP", "std"),
            minSHAP=("mean_abs_SHAP", "min"),
            maxSHAP=("mean_abs_SHAP", "max"),
            n_splits=("mean_abs_SHAP", "size"),
            mean_n_features=("n_features", "mean") if "n_features" in master_branch.columns else ("mean_abs_SHAP", "size"),
        )
        .reset_index()
        .sort_values("meanSHAP", ascending=False)
        .reset_index(drop=True)
    )

    summary_branch["sdSHAP"] = summary_branch["sdSHAP"].fillna(0.0)
    summary_branch["Trait"] = trait

    summary_branch_file = agg_dir / f"SUMMARY_branch_importance_{MODEL_NAME}_{trait}.csv"
    summary_branch.to_csv(summary_branch_file, index=False)

    mapped_mask = summary_branch["Branch"] == "Mapped_SNP"
    unmapped_mask = summary_branch["Branch"] == "Unmapped_SNP"

    mapped_branch_mean = (
        summary_branch.loc[mapped_mask, "meanSHAP"].iloc[0]
        if mapped_mask.any()
        else 0.0
    )

    unmapped_branch_mean = (
        summary_branch.loc[unmapped_mask, "meanSHAP"].iloc[0]
        if unmapped_mask.any()
        else 0.0
    )

    summary_branch_genomic = summary_branch.copy()

    extra = pd.DataFrame([{
        "Branch": "SNP_total",
        "meanSHAP": mapped_branch_mean + unmapped_branch_mean,
        "medianSHAP": np.nan,
        "sdSHAP": np.nan,
        "minSHAP": np.nan,
        "maxSHAP": np.nan,
        "n_splits": summary_branch["n_splits"].max() if len(summary_branch) > 0 else np.nan,
        "mean_n_features": np.nan,
        "Trait": trait,
    }])

    summary_branch_genomic = pd.concat(
        [summary_branch_genomic, extra],
        ignore_index=True
    )

    summary_branch_genomic_file = (
        agg_dir / f"SUMMARY_branch_importance_with_snp_total_{MODEL_NAME}_{trait}.csv"
    )
    summary_branch_genomic.to_csv(summary_branch_genomic_file, index=False)

    # -------------------------------------------------------------------------
    # Mapped vs unmapped
    # -------------------------------------------------------------------------
    print("Summary mapped vs unmapped...")

    mapped_vs_unmapped = pd.DataFrame({
        "Trait": [trait, trait],
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

    mapped_vs_unmapped_file = agg_dir / f"SUMMARY_mapped_vs_unmapped_{MODEL_NAME}_{trait}.csv"
    mapped_vs_unmapped.to_csv(mapped_vs_unmapped_file, index=False)

    # -------------------------------------------------------------------------
    # Gene-level
    # -------------------------------------------------------------------------
    print("Summary gene-level...")

    master_gene = build_gene_master_from_mapped(
        trait=trait,
        master_mapped=master_mapped,
        split_inputs_dir=split_inputs_dir,
    )

    master_gene.to_csv(master_gene_file, index=False)

    summary_gene = build_gene_summary(master_gene, trait)

    summary_gene_file = agg_dir / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{trait}.csv"
    summary_gene_stable5_file = agg_dir / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{trait}_stable_nfolds_ge5.csv"
    summary_gene_stable10_file = agg_dir / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{trait}_stable_nfolds_ge10.csv"

    summary_gene.to_csv(summary_gene_file, index=False)
    summary_gene[summary_gene["n_folds"] >= 5].to_csv(summary_gene_stable5_file, index=False)
    summary_gene[summary_gene["n_folds"] >= 10].to_csv(summary_gene_stable10_file, index=False)

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
            maxSHAP=("meanSHAP", "max"),
            n_features=("Feature", "size"),
        )
        .reset_index()
        .sort_values("meanSHAP", ascending=False)
        .reset_index(drop=True)
    )
    weather_period_summary["Trait"] = trait

    weather_variable_summary = (
        weather_aux.groupby("Variable")
        .agg(
            meanSHAP=("meanSHAP", "mean"),
            medianSHAP=("meanSHAP", "median"),
            maxSHAP=("meanSHAP", "max"),
            n_features=("Feature", "size"),
        )
        .reset_index()
        .sort_values("meanSHAP", ascending=False)
        .reset_index(drop=True)
    )
    weather_variable_summary["Trait"] = trait

    weather_stat_summary = (
        weather_aux.groupby("Statistic")
        .agg(
            meanSHAP=("meanSHAP", "mean"),
            medianSHAP=("meanSHAP", "median"),
            maxSHAP=("meanSHAP", "max"),
            n_features=("Feature", "size"),
        )
        .reset_index()
        .sort_values("meanSHAP", ascending=False)
        .reset_index(drop=True)
    )
    weather_stat_summary["Trait"] = trait

    weather_period_file = agg_dir / f"SUMMARY_weather_by_period_{MODEL_NAME}_{trait}.csv"
    weather_variable_file = agg_dir / f"SUMMARY_weather_by_variable_{MODEL_NAME}_{trait}.csv"
    weather_stat_file = agg_dir / f"SUMMARY_weather_by_stat_{MODEL_NAME}_{trait}.csv"

    weather_period_summary.to_csv(weather_period_file, index=False)
    weather_variable_summary.to_csv(weather_variable_file, index=False)
    weather_stat_summary.to_csv(weather_stat_file, index=False)

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    print("Scrittura report...")

    report_file = reports_dir / f"Q5_aggregation_report_{MODEL_NAME}_{trait}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== Q5 SHAP AGGREGATION REPORT FOR V3 NO-SOIL ===\n\n")

        f.write(f"MODEL_NAME: {MODEL_NAME}\n")
        f.write(f"TRAIT: {trait}\n")
        f.write("Soil branch: REMOVED\n\n")

        f.write("Input splitwise folder:\n")
        f.write(str(splitwise_dir))
        f.write("\n\n")

        f.write("Master shapes:\n")
        f.write(f"master_weather: {master_weather.shape}\n")
        f.write(f"master_pca: {master_pca.shape}\n")
        f.write(f"master_mapped: {master_mapped.shape}\n")
        f.write(f"master_unmapped: {master_unmapped.shape}\n")
        f.write(f"master_allsnp: {master_allsnp.shape}\n")
        f.write(f"master_branch: {master_branch.shape}\n")
        f.write(f"master_gene: {master_gene.shape}\n\n")

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

        f.write("Top stable genes n_folds >= 5:\n")
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
    print(agg_dir)

    return {
        "Trait": trait,
        "AggregatedDir": str(agg_dir),
        "n_weather_rows": len(master_weather),
        "n_pca_rows": len(master_pca),
        "n_mapped_rows": len(master_mapped),
        "n_unmapped_rows": len(master_unmapped),
        "n_allsnp_rows": len(master_allsnp),
        "n_branch_rows": len(master_branch),
        "n_gene_rows": len(master_gene),
        "n_unique_snps": master_allsnp["SNP"].nunique() if "SNP" in master_allsnp.columns else 0,
        "n_unique_genes": master_gene["Gene"].nunique() if "Gene" in master_gene.columns else 0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("Q5 - AGGREGATE SHAP NO-SOIL V3 FOR NEW TRAITS")
    print("=" * 100)

    rows = []

    for trait in TRAITS:
        rows.append(process_one_trait(trait))

    summary = pd.DataFrame(rows)

    summary_file = BASE_MODEL_DIR / "Q5_aggregation_summary_all_traits.csv"
    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 100)
    print("Q5 completed.")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("\nSaved summary:")
    print(summary_file)


if __name__ == "__main__":
    main()
