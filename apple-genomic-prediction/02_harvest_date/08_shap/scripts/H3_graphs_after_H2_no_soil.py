# -*- coding: utf-8 -*-

################################################################################
### H3_graphs_after_H2_no_soil.py
### Plot aggregated SHAP interpretation results for V3 no-soil model
################################################################################

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

TRAIT = "Harvest_date"
MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"

OUT_DIR = Path("Output")

AGG_DIR = OUT_DIR / "Interpretation" / TRAIT / "Aggregated_tables"
GRAPH_DIR = OUT_DIR / "Interpretation" / TRAIT / "Graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_WEATHER = AGG_DIR / f"SUMMARY_weather_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_PCA = AGG_DIR / f"SUMMARY_pca_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_MAPPED = AGG_DIR / f"SUMMARY_mapped_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_UNMAPPED = AGG_DIR / f"SUMMARY_unmapped_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_ALLSNP = AGG_DIR / f"SUMMARY_all_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_BRANCH = AGG_DIR / f"SUMMARY_branch_importance_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_GENE = AGG_DIR / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_MAPPED_UNMAPPED = AGG_DIR / f"SUMMARY_mapped_vs_unmapped_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_WEATHER_PERIOD = AGG_DIR / f"SUMMARY_weather_by_period_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_WEATHER_VARIABLE = AGG_DIR / f"SUMMARY_weather_by_variable_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_BRANCH_GENOMIC = AGG_DIR / f"SUMMARY_branch_importance_with_snp_total_{MODEL_NAME}_{TRAIT}.csv"
SUMMARY_GENE_STABLE5 = AGG_DIR / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{TRAIT}_stable_nfolds_ge5.csv"
SUMMARY_GENE_STABLE10 = AGG_DIR / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{TRAIT}_stable_nfolds_ge10.csv"
SUMMARY_WEATHER_STAT = AGG_DIR / f"SUMMARY_weather_by_stat_{MODEL_NAME}_{TRAIT}.csv"

MASTER_ALLSNP = AGG_DIR / f"MASTER_all_snp_SHAP_{MODEL_NAME}_{TRAIT}.csv"
MASTER_GENE = AGG_DIR / f"MASTER_gene_SHAP_{MODEL_NAME}_{TRAIT}.csv"

SUMMARY_TXT = GRAPH_DIR / f"shap_summary_{MODEL_NAME}_{TRAIT}.txt"


# =============================================================================
# UTILITIES
# =============================================================================

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_barh(df, value_col, label_col, title, xlabel, outpath, topn=30, sort_desc=True):
    if df is None or len(df) == 0:
        print(f"[SKIP] Empty dataframe for {outpath}")
        return

    sub = df.sort_values(value_col, ascending=not sort_desc).head(topn).copy()

    plt.figure(figsize=(10, max(6, topn * 0.28)))
    plt.barh(sub[label_col][::-1], sub[value_col][::-1])
    plt.xlabel(xlabel)
    plt.title(title)
    savefig(outpath)


def read_if_exists(path):
    if path.exists():
        return pd.read_csv(path)
    print(f"[WARNING] Missing file: {path}")
    return pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("H3 - GRAPHS AFTER H2 NO-SOIL")
    print("=" * 80)

    print("Loading aggregated SHAP tables...")

    summary_weather = pd.read_csv(SUMMARY_WEATHER)
    summary_pca = pd.read_csv(SUMMARY_PCA)
    summary_mapped = pd.read_csv(SUMMARY_MAPPED)
    summary_unmapped = pd.read_csv(SUMMARY_UNMAPPED)
    summary_allsnp = pd.read_csv(SUMMARY_ALLSNP)
    summary_branch = pd.read_csv(SUMMARY_BRANCH)
    summary_gene = pd.read_csv(SUMMARY_GENE)
    summary_mu = pd.read_csv(SUMMARY_MAPPED_UNMAPPED)
    summary_w_period = pd.read_csv(SUMMARY_WEATHER_PERIOD)
    summary_w_var = pd.read_csv(SUMMARY_WEATHER_VARIABLE)
    summary_branch_genomic = pd.read_csv(SUMMARY_BRANCH_GENOMIC)
    summary_w_stat = pd.read_csv(SUMMARY_WEATHER_STAT)

    summary_gene_stable5 = read_if_exists(SUMMARY_GENE_STABLE5)
    summary_gene_stable10 = read_if_exists(SUMMARY_GENE_STABLE10)

    master_allsnp = pd.read_csv(MASTER_ALLSNP)
    master_gene = pd.read_csv(MASTER_GENE) if MASTER_GENE.exists() else pd.DataFrame()

    print("Loaded:")
    print("  weather:", summary_weather.shape)
    print("  pca:", summary_pca.shape)
    print("  mapped:", summary_mapped.shape)
    print("  unmapped:", summary_unmapped.shape)
    print("  all snp:", summary_allsnp.shape)
    print("  branch:", summary_branch.shape)
    print("  gene:", summary_gene.shape)

    # -------------------------------------------------------------------------
    # 1. Top SNP overall
    # -------------------------------------------------------------------------
    top30_snp = summary_allsnp.head(30).copy()
    top50_snp = summary_allsnp.head(50).copy()

    top30_snp.to_csv(GRAPH_DIR / "top30_snps_overall.csv", index=False)
    top50_snp.to_csv(GRAPH_DIR / "top50_snps_overall.csv", index=False)

    plot_barh(
        summary_allsnp,
        "meanSHAP",
        "SNP",
        "Top 30 SNPs overall by meanSHAP - no soil",
        "Mean absolute SHAP",
        GRAPH_DIR / "top30_snps_overall.png",
        topn=30
    )

    # -------------------------------------------------------------------------
    # 2. Top SNP by top20_count
    # -------------------------------------------------------------------------
    top30_snp_stable = summary_allsnp.sort_values(
        ["top20_count", "meanSHAP"],
        ascending=[False, False]
    ).head(30).copy()

    top30_snp_stable.to_csv(GRAPH_DIR / "top30_snps_by_top20_count.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.barh(top30_snp_stable["SNP"][::-1], top30_snp_stable["top20_count"][::-1])
    plt.xlabel("Number of splits in top 20")
    plt.title("Top 30 SNPs by top20_count - no soil")
    savefig(GRAPH_DIR / "top30_snps_by_top20_count.png")

    # -------------------------------------------------------------------------
    # 3. Top genes overall
    # -------------------------------------------------------------------------
    if len(summary_gene) > 0:
        top30_gene = summary_gene.head(30).copy()
        top30_gene.to_csv(GRAPH_DIR / "top30_genes_overall.csv", index=False)

        plot_barh(
            summary_gene,
            "meanSHAP",
            "Gene",
            "Top 30 genes by aggregated mapped SNP SHAP - no soil",
            "Mean aggregated gene SHAP",
            GRAPH_DIR / "top30_genes_overall.png",
            topn=30
        )

        top30_gene_stable = summary_gene.sort_values(
            ["top20_count", "meanSHAP"],
            ascending=[False, False]
        ).head(30).copy()

        top30_gene_stable.to_csv(GRAPH_DIR / "top30_genes_by_top20_count.csv", index=False)

        plt.figure(figsize=(10, 8))
        plt.barh(top30_gene_stable["Gene"][::-1], top30_gene_stable["top20_count"][::-1])
        plt.xlabel("Number of splits in top 20")
        plt.title("Top 30 genes by top20_count - no soil")
        savefig(GRAPH_DIR / "top30_genes_by_top20_count.png")

    # -------------------------------------------------------------------------
    # 4. Branch importance
    # -------------------------------------------------------------------------
    summary_branch.to_csv(GRAPH_DIR / "branch_importance_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(summary_branch["Branch"], summary_branch["meanSHAP"])
    plt.ylabel("Mean absolute SHAP")
    plt.title("Branch importance - no soil")
    plt.xticks(rotation=30)
    savefig(GRAPH_DIR / "branch_importance.png")

    # -------------------------------------------------------------------------
    # 5. Mapped vs unmapped
    # -------------------------------------------------------------------------
    summary_mu.to_csv(GRAPH_DIR / "mapped_vs_unmapped_summary.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.bar(summary_mu["Category"], summary_mu["meanSHAP"])
    plt.ylabel("Mean absolute SHAP")
    plt.title("Mapped vs unmapped SNP importance - no soil")
    savefig(GRAPH_DIR / "mapped_vs_unmapped.png")

    # -------------------------------------------------------------------------
    # 6. Weather by period
    # -------------------------------------------------------------------------
    summary_w_period.to_csv(GRAPH_DIR / "weather_by_period_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(summary_w_period["Period"], summary_w_period["meanSHAP"])
    plt.ylabel("Mean feature SHAP")
    plt.title("Weather importance by period - no soil")
    savefig(GRAPH_DIR / "weather_by_period.png")

    # -------------------------------------------------------------------------
    # 7. Weather by variable
    # -------------------------------------------------------------------------
    summary_w_var.to_csv(GRAPH_DIR / "weather_by_variable_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(summary_w_var["Variable"], summary_w_var["meanSHAP"])
    plt.ylabel("Mean feature SHAP")
    plt.title("Weather importance by variable - no soil")
    savefig(GRAPH_DIR / "weather_by_variable.png")

    # -------------------------------------------------------------------------
    # 8. Top weather features
    # -------------------------------------------------------------------------
    top30_weather = summary_weather.head(30).copy()
    top30_weather.to_csv(GRAPH_DIR / "top30_weather_features.csv", index=False)

    plot_barh(
        summary_weather,
        "meanSHAP",
        "Feature",
        "Top 30 weather features - no soil",
        "Mean absolute SHAP",
        GRAPH_DIR / "top30_weather_features.png",
        topn=30
    )

    # -------------------------------------------------------------------------
    # 9. Top PCA features
    # -------------------------------------------------------------------------
    plot_barh(
        summary_pca,
        "meanSHAP",
        "Feature",
        "Top PCA features - no soil",
        "Mean absolute SHAP",
        GRAPH_DIR / "top_pca_features.png",
        topn=min(20, len(summary_pca))
    )

    # -------------------------------------------------------------------------
    # 10. Scatter meanSHAP vs top20_count for SNPs
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.scatter(summary_allsnp["top20_count"], summary_allsnp["meanSHAP"], alpha=0.6, s=25)
    plt.xlabel("top20_count")
    plt.ylabel("Mean absolute SHAP")
    plt.title("SNP meanSHAP vs top20_count - no soil")
    plt.grid(True, alpha=0.25)
    savefig(GRAPH_DIR / "scatter_snp_meanSHAP_vs_top20count.png")

    # -------------------------------------------------------------------------
    # 11. Scatter meanSHAP vs n_folds for SNPs
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.scatter(summary_allsnp["n_folds"], summary_allsnp["meanSHAP"], alpha=0.6, s=25)
    plt.xlabel("n_folds")
    plt.ylabel("Mean absolute SHAP")
    plt.title("SNP meanSHAP vs n_folds - no soil")
    plt.grid(True, alpha=0.25)
    savefig(GRAPH_DIR / "scatter_snp_meanSHAP_vs_nfolds.png")

    # -------------------------------------------------------------------------
    # 12. Heatmap-like matrix for top SNPs across splits
    # -------------------------------------------------------------------------
    top30_snp_names = top30_snp["SNP"].tolist()
    heat_df = master_allsnp[master_allsnp["SNP"].isin(top30_snp_names)].copy()

    heat_mat = heat_df.pivot(index="SNP", columns="Split", values="mean_abs_SHAP")
    heat_mat = heat_mat.reindex(top30_snp_names)

    heat_mat.to_csv(GRAPH_DIR / "top30_snps_heatmap_matrix.csv")

    plt.figure(figsize=(14, 10))
    plt.imshow(heat_mat.values, aspect="auto")
    plt.colorbar(label="mean_abs_SHAP")
    plt.yticks(np.arange(len(heat_mat.index)), heat_mat.index)
    plt.xticks(np.arange(len(heat_mat.columns)), heat_mat.columns, rotation=90)
    plt.title("Top 30 SNPs across splits - no soil")
    savefig(GRAPH_DIR / "top30_snps_heatmap.png")

    # -------------------------------------------------------------------------
    # 13. Heatmap-like matrix for top genes across splits
    # -------------------------------------------------------------------------
    if len(master_gene) > 0 and len(summary_gene) > 0:
        top30_gene_names = summary_gene.head(30)["Gene"].tolist()
        gene_heat_df = master_gene[master_gene["Gene"].isin(top30_gene_names)].copy()

        gene_heat_mat = gene_heat_df.pivot(
            index="Gene",
            columns="Split",
            values="gene_mean_abs_SHAP"
        )

        gene_heat_mat = gene_heat_mat.reindex(top30_gene_names)
        gene_heat_mat.to_csv(GRAPH_DIR / "top30_genes_heatmap_matrix.csv")

        plt.figure(figsize=(14, 10))
        plt.imshow(gene_heat_mat.values, aspect="auto")
        plt.colorbar(label="gene_mean_abs_SHAP")
        plt.yticks(np.arange(len(gene_heat_mat.index)), gene_heat_mat.index)
        plt.xticks(np.arange(len(gene_heat_mat.columns)), gene_heat_mat.columns, rotation=90)
        plt.title("Top 30 genes across splits - no soil")
        savefig(GRAPH_DIR / "top30_genes_heatmap.png")

    # -------------------------------------------------------------------------
    # 14. Branch importance with SNP_total
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    tmp = summary_branch_genomic.copy()
    plt.bar(tmp["Branch"], tmp["meanSHAP"])
    plt.ylabel("Mean absolute SHAP")
    plt.title("Branch importance with SNP_total - no soil")
    plt.xticks(rotation=30)
    savefig(GRAPH_DIR / "branch_importance_with_snp_total.png")

    # -------------------------------------------------------------------------
    # 15. Weather by statistic
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(summary_w_stat["Statistic"], summary_w_stat["meanSHAP"])
    plt.ylabel("Mean feature SHAP")
    plt.title("Weather importance by statistic - no soil")
    savefig(GRAPH_DIR / "weather_by_statistic.png")

    # -------------------------------------------------------------------------
    # 16. Top stable genes n_folds >= 5
    # -------------------------------------------------------------------------
    if len(summary_gene_stable5) > 0:
        top30 = summary_gene_stable5.head(30)

        plt.figure(figsize=(10, 8))
        plt.barh(top30["Gene"][::-1], top30["meanSHAP"][::-1])
        plt.xlabel("Mean aggregated gene SHAP")
        plt.title("Top stable genes (n_folds >= 5) - no soil")
        savefig(GRAPH_DIR / "top30_stable_genes_nfolds_ge5.png")

    # -------------------------------------------------------------------------
    # 17. Top stable genes n_folds >= 10
    # -------------------------------------------------------------------------
    if len(summary_gene_stable10) > 0:
        top30 = summary_gene_stable10.head(30)

        plt.figure(figsize=(10, 8))
        plt.barh(top30["Gene"][::-1], top30["meanSHAP"][::-1])
        plt.xlabel("Mean aggregated gene SHAP")
        plt.title("Top stable genes (n_folds >= 10) - no soil")
        savefig(GRAPH_DIR / "top30_stable_genes_nfolds_ge10.png")

    # -------------------------------------------------------------------------
    # 18. Text summary
    # -------------------------------------------------------------------------
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(f"=== SHAP SUMMARY FOR {MODEL_NAME} / {TRAIT} ===\n\n")
        f.write("Soil branch: REMOVED\n\n")

        f.write("Branch importance:\n")
        f.write(summary_branch.to_string(index=False))
        f.write("\n\n")

        f.write("Branch importance with SNP_total:\n")
        f.write(summary_branch_genomic.to_string(index=False))
        f.write("\n\n")

        f.write("Mapped vs unmapped:\n")
        f.write(summary_mu.to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 SNPs overall:\n")
        f.write(summary_allsnp.head(20).to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 SNPs by top20_count:\n")
        f.write(top30_snp_stable.head(20).to_string(index=False))
        f.write("\n\n")

        if len(summary_gene) > 0:
            f.write("Top 20 genes overall:\n")
            f.write(summary_gene.head(20).to_string(index=False))
            f.write("\n\n")

        f.write("Top 20 weather features:\n")
        f.write(summary_weather.head(20).to_string(index=False))
        f.write("\n\n")

        f.write("Weather by period:\n")
        f.write(summary_w_period.to_string(index=False))
        f.write("\n\n")

        f.write("Weather by variable:\n")
        f.write(summary_w_var.to_string(index=False))
        f.write("\n\n")

        f.write("Weather by statistic:\n")
        f.write(summary_w_stat.to_string(index=False))
        f.write("\n\n")

        f.write("Summary of SNP meanSHAP:\n")
        f.write(summary_allsnp["meanSHAP"].describe().to_string())
        f.write("\n\n")

        f.write("Summary of SNP n_folds:\n")
        f.write(summary_allsnp["n_folds"].describe().to_string())
        f.write("\n\n")

    print("SHAP plots and summaries saved in:")
    print(GRAPH_DIR)


if __name__ == "__main__":
    main()