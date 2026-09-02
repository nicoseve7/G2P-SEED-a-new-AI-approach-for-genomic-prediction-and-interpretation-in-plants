# -*- coding: utf-8 -*-

################################################################################
### Q6_graphs_after_Q5_no_soil_newtraits.py
###
### Plot aggregated SHAP interpretation results for V3 no-soil model
### for:
###   - Acidity
###   - Color_over
################################################################################

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"

BASE_MODEL_DIR = Path("Output/02_no_soil_model")


# =============================================================================
# UTILITIES
# =============================================================================

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def read_if_exists(path):
    if path.exists():
        return pd.read_csv(path)
    print(f"[WARNING] Missing file: {path}")
    return pd.DataFrame()


def plot_barh(
    df,
    value_col,
    label_col,
    title,
    xlabel,
    outpath,
    topn=30,
    sort_desc=True,
):
    if df is None or len(df) == 0:
        print(f"[SKIP] Empty dataframe for {outpath}")
        return

    if value_col not in df.columns or label_col not in df.columns:
        print(f"[SKIP] Missing columns for {outpath}")
        print(f"Needed: {value_col}, {label_col}")
        print(f"Found: {df.columns.tolist()}")
        return

    sub = df.sort_values(value_col, ascending=not sort_desc).head(topn).copy()

    plt.figure(figsize=(10, max(6, topn * 0.28)))
    plt.barh(sub[label_col].astype(str)[::-1], sub[value_col][::-1])
    plt.xlabel(xlabel)
    plt.title(title)
    savefig(outpath)


def get_file_paths(trait: str):
    trait_dir = BASE_MODEL_DIR / trait
    agg_dir = trait_dir / "Interpretation" / "Aggregated_tables"
    graph_dir = trait_dir / "Interpretation" / "Graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "trait_dir": trait_dir,
        "agg_dir": agg_dir,
        "graph_dir": graph_dir,

        "summary_weather": agg_dir / f"SUMMARY_weather_SHAP_{MODEL_NAME}_{trait}.csv",
        "summary_pca": agg_dir / f"SUMMARY_pca_SHAP_{MODEL_NAME}_{trait}.csv",
        "summary_mapped": agg_dir / f"SUMMARY_mapped_snp_SHAP_{MODEL_NAME}_{trait}.csv",
        "summary_unmapped": agg_dir / f"SUMMARY_unmapped_snp_SHAP_{MODEL_NAME}_{trait}.csv",
        "summary_allsnp": agg_dir / f"SUMMARY_all_snp_SHAP_{MODEL_NAME}_{trait}.csv",
        "summary_branch": agg_dir / f"SUMMARY_branch_importance_{MODEL_NAME}_{trait}.csv",
        "summary_branch_genomic": agg_dir / f"SUMMARY_branch_importance_with_snp_total_{MODEL_NAME}_{trait}.csv",
        "summary_gene": agg_dir / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{trait}.csv",
        "summary_gene_stable5": agg_dir / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{trait}_stable_nfolds_ge5.csv",
        "summary_gene_stable10": agg_dir / f"SUMMARY_gene_SHAP_{MODEL_NAME}_{trait}_stable_nfolds_ge10.csv",
        "summary_mu": agg_dir / f"SUMMARY_mapped_vs_unmapped_{MODEL_NAME}_{trait}.csv",
        "summary_w_period": agg_dir / f"SUMMARY_weather_by_period_{MODEL_NAME}_{trait}.csv",
        "summary_w_var": agg_dir / f"SUMMARY_weather_by_variable_{MODEL_NAME}_{trait}.csv",
        "summary_w_stat": agg_dir / f"SUMMARY_weather_by_stat_{MODEL_NAME}_{trait}.csv",

        "master_allsnp": agg_dir / f"MASTER_all_snp_SHAP_{MODEL_NAME}_{trait}.csv",
        "master_gene": agg_dir / f"MASTER_gene_SHAP_{MODEL_NAME}_{trait}.csv",

        "summary_txt": graph_dir / f"shap_summary_{MODEL_NAME}_{trait}.txt",
    }

    return paths


def check_required_files(paths):
    required = [
        "summary_weather",
        "summary_pca",
        "summary_mapped",
        "summary_unmapped",
        "summary_allsnp",
        "summary_branch",
        "summary_branch_genomic",
        "summary_gene",
        "summary_mu",
        "summary_w_period",
        "summary_w_var",
        "summary_w_stat",
        "master_allsnp",
    ]

    missing = []

    for key in required:
        if not paths[key].exists():
            missing.append(str(paths[key]))

    if missing:
        raise FileNotFoundError(
            "Mancano file aggregati necessari. Hai eseguito Q5?\n\n"
            + "\n".join(missing[:30])
        )


# =============================================================================
# PLOT ONE TRAIT
# =============================================================================

def process_one_trait(trait: str):
    print("\n" + "=" * 100)
    print(f"Q6 - GRAPHS AFTER Q5 NO-SOIL FOR TRAIT: {trait}")
    print("=" * 100)

    paths = get_file_paths(trait)
    check_required_files(paths)

    graph_dir = paths["graph_dir"]

    print("Loading aggregated SHAP tables...")

    summary_weather = pd.read_csv(paths["summary_weather"])
    summary_pca = pd.read_csv(paths["summary_pca"])
    summary_mapped = pd.read_csv(paths["summary_mapped"])
    summary_unmapped = pd.read_csv(paths["summary_unmapped"])
    summary_allsnp = pd.read_csv(paths["summary_allsnp"])
    summary_branch = pd.read_csv(paths["summary_branch"])
    summary_branch_genomic = pd.read_csv(paths["summary_branch_genomic"])
    summary_gene = pd.read_csv(paths["summary_gene"])
    summary_mu = pd.read_csv(paths["summary_mu"])
    summary_w_period = pd.read_csv(paths["summary_w_period"])
    summary_w_var = pd.read_csv(paths["summary_w_var"])
    summary_w_stat = pd.read_csv(paths["summary_w_stat"])

    summary_gene_stable5 = read_if_exists(paths["summary_gene_stable5"])
    summary_gene_stable10 = read_if_exists(paths["summary_gene_stable10"])

    master_allsnp = pd.read_csv(paths["master_allsnp"])
    master_gene = read_if_exists(paths["master_gene"])

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

    top30_snp.to_csv(graph_dir / "top30_snps_overall.csv", index=False)
    top50_snp.to_csv(graph_dir / "top50_snps_overall.csv", index=False)

    plot_barh(
        summary_allsnp,
        "meanSHAP",
        "SNP",
        f"{trait} - Top 30 SNPs overall by meanSHAP - no soil",
        "Mean absolute SHAP",
        graph_dir / "top30_snps_overall.png",
        topn=30,
    )

    # -------------------------------------------------------------------------
    # 2. Top SNP by top20_count
    # -------------------------------------------------------------------------
    top30_snp_stable = summary_allsnp.sort_values(
        ["top20_count", "meanSHAP"],
        ascending=[False, False],
    ).head(30).copy()

    top30_snp_stable.to_csv(
        graph_dir / "top30_snps_by_top20_count.csv",
        index=False,
    )

    plt.figure(figsize=(10, 8))
    plt.barh(
        top30_snp_stable["SNP"].astype(str)[::-1],
        top30_snp_stable["top20_count"][::-1],
    )
    plt.xlabel("Number of splits in top 20")
    plt.title(f"{trait} - Top 30 SNPs by top20_count - no soil")
    savefig(graph_dir / "top30_snps_by_top20_count.png")

    # -------------------------------------------------------------------------
    # 3. Top mapped SNPs and unmapped SNPs separately
    # -------------------------------------------------------------------------
    plot_barh(
        summary_mapped,
        "meanSHAP",
        "SNP",
        f"{trait} - Top 30 mapped SNPs - no soil",
        "Mean absolute SHAP",
        graph_dir / "top30_mapped_snps.png",
        topn=30,
    )

    plot_barh(
        summary_unmapped,
        "meanSHAP",
        "SNP",
        f"{trait} - Top 30 unmapped SNPs - no soil",
        "Mean absolute SHAP",
        graph_dir / "top30_unmapped_snps.png",
        topn=30,
    )

    # -------------------------------------------------------------------------
    # 4. Top genes overall
    # -------------------------------------------------------------------------
    if len(summary_gene) > 0:
        top30_gene = summary_gene.head(30).copy()
        top30_gene.to_csv(graph_dir / "top30_genes_overall.csv", index=False)

        plot_barh(
            summary_gene,
            "meanSHAP",
            "Gene",
            f"{trait} - Top 30 genes by aggregated mapped SNP SHAP - no soil",
            "Mean aggregated gene SHAP",
            graph_dir / "top30_genes_overall.png",
            topn=30,
        )

        top30_gene_stable = summary_gene.sort_values(
            ["top20_count", "meanSHAP"],
            ascending=[False, False],
        ).head(30).copy()

        top30_gene_stable.to_csv(
            graph_dir / "top30_genes_by_top20_count.csv",
            index=False,
        )

        plt.figure(figsize=(10, 8))
        plt.barh(
            top30_gene_stable["Gene"].astype(str)[::-1],
            top30_gene_stable["top20_count"][::-1],
        )
        plt.xlabel("Number of splits in top 20")
        plt.title(f"{trait} - Top 30 genes by top20_count - no soil")
        savefig(graph_dir / "top30_genes_by_top20_count.png")

    # -------------------------------------------------------------------------
    # 5. Branch importance
    # -------------------------------------------------------------------------
    summary_branch.to_csv(graph_dir / "branch_importance_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(summary_branch["Branch"].astype(str), summary_branch["meanSHAP"])
    plt.ylabel("Mean absolute SHAP")
    plt.title(f"{trait} - Branch importance - no soil")
    plt.xticks(rotation=30)
    savefig(graph_dir / "branch_importance.png")

    # -------------------------------------------------------------------------
    # 6. Branch importance with SNP_total
    # -------------------------------------------------------------------------
    summary_branch_genomic.to_csv(
        graph_dir / "branch_importance_with_snp_total_summary.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        summary_branch_genomic["Branch"].astype(str),
        summary_branch_genomic["meanSHAP"],
    )
    plt.ylabel("Mean absolute SHAP")
    plt.title(f"{trait} - Branch importance with SNP_total - no soil")
    plt.xticks(rotation=30)
    savefig(graph_dir / "branch_importance_with_snp_total.png")

    # -------------------------------------------------------------------------
    # 7. Mapped vs unmapped
    # -------------------------------------------------------------------------
    summary_mu.to_csv(graph_dir / "mapped_vs_unmapped_summary.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.bar(summary_mu["Category"].astype(str), summary_mu["meanSHAP"])
    plt.ylabel("Mean absolute SHAP")
    plt.title(f"{trait} - Mapped vs unmapped SNP importance - no soil")
    savefig(graph_dir / "mapped_vs_unmapped.png")

    # -------------------------------------------------------------------------
    # 8. Weather by period
    # -------------------------------------------------------------------------
    summary_w_period.to_csv(graph_dir / "weather_by_period_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(summary_w_period["Period"].astype(str), summary_w_period["meanSHAP"])
    plt.ylabel("Mean feature SHAP")
    plt.title(f"{trait} - Weather importance by period - no soil")
    savefig(graph_dir / "weather_by_period.png")

    # -------------------------------------------------------------------------
    # 9. Weather by variable
    # -------------------------------------------------------------------------
    summary_w_var.to_csv(graph_dir / "weather_by_variable_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(summary_w_var["Variable"].astype(str), summary_w_var["meanSHAP"])
    plt.ylabel("Mean feature SHAP")
    plt.title(f"{trait} - Weather importance by variable - no soil")
    savefig(graph_dir / "weather_by_variable.png")

    # -------------------------------------------------------------------------
    # 10. Weather by statistic
    # -------------------------------------------------------------------------
    summary_w_stat.to_csv(graph_dir / "weather_by_statistic_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(summary_w_stat["Statistic"].astype(str), summary_w_stat["meanSHAP"])
    plt.ylabel("Mean feature SHAP")
    plt.title(f"{trait} - Weather importance by statistic - no soil")
    savefig(graph_dir / "weather_by_statistic.png")

    # -------------------------------------------------------------------------
    # 11. Top weather features
    # -------------------------------------------------------------------------
    top30_weather = summary_weather.head(30).copy()
    top30_weather.to_csv(graph_dir / "top30_weather_features.csv", index=False)

    plot_barh(
        summary_weather,
        "meanSHAP",
        "Feature",
        f"{trait} - Top 30 weather features - no soil",
        "Mean absolute SHAP",
        graph_dir / "top30_weather_features.png",
        topn=30,
    )

    # -------------------------------------------------------------------------
    # 12. Top PCA features
    # -------------------------------------------------------------------------
    plot_barh(
        summary_pca,
        "meanSHAP",
        "Feature",
        f"{trait} - Top PCA features - no soil",
        "Mean absolute SHAP",
        graph_dir / "top_pca_features.png",
        topn=min(20, len(summary_pca)),
    )

    # -------------------------------------------------------------------------
    # 13. Scatter meanSHAP vs top20_count for SNPs
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.scatter(
        summary_allsnp["top20_count"],
        summary_allsnp["meanSHAP"],
        alpha=0.6,
        s=25,
    )
    plt.xlabel("top20_count")
    plt.ylabel("Mean absolute SHAP")
    plt.title(f"{trait} - SNP meanSHAP vs top20_count - no soil")
    plt.grid(True, alpha=0.25)
    savefig(graph_dir / "scatter_snp_meanSHAP_vs_top20count.png")

    # -------------------------------------------------------------------------
    # 14. Scatter meanSHAP vs n_folds for SNPs
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.scatter(
        summary_allsnp["n_folds"],
        summary_allsnp["meanSHAP"],
        alpha=0.6,
        s=25,
    )
    plt.xlabel("n_folds")
    plt.ylabel("Mean absolute SHAP")
    plt.title(f"{trait} - SNP meanSHAP vs n_folds - no soil")
    plt.grid(True, alpha=0.25)
    savefig(graph_dir / "scatter_snp_meanSHAP_vs_nfolds.png")

    # -------------------------------------------------------------------------
    # 15. Heatmap-like matrix for top SNPs across splits
    # -------------------------------------------------------------------------
    if len(top30_snp) > 0 and len(master_allsnp) > 0:
        top30_snp_names = top30_snp["SNP"].astype(str).tolist()

        heat_df = master_allsnp[
            master_allsnp["SNP"].astype(str).isin(top30_snp_names)
        ].copy()

        heat_mat = heat_df.pivot(
            index="SNP",
            columns="Split",
            values="mean_abs_SHAP",
        )

        heat_mat = heat_mat.reindex(top30_snp_names)
        heat_mat.to_csv(graph_dir / "top30_snps_heatmap_matrix.csv")

        plt.figure(figsize=(14, 10))
        plt.imshow(heat_mat.values, aspect="auto")
        plt.colorbar(label="mean_abs_SHAP")
        plt.yticks(np.arange(len(heat_mat.index)), heat_mat.index)
        plt.xticks(np.arange(len(heat_mat.columns)), heat_mat.columns, rotation=90)
        plt.title(f"{trait} - Top 30 SNPs across splits - no soil")
        savefig(graph_dir / "top30_snps_heatmap.png")

    # -------------------------------------------------------------------------
    # 16. Heatmap-like matrix for top genes across splits
    # -------------------------------------------------------------------------
    if len(master_gene) > 0 and len(summary_gene) > 0:
        top30_gene_names = summary_gene.head(30)["Gene"].astype(str).tolist()

        gene_heat_df = master_gene[
            master_gene["Gene"].astype(str).isin(top30_gene_names)
        ].copy()

        gene_heat_mat = gene_heat_df.pivot(
            index="Gene",
            columns="Split",
            values="gene_mean_abs_SHAP",
        )

        gene_heat_mat = gene_heat_mat.reindex(top30_gene_names)
        gene_heat_mat.to_csv(graph_dir / "top30_genes_heatmap_matrix.csv")

        plt.figure(figsize=(14, 10))
        plt.imshow(gene_heat_mat.values, aspect="auto")
        plt.colorbar(label="gene_mean_abs_SHAP")
        plt.yticks(np.arange(len(gene_heat_mat.index)), gene_heat_mat.index)
        plt.xticks(np.arange(len(gene_heat_mat.columns)), gene_heat_mat.columns, rotation=90)
        plt.title(f"{trait} - Top 30 genes across splits - no soil")
        savefig(graph_dir / "top30_genes_heatmap.png")

    # -------------------------------------------------------------------------
    # 17. Top stable genes n_folds >= 5
    # -------------------------------------------------------------------------
    if len(summary_gene_stable5) > 0:
        top30 = summary_gene_stable5.head(30).copy()

        plt.figure(figsize=(10, 8))
        plt.barh(top30["Gene"].astype(str)[::-1], top30["meanSHAP"][::-1])
        plt.xlabel("Mean aggregated gene SHAP")
        plt.title(f"{trait} - Top stable genes n_folds >= 5 - no soil")
        savefig(graph_dir / "top30_stable_genes_nfolds_ge5.png")

    # -------------------------------------------------------------------------
    # 18. Top stable genes n_folds >= 10
    # -------------------------------------------------------------------------
    if len(summary_gene_stable10) > 0:
        top30 = summary_gene_stable10.head(30).copy()

        plt.figure(figsize=(10, 8))
        plt.barh(top30["Gene"].astype(str)[::-1], top30["meanSHAP"][::-1])
        plt.xlabel("Mean aggregated gene SHAP")
        plt.title(f"{trait} - Top stable genes n_folds >= 10 - no soil")
        savefig(graph_dir / "top30_stable_genes_nfolds_ge10.png")

    # -------------------------------------------------------------------------
    # 19. Text summary
    # -------------------------------------------------------------------------
    with open(paths["summary_txt"], "w", encoding="utf-8") as f:
        f.write(f"=== SHAP SUMMARY FOR {MODEL_NAME} / {trait} ===\n\n")
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

        f.write("Top 20 mapped SNPs:\n")
        f.write(summary_mapped.head(20).to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 unmapped SNPs:\n")
        f.write(summary_unmapped.head(20).to_string(index=False))
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

        if len(summary_gene) > 0:
            f.write("Summary of gene meanSHAP:\n")
            f.write(summary_gene["meanSHAP"].describe().to_string())
            f.write("\n\n")

            f.write("Summary of gene n_folds:\n")
            f.write(summary_gene["n_folds"].describe().to_string())
            f.write("\n\n")

    print("SHAP plots and summaries saved in:")
    print(graph_dir)

    return {
        "Trait": trait,
        "GraphDir": str(graph_dir),
        "n_snp_summary": len(summary_allsnp),
        "n_gene_summary": len(summary_gene),
        "n_weather_summary": len(summary_weather),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("Q6 - GRAPHS AFTER Q5 NO-SOIL NEW TRAITS")
    print("=" * 100)

    rows = []

    for trait in TRAITS:
        rows.append(process_one_trait(trait))

    summary = pd.DataFrame(rows)

    summary_file = BASE_MODEL_DIR / "Q6_graphs_summary_all_traits.csv"
    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 100)
    print("Q6 completed.")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("\nSaved summary:")
    print(summary_file)


if __name__ == "__main__":
    main()
