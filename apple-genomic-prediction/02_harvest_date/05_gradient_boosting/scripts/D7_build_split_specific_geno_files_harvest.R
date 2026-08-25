# ==================================================
# D7_build_split_specific_geno_files_harvest.R
# Costruisce i file genomici split-specifici per Harvest_date
# seguendo la logica paper-inspired di C_PreProcess_Data_2.R
# ==================================================

rm(list = ls())

cat("=== STEP D7: BUILD SPLIT-SPECIFIC GENO FILES FOR HARVEST_DATE ===\n\n")

library(SeqArray)
library(SNPRelate)
library(dplyr)
library(readr)
library(readxl)

# -------------------------------
# 1. Percorsi file
# -------------------------------
gb_dir <- "02_harvest_date/05_gradient_boosting/output"

importance_file <- file.path(
  gb_dir,
  "feature_selection_results_harvest_date.csv"
)

gds_path <- paste0(
  "01_common_genomic_preprocessing/output/",
  "SNPs_final_2022.gds"
)

gwas_file <- "data/raw/external/SupTable3_SNPS_GWAS.xls"

save_dir <- file.path(
  gb_dir,
  "geno_files",
  "Harvest_date"
)

dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
# -------------------------------
# 2. Caricamento risultati boosting
# -------------------------------
cat("Caricamento feature selection results...\n")
importance_all <- read_csv(importance_file, show_col_types = FALSE)

cat("Numero righe risultati boosting:", nrow(importance_all), "\n")
cat("Numero split trovati:", n_distinct(importance_all$Split), "\n\n")

# -------------------------------
# 3. Gestione GDS
# -------------------------------
if (!file.exists(gds_path)) {
  stop(
    paste0(
      "GDS file not found: ",
      gds_path,
      "\nRun the common genomic preprocessing first."
    )
  )
}

cat("Uso GDS:", gds_path, "\n\n")

gds <- seqOpen(gds.fn = gds_path)
snp_idx_gds <- seqGetData(gds, var.name = "annotation/id")
genotypes <- seqGetData(gds, var.name = "sample.id")
seqClose(gds)

cat("Numero SNP nel GDS:", length(snp_idx_gds), "\n")
cat("Numero genotipi nel GDS:", length(genotypes), "\n\n")

# -------------------------------
# 4. Caricamento GWAS SNPs (se presente)
# -------------------------------
snps_gwas_add <- character(0)

if (file.exists(gwas_file)) {
  cat("Caricamento GWAS SNPs...\n")
  gwas_snps <- read_xls(gwas_file)[, 1:2]

  # stessa logica del repo per rinominare i trait
  traits <- unique(gwas_snps$Trait)
  traits_ord <- traits[order(traits)]
  renamed_traits_ord <- c(
    "Flowering_begin", "Flowering_intensity", "Firmness", "Harvest_date",
    "Fruit_number", "Color_over", "Russet_freq_all", "Fruit_weight_single",
    "Sugar", "Acidity", "Fruit_weight"
  )

  conversion_table <- as.data.frame(cbind(traits_ord, renamed_traits_ord))
  gwas_snps <- merge(
    gwas_snps,
    conversion_table,
    by.x = "Trait",
    by.y = "traits_ord",
    all.x = TRUE
  )

  gwas_snps <- distinct(gwas_snps)
  gwas_snps <- gwas_snps[, -1]

  snps_gwas_add <- gwas_snps[gwas_snps$renamed_traits_ord == "Harvest_date", ]$SNP
  snps_gwas_add <- unique(snps_gwas_add)

  cat("Numero GWAS SNP aggiunti per Harvest_date:", length(snps_gwas_add), "\n\n")
} else {
  cat("File GWAS non trovato, procedo senza SNP GWAS aggiuntivi.\n\n")
}

# -------------------------------
# 5. Iterazione sugli split
# -------------------------------
splits <- unique(importance_all$Split)
splits <- sort(splits)

summary_list <- list()

for (split in splits) {
  cat("Processing split:", split, "\n")

  gds <- seqOpen(gds.fn = gds_path)

  # SNP importanti per questo split
  snps_important_df <- importance_all %>%
    filter(Trait == "Harvest_date", Split == split) %>%
    arrange(desc(importance))

  n_available <- nrow(snps_important_df)
  n_take <- min(1000, n_available)

  snps_important <- snps_important_df$SNP[1:n_take]

  # aggiunta SNP GWAS
  snps_important_all <- unique(c(snps_important, snps_gwas_add))

  # match con GDS
  snp.id <- which(snp_idx_gds %in% snps_important_all)

  # estrazione matrice
  geno_split <- snpgdsGetGeno(gds, snp.id = snp.id, snpfirstdim = FALSE)

  # codifica paper-inspired:
  # 0/1/2 -> 1/0.5/0
  geno_split <- (2 - geno_split) / 2

  # naming
  colnames(geno_split) <- snp_idx_gds[snp.id]
  rownames(geno_split) <- paste("G", genotypes, sep = "_")

  # salvataggio
  out_file <- file.path(save_dir, paste0("geno_", split, ".csv"))
  write.csv(geno_split, out_file, quote = FALSE, row.names = TRUE)

  seqClose(gds)

  summary_list[[split]] <- data.frame(
    Split = split,
    n_boosting_selected = n_available,
    n_top_boosting_used = n_take,
    n_gwas_added = length(snps_gwas_add),
    n_final_unique_snps = length(snp.id),
    n_genotypes = nrow(geno_split),
    stringsAsFactors = FALSE
  )

  cat("  SNP selezionati dal boosting:", n_available, "\n")
  cat("  Top SNP usati:", n_take, "\n")
  cat("  SNP finali nel file:", length(snp.id), "\n")
  cat("  Genotipi nel file:", nrow(geno_split), "\n\n")
}

summary_df <- bind_rows(summary_list)

# -------------------------------
# 6. Salvataggio summary e report
# -------------------------------
write_csv(
  summary_df,
  file.path(save_dir, "geno_split_summary_harvest_date.csv")
)

sink(
  file.path(save_dir, "geno_split_report_harvest_date.txt")
)

cat("=== STEP D7: SPLIT-SPECIFIC GENO FILES REPORT ===\n\n")
cat("GDS usato:", gds_path, "\n\n")
cat("Numero split processati:", length(splits), "\n\n")

cat("Numero GWAS SNP aggiunti per Harvest_date:", length(snps_gwas_add), "\n\n")

cat("Riassunto split:\n")
print(summary_df)
cat("\n")

sink()

cat("File salvati:\n")
cat("- ", file.path(save_dir, "geno_<split>.csv"), "\n")
cat("- ", file.path(save_dir, "geno_split_summary_harvest_date.csv"), "\n")
cat("- ", file.path(save_dir, "geno_split_report_harvest_date.txt"), "\n")
