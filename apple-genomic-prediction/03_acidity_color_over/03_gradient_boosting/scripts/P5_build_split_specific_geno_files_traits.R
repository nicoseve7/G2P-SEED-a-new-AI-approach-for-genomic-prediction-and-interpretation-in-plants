# ==================================================
# P5_build_split_specific_geno_files_traits.R
#
# Costruisce i file genomici split-specifici
# per i nuovi tratti:
#   - Acidity
#   - Color_over
#
# Logica:
#   per ogni trait e split:
#     - prende top 1000 SNP dal GB
#     - aggiunge eventuali SNP GWAS del trait
#     - estrae genotipi dal GDS
#     - ricodifica 0/1/2 -> 1/0.5/0 come nella pipeline Harvest_date
#
# Input:
#   Output/Intermediate/GB_feature_selection/feature_selection_results_acidity.csv
#   Output/Intermediate/GB_feature_selection/feature_selection_results_color_over.csv
#   Output/Intermediate/SNPs_final_2022.gds oppure Output/SNPs_final_2022.gds
#   Input/SupTable3_SNPS_GWAS.xls opzionale
#
# Output:
#   Output/Intermediate/geno_files/Acidity/geno_CV*_Split*.csv
#   Output/Intermediate/geno_files/Color_over/geno_CV*_Split*.csv
# ==================================================

rm(list = ls())

cat("=== P5: BUILD SPLIT-SPECIFIC GENO FILES FOR NEW TRAITS ===\n\n")

library(SeqArray)
library(SNPRelate)
library(dplyr)
library(readr)
library(readxl)

# =============================================================================
# SETTINGS
# =============================================================================

TRAITS <- c("Acidity", "Color_over")

TOP_N_GB <- 1000

GB_DIR <- "Output/Intermediate/GB_feature_selection"

GDS_CANDIDATES <- c(
  "Output/Intermediate/SNPs_final_2022.gds",
  "Output/SNPs_final_2022.gds"
)

GWAS_FILE <- "Input/SupTable3_SNPS_GWAS.xls"

GENO_BASE_DIR <- "Output/Intermediate/geno_files"
dir.create(GENO_BASE_DIR, showWarnings = FALSE, recursive = TRUE)

# Trait renaming usato anche nel vecchio D7.
# Serve perché nella tabella GWAS i nomi originali possono essere diversi/ordinati.
RENAMED_TRAITS_ORD <- c(
  "Flowering_begin",
  "Flowering_intensity",
  "Firmness",
  "Harvest_date",
  "Fruit_number",
  "Color_over",
  "Russet_freq_all",
  "Fruit_weight_single",
  "Sugar",
  "Acidity",
  "Fruit_weight"
)


# =============================================================================
# HELPERS
# =============================================================================

trait_label <- function(trait) {
  tolower(trait)
}

find_gds_path <- function() {
  gds_path <- GDS_CANDIDATES[file.exists(GDS_CANDIDATES)][1]
  
  if (is.na(gds_path)) {
    stop(paste0(
      "Nessun file GDS trovato nei path attesi:\n",
      paste(GDS_CANDIDATES, collapse = "\n")
    ))
  }
  
  return(gds_path)
}

load_gwas_snps <- function() {
  
  if (!file.exists(GWAS_FILE)) {
    cat("File GWAS non trovato, procedo senza SNP GWAS aggiuntivi.\n\n")
    return(data.frame(
      Trait = character(0),
      SNP = character(0),
      stringsAsFactors = FALSE
    ))
  }
  
  cat("Caricamento GWAS SNPs...\n")
  
  gwas_snps <- read_xls(GWAS_FILE)[, 1:2]
  gwas_snps <- as.data.frame(gwas_snps)
  
  if (!all(c("Trait", "SNP") %in% colnames(gwas_snps))) {
    stop(paste0(
      "Mi aspettavo colonne Trait e SNP in ",
      GWAS_FILE,
      ". Colonne trovate: ",
      paste(colnames(gwas_snps), collapse = ", ")
    ))
  }
  
  traits <- unique(gwas_snps$Trait)
  traits_ord <- traits[order(traits)]
  
  if (length(traits_ord) != length(RENAMED_TRAITS_ORD)) {
    cat("[WARNING] Numero trait in SupTable3 diverso da RENAMED_TRAITS_ORD.\n")
    cat("[WARNING] Provo comunque ad applicare la conversione per ordine.\n")
    cat("Trait originali ordinati:\n")
    print(traits_ord)
    cat("Trait rinominati attesi:\n")
    print(RENAMED_TRAITS_ORD)
    cat("\n")
  }
  
  n_map <- min(length(traits_ord), length(RENAMED_TRAITS_ORD))
  
  conversion_table <- data.frame(
    traits_ord = traits_ord[seq_len(n_map)],
    renamed_traits_ord = RENAMED_TRAITS_ORD[seq_len(n_map)],
    stringsAsFactors = FALSE
  )
  
  gwas_snps <- merge(
    gwas_snps,
    conversion_table,
    by.x = "Trait",
    by.y = "traits_ord",
    all.x = TRUE
  )
  
  gwas_snps <- distinct(gwas_snps)
  
  out <- gwas_snps %>%
    transmute(
      Trait = renamed_traits_ord,
      SNP = as.character(SNP)
    ) %>%
    filter(!is.na(Trait), !is.na(SNP)) %>%
    distinct()
  
  cat("GWAS SNPs caricati:\n")
  print(out %>% group_by(Trait) %>% summarise(n_gwas_snps = n_distinct(SNP), .groups = "drop"))
  cat("\n")
  
  return(out)
}

load_gds_info <- function(gds_path) {
  
  cat("Apro GDS per leggere SNP IDs e genotipi...\n")
  gds <- seqOpen(gds.fn = gds_path)
  
  snp_idx_gds <- seqGetData(gds, var.name = "annotation/id")
  genotypes <- seqGetData(gds, var.name = "sample.id")
  
  seqClose(gds)
  
  cat("Numero SNP nel GDS:", length(snp_idx_gds), "\n")
  cat("Numero genotipi nel GDS:", length(genotypes), "\n\n")
  
  return(list(
    snp_idx_gds = snp_idx_gds,
    genotypes = genotypes
  ))
}

process_one_trait <- function(trait, gds_path, snp_idx_gds, genotypes, gwas_table) {
  
  label <- trait_label(trait)
  
  cat("\n", paste(rep("#", 80), collapse = ""), "\n", sep = "")
  cat("Processing trait: ", trait, "\n", sep = "")
  cat(paste(rep("#", 80), collapse = ""), "\n\n", sep = "")
  
  importance_file <- file.path(
    GB_DIR,
    paste0("feature_selection_results_", label, ".csv")
  )
  
  if (!file.exists(importance_file)) {
    stop(paste0("File importance non trovato:\n", importance_file))
  }
  
  save_dir <- file.path(GENO_BASE_DIR, trait)
  dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
  
  cat("Caricamento feature selection results...\n")
  importance_all <- read_csv(importance_file, show_col_types = FALSE)
  
  required_cols <- c("Trait", "Split", "SNP", "importance")
  missing_cols <- setdiff(required_cols, colnames(importance_all))
  
  if (length(missing_cols) > 0) {
    stop(paste0(
      "Nel file ", importance_file, " mancano colonne: ",
      paste(missing_cols, collapse = ", "),
      "\nColonne trovate: ",
      paste(colnames(importance_all), collapse = ", ")
    ))
  }
  
  importance_all <- importance_all %>%
    filter(Trait == trait) %>%
    mutate(
      Split = as.character(Split),
      SNP = as.character(SNP),
      importance = as.numeric(importance)
    )
  
  cat("Numero righe risultati boosting:", nrow(importance_all), "\n")
  cat("Numero split trovati:", n_distinct(importance_all$Split), "\n\n")
  
  if (nrow(importance_all) == 0) {
    stop(paste0("Nessun risultato boosting per trait: ", trait))
  }
  
  snps_gwas_add <- gwas_table %>%
    filter(Trait == trait) %>%
    pull(SNP) %>%
    unique()
  
  cat("Numero GWAS SNP aggiunti per ", trait, ": ", length(snps_gwas_add), "\n\n", sep = "")
  
  splits <- sort(unique(importance_all$Split))
  
  summary_list <- list()
  
  for (split in splits) {
    
    cat("Processing split: ", split, "\n", sep = "")
    
    # -------------------------------------------------------------------------
    # Top GB SNP per questo split
    # -------------------------------------------------------------------------
    snps_important_df <- importance_all %>%
      filter(Split == split) %>%
      arrange(desc(importance))
    
    n_available <- nrow(snps_important_df)
    n_take <- min(TOP_N_GB, n_available)
    
    snps_important <- snps_important_df$SNP[seq_len(n_take)]
    
    # aggiunta eventuale GWAS
    snps_important_all <- unique(c(snps_important, snps_gwas_add))
    
    # match con GDS
    snp.id <- which(snp_idx_gds %in% snps_important_all)
    matched_snps <- snp_idx_gds[snp.id]
    
    missing_from_gds <- setdiff(snps_important_all, matched_snps)
    
    if (length(snp.id) == 0) {
      warning(paste0(
        "Nessuno SNP trovato nel GDS per trait ",
        trait,
        ", split ",
        split
      ))
      next
    }
    
    # -------------------------------------------------------------------------
    # Estrazione matrice dal GDS
    # -------------------------------------------------------------------------
    gds <- seqOpen(gds.fn = gds_path)
    
    geno_split <- snpgdsGetGeno(
      gds,
      snp.id = snp.id,
      snpfirstdim = FALSE
    )
    
    seqClose(gds)
    
    # codifica paper-inspired:
    # 0/1/2 -> 1/0.5/0
    geno_split <- (2 - geno_split) / 2
    
    colnames(geno_split) <- matched_snps
    rownames(geno_split) <- paste("G", genotypes, sep = "_")
    
    # -------------------------------------------------------------------------
    # Salvataggio
    # -------------------------------------------------------------------------
    out_file <- file.path(save_dir, paste0("geno_", split, ".csv"))
    write.csv(geno_split, out_file, quote = FALSE, row.names = TRUE)
    
    summary_list[[split]] <- data.frame(
      Trait = trait,
      Split = split,
      n_boosting_selected_available = n_available,
      n_top_boosting_used = n_take,
      n_gwas_added_trait = length(snps_gwas_add),
      n_requested_unique_snps = length(snps_important_all),
      n_final_unique_snps_in_gds = length(matched_snps),
      n_missing_requested_snps_from_gds = length(missing_from_gds),
      n_genotypes = nrow(geno_split),
      output_file = out_file,
      stringsAsFactors = FALSE
    )
    
    cat("  SNP selezionati dal boosting disponibili:", n_available, "\n")
    cat("  Top SNP usati:", n_take, "\n")
    cat("  SNP GWAS aggiunti:", length(snps_gwas_add), "\n")
    cat("  SNP richiesti unici:", length(snps_important_all), "\n")
    cat("  SNP finali nel file:", length(matched_snps), "\n")
    cat("  SNP richiesti mancanti nel GDS:", length(missing_from_gds), "\n")
    cat("  Genotipi nel file:", nrow(geno_split), "\n\n")
  }
  
  summary_df <- bind_rows(summary_list)
  
  summary_file <- file.path(
    save_dir,
    paste0("geno_split_summary_", label, ".csv")
  )
  
  report_file <- file.path(
    save_dir,
    paste0("geno_split_report_", label, ".txt")
  )
  
  write_csv(summary_df, summary_file)
  
  sink(report_file)
  
  cat("=== P5 SPLIT-SPECIFIC GENO FILES REPORT ===\n\n")
  
  cat("Trait:", trait, "\n")
  cat("GDS usato:", gds_path, "\n")
  cat("Importance file:", importance_file, "\n\n")
  
  cat("Numero split processati:", length(splits), "\n")
  cat("TOP_N_GB:", TOP_N_GB, "\n\n")
  
  cat("Numero GWAS SNP aggiunti per trait:", length(snps_gwas_add), "\n")
  if (length(snps_gwas_add) > 0) {
    cat("GWAS SNPs:\n")
    print(snps_gwas_add)
    cat("\n")
  }
  
  cat("Riassunto split:\n")
  print(summary_df)
  cat("\n")
  
  sink()
  
  cat("File salvati per ", trait, ":\n", sep = "")
  cat("- ", save_dir, "/geno_<split>.csv\n", sep = "")
  cat("- ", summary_file, "\n", sep = "")
  cat("- ", report_file, "\n\n", sep = "")
  
  return(data.frame(
    Trait = trait,
    n_splits_processed = nrow(summary_df),
    mean_final_snps = mean(summary_df$n_final_unique_snps_in_gds),
    min_final_snps = min(summary_df$n_final_unique_snps_in_gds),
    max_final_snps = max(summary_df$n_final_unique_snps_in_gds),
    mean_missing_from_gds = mean(summary_df$n_missing_requested_snps_from_gds),
    stringsAsFactors = FALSE
  ))
}


# =============================================================================
# MAIN
# =============================================================================

gds_path <- find_gds_path()

cat("Uso GDS:", gds_path, "\n\n")

gds_info <- load_gds_info(gds_path)

gwas_table <- load_gwas_snps()

global_summary_list <- list()

for (trait in TRAITS) {
  global_summary_list[[trait]] <- process_one_trait(
    trait = trait,
    gds_path = gds_path,
    snp_idx_gds = gds_info$snp_idx_gds,
    genotypes = gds_info$genotypes,
    gwas_table = gwas_table
  )
}

global_summary <- bind_rows(global_summary_list)

global_summary_file <- file.path(
  GENO_BASE_DIR,
  "P5_geno_split_summary_new_traits_global.csv"
)

write_csv(global_summary, global_summary_file)

global_report_file <- file.path(
  GENO_BASE_DIR,
  "P5_geno_split_report_new_traits_global.txt"
)

sink(global_report_file)

cat("=== P5 GLOBAL REPORT ===\n\n")
cat("Traits processed:\n")
print(TRAITS)
cat("\n\n")

cat("GDS used:\n")
cat(gds_path, "\n\n")

cat("TOP_N_GB:", TOP_N_GB, "\n\n")

cat("Global summary:\n")
print(global_summary)
cat("\n")

sink()

cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("P5 completed.\n")
cat("Global summary:\n")
print(global_summary)
cat("\nSaved:\n")
cat(global_summary_file, "\n")
cat(global_report_file, "\n")
cat(paste(rep("=", 80), collapse = ""), "\n", sep = "")