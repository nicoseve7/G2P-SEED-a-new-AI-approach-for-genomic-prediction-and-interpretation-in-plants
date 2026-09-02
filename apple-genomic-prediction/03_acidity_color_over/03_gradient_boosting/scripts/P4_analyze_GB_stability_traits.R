# ==================================================
# P4_analyze_GB_stability_traits.R
#
# Analisi stabilità SNP selezionati dal Gradient Boosting
# per i nuovi tratti:
#   - Acidity
#   - Color_over
#
# Input attesi da P3:
#   Output/Intermediate/GB_feature_selection/feature_selection_results_acidity.csv
#   Output/Intermediate/GB_feature_selection/feature_selection_summary_acidity.csv
#
#   Output/Intermediate/GB_feature_selection/feature_selection_results_color_over.csv
#   Output/Intermediate/GB_feature_selection/feature_selection_summary_color_over.csv
#
# Output:
#   feature_selection_snp_stability_<trait>.csv
#   feature_selection_overlap_summary_<trait>.csv
#   feature_selection_top50_stable_snps_<trait>.csv
#   feature_selection_split_counts_<trait>.csv
#   feature_selection_stability_report_<trait>.txt
# ==================================================

rm(list = ls())

cat("=== P4: ANALYZE GB FEATURE SELECTION STABILITY FOR NEW TRAITS ===\n\n")

library(readr)
library(dplyr)

# =============================================================================
# SETTINGS
# =============================================================================

TRAITS <- c("Acidity", "Color_over")

GB_DIR <- "Output/Intermediate/GB_feature_selection"

THRESHOLDS <- c(1, 2, 5, 10, 15, 20, 25)


# =============================================================================
# HELPERS
# =============================================================================

trait_label <- function(trait) {
  tolower(trait)
}

process_one_trait <- function(trait) {
  
  label <- trait_label(trait)
  
  cat("\n", paste(rep("#", 80), collapse = ""), "\n", sep = "")
  cat("Processing trait: ", trait, "\n", sep = "")
  cat(paste(rep("#", 80), collapse = ""), "\n\n", sep = "")
  
  fs_file <- file.path(
    GB_DIR,
    paste0("feature_selection_results_", label, ".csv")
  )
  
  summary_file <- file.path(
    GB_DIR,
    paste0("feature_selection_summary_", label, ".csv")
  )
  
  if (!file.exists(fs_file)) {
    stop(paste0("File risultati GB non trovato:\n", fs_file))
  }
  
  if (!file.exists(summary_file)) {
    stop(paste0("File summary GB non trovato:\n", summary_file))
  }
  
  cat("Caricamento risultati feature selection...\n")
  fs <- read_csv(fs_file, show_col_types = FALSE)
  
  cat("Caricamento summary split...\n")
  fs_summary <- read_csv(summary_file, show_col_types = FALSE)
  
  cat("Caricamento completato.\n\n")
  
  required_cols <- c("Trait", "Split", "SNP", "importance", "n_selected")
  missing_cols <- setdiff(required_cols, colnames(fs))
  
  if (length(missing_cols) > 0) {
    stop(paste0(
      "Nel file ", fs_file, " mancano colonne: ",
      paste(missing_cols, collapse = ", "),
      "\nColonne trovate: ",
      paste(colnames(fs), collapse = ", ")
    ))
  }
  
  fs <- fs %>%
    filter(Trait == trait) %>%
    mutate(
      SNP = as.character(SNP),
      Split = as.character(Split),
      importance = as.numeric(importance)
    )
  
  cat("Numero righe feature selection:", nrow(fs), "\n")
  cat("Numero split unici:", n_distinct(fs$Split), "\n")
  cat("Numero SNP unici selezionati:", n_distinct(fs$SNP), "\n\n")
  
  if (nrow(fs) == 0) {
    stop(paste0("Nessun risultato GB trovato per trait: ", trait))
  }
  
  # ---------------------------------------------------------------------------
  # Stabilità SNP tra split
  # ---------------------------------------------------------------------------
  cat("Calcolo stabilità SNP...\n")
  
  snp_stability <- fs %>%
    group_by(SNP) %>%
    summarise(
      Trait = first(Trait),
      n_splits_selected = n_distinct(Split),
      mean_importance = mean(importance, na.rm = TRUE),
      median_importance = median(importance, na.rm = TRUE),
      max_importance = max(importance, na.rm = TRUE),
      min_importance = min(importance, na.rm = TRUE),
      selected_splits = paste(sort(unique(Split)), collapse = ","),
      .groups = "drop"
    ) %>%
    arrange(desc(n_splits_selected), desc(mean_importance), SNP)
  
  # ---------------------------------------------------------------------------
  # Overlap summary: quanti SNP compaiono in almeno k split
  # ---------------------------------------------------------------------------
  overlap_summary <- data.frame(
    Trait = trait,
    min_splits = THRESHOLDS,
    n_snps = sapply(THRESHOLDS, function(k) {
      sum(snp_stability$n_splits_selected >= k)
    }),
    stringsAsFactors = FALSE
  )
  
  overlap_summary$frequency_threshold <- overlap_summary$min_splits / 25
  
  # ---------------------------------------------------------------------------
  # Top SNP più stabili
  # ---------------------------------------------------------------------------
  top_stable_snps <- snp_stability %>%
    arrange(desc(n_splits_selected), desc(mean_importance), SNP) %>%
    slice_head(n = 50)
  
  # ---------------------------------------------------------------------------
  # Numero SNP selezionati per split
  # ---------------------------------------------------------------------------
  split_counts <- fs %>%
    group_by(Trait, Split) %>%
    summarise(
      n_selected_unique = n_distinct(SNP),
      n_rows = n(),
      .groups = "drop"
    ) %>%
    arrange(Split)
  
  # ---------------------------------------------------------------------------
  # Matrice presenza/assenza SNP x split
  # ---------------------------------------------------------------------------
  fs_presence <- fs %>%
    distinct(SNP, Split) %>%
    mutate(selected = 1)
  
  presence_wide <- fs_presence %>%
    tidyr::pivot_wider(
      names_from = Split,
      values_from = selected,
      values_fill = 0
    ) %>%
    arrange(SNP)
  
  # ---------------------------------------------------------------------------
  # Salvataggi
  # ---------------------------------------------------------------------------
  out_snp_stability <- file.path(
    GB_DIR,
    paste0("feature_selection_snp_stability_", label, ".csv")
  )
  
  out_overlap <- file.path(
    GB_DIR,
    paste0("feature_selection_overlap_summary_", label, ".csv")
  )
  
  out_top50 <- file.path(
    GB_DIR,
    paste0("feature_selection_top50_stable_snps_", label, ".csv")
  )
  
  out_split_counts <- file.path(
    GB_DIR,
    paste0("feature_selection_split_counts_", label, ".csv")
  )
  
  out_presence <- file.path(
    GB_DIR,
    paste0("feature_selection_presence_matrix_", label, ".csv")
  )
  
  out_report <- file.path(
    GB_DIR,
    paste0("feature_selection_stability_report_", label, ".txt")
  )
  
  write_csv(snp_stability, out_snp_stability)
  write_csv(overlap_summary, out_overlap)
  write_csv(top_stable_snps, out_top50)
  write_csv(split_counts, out_split_counts)
  write_csv(presence_wide, out_presence)
  
  sink(out_report)
  
  cat("=== P4 FEATURE SELECTION STABILITY REPORT ===\n\n")
  
  cat("Trait:", trait, "\n\n")
  
  cat("Input files:\n")
  cat("- ", fs_file, "\n", sep = "")
  cat("- ", summary_file, "\n\n", sep = "")
  
  cat("Numero righe feature selection:", nrow(fs), "\n")
  cat("Numero split unici:", n_distinct(fs$Split), "\n")
  cat("Numero SNP unici selezionati:", n_distinct(fs$SNP), "\n\n")
  
  cat("SUMMARY PER SPLIT\n")
  print(split_counts)
  cat("\n\n")
  
  cat("OVERLAP SUMMARY\n")
  print(overlap_summary)
  cat("\n\n")
  
  cat("TOP 20 SNP PIU' STABILI\n")
  print(head(top_stable_snps, 20))
  cat("\n\n")
  
  cat("Prime 20 righe tabella stabilità SNP\n")
  print(head(snp_stability, 20))
  cat("\n")
  
  sink()
  
  cat("File salvati per ", trait, ":\n", sep = "")
  cat("- ", out_snp_stability, "\n", sep = "")
  cat("- ", out_overlap, "\n", sep = "")
  cat("- ", out_top50, "\n", sep = "")
  cat("- ", out_split_counts, "\n", sep = "")
  cat("- ", out_presence, "\n", sep = "")
  cat("- ", out_report, "\n\n", sep = "")
  
  return(data.frame(
    Trait = trait,
    n_rows_feature_selection = nrow(fs),
    n_splits = n_distinct(fs$Split),
    n_unique_snps = n_distinct(fs$SNP),
    n_snps_selected_ge_5_splits = sum(snp_stability$n_splits_selected >= 5),
    n_snps_selected_ge_10_splits = sum(snp_stability$n_splits_selected >= 10),
    n_snps_selected_ge_20_splits = sum(snp_stability$n_splits_selected >= 20),
    stringsAsFactors = FALSE
  ))
}


# =============================================================================
# MAIN
# =============================================================================

global_rows <- list()

for (trait in TRAITS) {
  global_rows[[trait]] <- process_one_trait(trait)
}

global_summary <- bind_rows(global_rows)

global_summary_file <- file.path(
  GB_DIR,
  "P4_feature_selection_stability_global_summary.csv"
)

write_csv(global_summary, global_summary_file)

cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("P4 completed.\n")
cat("Global summary:\n")
print(global_summary)
cat("\nSaved:\n")
cat(global_summary_file, "\n")
cat(paste(rep("=", 80), collapse = ""), "\n", sep = "")