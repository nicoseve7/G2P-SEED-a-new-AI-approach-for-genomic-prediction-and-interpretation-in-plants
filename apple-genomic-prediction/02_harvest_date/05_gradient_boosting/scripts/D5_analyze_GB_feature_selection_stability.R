# ==================================================
# D5_analyze_GB_feature_selection_stability.R
# Analisi di stabilità e overlap degli SNP selezionati
# ==================================================

rm(list = ls())

cat("=== STEP D5: ANALYZE GB FEATURE SELECTION STABILITY ===\n\n")

library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
fs_file <- "Output/Intermediate/GB_feature_selection/feature_selection_results_harvest_date.csv"
summary_file <- "Output/Intermediate/GB_feature_selection/feature_selection_summary_harvest_date.csv"

# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento risultati feature selection...\n")
fs <- read_csv(fs_file, show_col_types = FALSE)

cat("Caricamento summary split...\n")
fs_summary <- read_csv(summary_file, show_col_types = FALSE)

cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Controlli base
# -------------------------------
cat("Numero righe feature selection:", nrow(fs), "\n")
cat("Numero split unici:", n_distinct(fs$Split), "\n")
cat("Numero SNP unici selezionati:", n_distinct(fs$SNP), "\n\n")

# -------------------------------
# 4. Stabilità SNP tra split
# -------------------------------
cat("Calcolo stabilità SNP...\n")

snp_stability <- fs %>%
  group_by(SNP) %>%
  summarise(
    n_splits_selected = n_distinct(Split),
    mean_importance = mean(importance, na.rm = TRUE),
    median_importance = median(importance, na.rm = TRUE),
    max_importance = max(importance, na.rm = TRUE),
    min_importance = min(importance, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n_splits_selected), desc(mean_importance))

# -------------------------------
# 5. Riassunto globale di overlap
# -------------------------------
thresholds <- c(1, 2, 5, 10, 15, 20, 25)

overlap_summary <- data.frame(
  min_splits = thresholds,
  n_snps = sapply(thresholds, function(k) {
    sum(snp_stability$n_splits_selected >= k)
  })
)

# -------------------------------
# 6. Top SNP più stabili
# -------------------------------
top_stable_snps <- snp_stability %>%
  arrange(desc(n_splits_selected), desc(mean_importance)) %>%
  slice_head(n = 50)

# -------------------------------
# 7. Tabella per split -> numero SNP
# -------------------------------
split_counts <- fs %>%
  group_by(Split) %>%
  summarise(
    n_selected = n_distinct(SNP),
    .groups = "drop"
  ) %>%
  arrange(Split)

# -------------------------------
# 8. Matrice presenza/assenza SNP x split
# -------------------------------
fs_presence <- fs %>%
  distinct(SNP, Split) %>%
  mutate(selected = 1)

# -------------------------------
# 9. Output console
# -------------------------------
cat("Prime righe snp_stability:\n")
print(head(snp_stability, 10))
cat("\n")

cat("Overlap summary:\n")
print(overlap_summary)
cat("\n")

cat("Top 10 SNP più stabili:\n")
print(head(top_stable_snps, 10))
cat("\n")

# -------------------------------
# 10. Salvataggio output
# -------------------------------
out_dir <- "Output/Intermediate/GB_feature_selection"

write_csv(snp_stability, file.path(out_dir, "feature_selection_snp_stability_harvest_date.csv"))
write_csv(overlap_summary, file.path(out_dir, "feature_selection_overlap_summary_harvest_date.csv"))
write_csv(top_stable_snps, file.path(out_dir, "feature_selection_top50_stable_snps_harvest_date.csv"))
write_csv(split_counts, file.path(out_dir, "feature_selection_split_counts_harvest_date.csv"))

sink(file.path(out_dir, "feature_selection_stability_report_harvest_date.txt"))

cat("=== STEP D5: FEATURE SELECTION STABILITY REPORT ===\n\n")

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

# -------------------------------
# 11. Fine
# -------------------------------
cat("File salvati:\n")
cat("- Output/Intermediate/GB_feature_selection/feature_selection_snp_stability_harvest_date.csv\n")
cat("- Output/Intermediate/GB_feature_selection/feature_selection_overlap_summary_harvest_date.csv\n")
cat("- Output/Intermediate/GB_feature_selection/feature_selection_top50_stable_snps_harvest_date.csv\n")
cat("- Output/Intermediate/GB_feature_selection/feature_selection_split_counts_harvest_date.csv\n")
cat("- Output/Intermediate/GB_feature_selection/feature_selection_stability_report_harvest_date.txt\n")