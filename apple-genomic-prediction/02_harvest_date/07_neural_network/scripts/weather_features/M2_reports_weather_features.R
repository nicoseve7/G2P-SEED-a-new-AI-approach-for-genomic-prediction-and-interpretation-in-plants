# Questo confronta in modo semplice i file prodotti.

# ==================================================
# M2_reports_weather_features.R
# Report di confronto per le nuove feature meteo
# ==================================================

rm(list = ls())

cat("=== M2: REPORT WEATHER FEATURES ===\n\n")

library(readr)
library(dplyr)

outdir <- "02_harvest_date/07_neural_network/output/weather_features"
report_dir <- file.path(outdir, "reports")

dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

v2_file <- file.path(
  outdir,
  "weather_period_features_v2.csv"
)

v3_file <- file.path(
  outdir,
  "weather_period_features_v3_splitP2.csv"
)

report_file <- file.path(
  report_dir,
  "weather_features_comparison_report.txt"
)

cat("Caricamento file derivati...\n")
v2 <- read_csv(v2_file, show_col_types = FALSE)
v3 <- read_csv(v3_file, show_col_types = FALSE)

cat("Caricamento completato.\n\n")

sink(report_file)

cat("=== WEATHER FEATURES COMPARISON REPORT ===\n\n")

cat("V2 (P1 + P2)\n")
cat("Numero righe:", nrow(v2), "\n")
cat("Numero colonne:", ncol(v2), "\n")
cat("Prime colonne:\n")
print(colnames(v2))
cat("\n")
cat("Missing per colonna:\n")
print(colSums(is.na(v2)))
cat("\n\n")

cat("V3 (P1 + P2a + P2b)\n")
cat("Numero righe:", nrow(v3), "\n")
cat("Numero colonne:", ncol(v3), "\n")
cat("Prime colonne:\n")
print(colnames(v3))
cat("\n")
cat("Missing per colonna:\n")
print(colSums(is.na(v3)))
cat("\n\n")

common_env <- intersect(v2$Envir, v3$Envir)

cat("Environment in comune tra V2 e V3:", length(common_env), "\n")
cat("Environment presenti in V2 ma non in V3:\n")
print(setdiff(v2$Envir, v3$Envir))
cat("\n")
cat("Environment presenti in V3 ma non in V2:\n")
print(setdiff(v3$Envir, v2$Envir))
cat("\n")

sink()

cat("File salvato:\n")
cat("-", report_file, "\n")
