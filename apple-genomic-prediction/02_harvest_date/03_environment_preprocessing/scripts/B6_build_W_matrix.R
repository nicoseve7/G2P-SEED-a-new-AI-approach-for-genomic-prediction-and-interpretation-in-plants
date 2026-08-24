# ==================================================
# B6_build_W_matrix.R
# Fusione finale weather + soil e costruzione di W
# ==================================================

rm(list = ls())

cat("=== STEP B6: BUILD W MATRIX ===\n\n")

library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
weather_file <- "Output/weather_period_aggregation_wide.csv"
soil_file <- "Output/soil_by_envir.csv"

# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento weather aggregated wide...\n")
weather_wide <- read_csv(weather_file, show_col_types = FALSE)

cat("Caricamento soil_by_envir...\n")
soil_by_envir <- read_csv(soil_file, show_col_types = FALSE)

cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Controlli base
# -------------------------------
cat("Numero environment weather:", nrow(weather_wide), "\n")
cat("Numero environment soil:", nrow(soil_by_envir), "\n\n")

cat("Prime righe weather:\n")
print(head(weather_wide))
cat("\n")

cat("Prime righe soil:\n")
print(head(soil_by_envir))
cat("\n")

# -------------------------------
# 4. Fusione finale
# -------------------------------
cat("Merge weather + soil by Envir...\n")

W_df <- weather_wide %>%
  left_join(soil_by_envir, by = "Envir") %>%
  arrange(Envir)

cat("Merge completato.\n\n")

# -------------------------------
# 5. Controlli output
# -------------------------------
cat("Numero environment finali in W:", nrow(W_df), "\n")
cat("Numero colonne finali in W:", ncol(W_df), "\n\n")

cat("Prime righe W:\n")
print(head(W_df))
cat("\n")

cat("Missing in W:\n")
print(colSums(is.na(W_df)))
cat("\n")

# -------------------------------
# 6. Matrice numerica W
# -------------------------------
# W_matrix <- W_df %>%
#   column_to_rownames("Envir") %>%
#   as.matrix()
W_matrix <- as.data.frame(W_df)
rownames(W_matrix) <- W_matrix$Envir
W_matrix$Envir <- NULL
W_matrix <- as.matrix(W_matrix)
# -------------------------------
# 7. Salvataggio output
# -------------------------------
dir.create("Output", showWarnings = FALSE)

write_csv(W_df, "Output/W_environment_full.csv")

# salvataggio anche in formato RData, più vicino al paper
save(W_matrix, file = "Output/W_environment_full.RData")

sink("Output/W_environment_report.txt")

cat("=== STEP B6: W ENVIRONMENT REPORT ===\n\n")
cat("Numero environment finali:", nrow(W_df), "\n")
cat("Numero colonne finali:", ncol(W_df), "\n\n")

cat("Prime righe W:\n")
print(head(W_df))
cat("\n\n")

cat("Missing in W:\n")
print(colSums(is.na(W_df)))
cat("\n")

sink()

# -------------------------------
# 8. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- Output/W_environment_full.csv\n")
cat("- Output/W_environment_full.RData\n")
cat("- Output/W_environment_report.txt\n")