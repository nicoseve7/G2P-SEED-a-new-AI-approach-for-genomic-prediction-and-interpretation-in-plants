# ==================================================
# A8_finalize_harvestdate_pheno.R
# Finalizzazione del fenotipo processato di Harvest_date
# ==================================================

rm(list = ls())

cat("=== STEP A8: FINALIZE HARVEST_DATE PHENO ===\n\n")

library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
input_file <- "Output/harvestdate_processed_no_outliers.csv"

# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento fenotipo processato...\n")
pheno_final <- read_csv(input_file, show_col_types = FALSE)
cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Sistemazione colonne
# -------------------------------
pheno_final <- pheno_final %>%
  rename(Harvest_date = Value) %>%
  select(Genotype, Envir, Harvest_date)

# -------------------------------
# 4. Controlli finali
# -------------------------------
n_rows <- nrow(pheno_final)
n_geno <- n_distinct(pheno_final$Genotype)
n_env  <- n_distinct(pheno_final$Envir)

cat("Numero righe finali:", n_rows, "\n")
cat("Numero genotipi unici:", n_geno, "\n")
cat("Numero environment unici:", n_env, "\n\n")

# -------------------------------
# 5. Salvataggio output finale
# -------------------------------
dir.create("Output", showWarnings = FALSE)

write_csv(pheno_final, "Output/Harvest_date_processed_final.csv")

sink("Output/Harvest_date_processed_final_report.txt")

cat("=== STEP A8: FINAL HARVEST_DATE PHENO REPORT ===\n\n")
cat("Numero righe finali:", n_rows, "\n")
cat("Numero genotipi unici:", n_geno, "\n")
cat("Numero environment unici:", n_env, "\n\n")

cat("Prime righe:\n")
print(head(pheno_final))
cat("\n")

sink()

# -------------------------------
# 6. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- Output/Harvest_date_processed_final.csv\n")
cat("- Output/Harvest_date_processed_final_report.txt\n")