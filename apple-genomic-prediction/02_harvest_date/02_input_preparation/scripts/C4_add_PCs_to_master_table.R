# ==================================================
# C4_add_PCs_to_master_table.R
# Aggiunta delle PC genomiche alla tabella master
# ==================================================

rm(list = ls())

cat("=== STEP C4: ADD PCs TO MASTER TABLE ===\n\n")

library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
master_file <- "02_harvest_date/02_input_preparation/output/master_alignment_table.csv"
pcs_file <- "01_common_genomic_preprocessing/output/genomic_PCs_20_paper_style.csv"

# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento master table...\n")
master_df <- read_csv(master_file, show_col_types = FALSE)

cat("Caricamento genomic PCs...\n")
pcs_df <- read_csv(pcs_file, show_col_types = FALSE)

cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Controlli base
# -------------------------------
cat("Numero righe master:", nrow(master_df), "\n")
cat("Numero righe pcs_df:", nrow(pcs_df), "\n")
cat("Numero genotipi unici master:", n_distinct(master_df$Genotype), "\n")
cat("Numero genotipi unici pcs_df:", n_distinct(pcs_df$Genotype), "\n\n")

# -------------------------------
# 4. Join master + PCs
# -------------------------------
cat("Merge master + genomic PCs...\n")

master_pcs_df <- master_df %>%
  left_join(pcs_df, by = "Genotype") %>%
  arrange(Envir, Genotype)

cat("Merge completato.\n\n")

# -------------------------------
# 5. Controlli output
# -------------------------------
cat("Numero righe finali:", nrow(master_pcs_df), "\n")
cat("Numero colonne finali:", ncol(master_pcs_df), "\n\n")

missing_summary <- colSums(is.na(master_pcs_df))

cat("Missing per colonna:\n")
print(missing_summary)
cat("\n")

pc_missing_rows <- master_pcs_df %>%
  filter(if_any(starts_with("PC"), is.na))

cat("Numero righe con almeno una PC mancante:", nrow(pc_missing_rows), "\n\n")

cat("Prime righe master + PCs:\n")
print(head(master_pcs_df))
cat("\n")

# -------------------------------
# 6. Salvataggio output
# -------------------------------
outdir <- "02_harvest_date/02_input_preparation/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

write_csv(
  master_pcs_df,
  file.path(outdir, "master_alignment_table_with_PCs.csv")
)

sink(
  file.path(outdir, "master_alignment_table_with_PCs_report.txt")
)

cat("=== STEP C4: MASTER TABLE WITH PCs REPORT ===\n\n")
cat("Numero righe finali:", nrow(master_pcs_df), "\n")
cat("Numero colonne finali:", ncol(master_pcs_df), "\n\n")

cat("Missing per colonna:\n")
print(missing_summary)
cat("\n\n")

cat("Numero righe con almeno una PC mancante:", nrow(pc_missing_rows), "\n\n")

cat("Prime righe master + PCs:\n")
print(head(master_pcs_df))
cat("\n")

sink()

# -------------------------------
# 7. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- Output/master_alignment_table_with_PCs.csv\n")
cat("- Output/master_alignment_table_with_PCs_report.txt\n")
