# ==================================================
# C2_build_master_alignment_table.R
# Costruzione tabella master Genotype x Envir
# ==================================================

rm(list = ls())

cat("=== STEP C2: BUILD MASTER ALIGNMENT TABLE ===\n\n")

library(readr)
library(dplyr)
library(snpStats)

# -------------------------------
# 1. Percorsi file
# -------------------------------
pheno_file <- paste0(
  "02_harvest_date/02_phenotype_preprocessing/output/",
  "Harvest_date_processed_final.csv"
)

W_file <- paste0(
  "02_harvest_date/03_environment_preprocessing/output/",
  "W_environment_paper_style.csv"
)

bed_file <- "data/raw/genotype/SNPs_final_2022.bed"
bim_file <- "data/raw/genotype/SNPs_final_2022.bim"
fam_file <- "data/raw/genotype/SNPs_final_2022.fam"

outdir <- "02_harvest_date/04_input_preparation/output"

# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento fenotipo finale...\n")
pheno <- read_csv(pheno_file, show_col_types = FALSE)

cat("Caricamento W paper-style...\n")
W <- read_csv(W_file, show_col_types = FALSE)

cat("Caricamento genomica PLINK...\n")
geno <- read.plink(bed_file, bim_file, fam_file)

cat("Caricamento completato.\n\n")

# -------------------------------
# 3. ID genomici disponibili
# -------------------------------
geno_ids <- unique(as.character(geno$fam$member))

# -------------------------------
# 4. Merge fenotipo + ambiente
# -------------------------------
cat("Merge fenotipo + ambiente...\n")

master_df <- pheno %>%
  left_join(W, by = "Envir")

cat("Merge completato.\n\n")

# -------------------------------
# 5. Flag disponibilità genomica
# -------------------------------
master_df$geno_available <- as.character(master_df$Genotype) %in% geno_ids

# -------------------------------
# 6. Controlli base
# -------------------------------
cat("Numero righe master:", nrow(master_df), "\n")
cat("Numero genotipi unici:", n_distinct(master_df$Genotype), "\n")
cat("Numero environment unici:", n_distinct(master_df$Envir), "\n\n")

cat("Numero righe con genomica disponibile:", sum(master_df$geno_available), "\n")
cat("Numero righe senza genomica disponibile:", sum(!master_df$geno_available), "\n\n")

# -------------------------------
# 7. Controlli missing
# -------------------------------
missing_summary <- colSums(is.na(master_df))

cat("Missing per colonna:\n")
print(missing_summary)
cat("\n")

# -------------------------------
# 8. Controlli di coerenza
# -------------------------------
missing_W_rows <- master_df %>%
  filter(if_any(-c(Genotype, Envir, Harvest_date, geno_available), is.na))

cat("Numero righe con almeno un missing nelle covariate ambientali:", nrow(missing_W_rows), "\n\n")

# -------------------------------
# 9. Ordinamento finale
# -------------------------------
master_df <- master_df %>%
  arrange(Envir, Genotype)

# -------------------------------
# 10. Salvataggio output
# -------------------------------
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

write_csv(
  master_df,
  file.path(outdir, "master_alignment_table.csv")
)

sink(
  file.path(outdir, "master_alignment_table_report.txt")
)

cat("=== STEP C2: MASTER ALIGNMENT TABLE REPORT ===\n\n")
cat("Numero righe master:", nrow(master_df), "\n")
cat("Numero genotipi unici:", n_distinct(master_df$Genotype), "\n")
cat("Numero environment unici:", n_distinct(master_df$Envir), "\n\n")

cat("Numero righe con genomica disponibile:", sum(master_df$geno_available), "\n")
cat("Numero righe senza genomica disponibile:", sum(!master_df$geno_available), "\n\n")

cat("Missing per colonna:\n")
print(missing_summary)
cat("\n\n")

cat("Numero righe con almeno un missing nelle covariate ambientali:", nrow(missing_W_rows), "\n\n")

cat("Prime righe master_df:\n")
print(head(master_df))
cat("\n")

sink()

# -------------------------------
# 11. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- ", file.path(outdir, "master_alignment_table.csv"), "\n")
cat("- ", file.path(outdir, "master_alignment_table_report.txt"), "\n")
