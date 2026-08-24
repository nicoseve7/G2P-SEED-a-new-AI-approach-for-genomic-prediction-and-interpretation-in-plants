# ==================================================
# C6_final_check_model_inputs.R
# Verifica finale dei dataset pronti per i modelli
# ==================================================

rm(list = ls())

cat("=== STEP C6: FINAL CHECK MODEL INPUTS ===\n\n")

library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
master_pcs_file <- "02_harvest_date/02_input_preparation/output/master_alignment_table_with_PCs.csv"
pcs_file <- "01_common_genomic_preprocessing/output/genomic_PCs_20_paper_style.csv"
pve_file <- "01_common_genomic_preprocessing/output/genomic_PCs_variance_explained_paper_style.csv"
snp_info_file <- "01_common_genomic_preprocessing/output/SNP_info_modeling_var_gt0.csv"
snp_matrix_file <- "01_common_genomic_preprocessing/output/SNP_matrix_modeling_var_gt0.RData"

# -------------------------------
# 2. Controllo esistenza file
# -------------------------------
files_to_check <- c(
  master_pcs_file,
  pcs_file,
  pve_file,
  snp_info_file,
  snp_matrix_file
)

cat("Controllo file presenti:\n")
for (f in files_to_check) {
  cat(f, "->", file.exists(f), "\n")
}
cat("\n")

# -------------------------------
# 3. Caricamento file tabellari
# -------------------------------
cat("Caricamento master table con PCs...\n")
master_pcs <- read_csv(master_pcs_file, show_col_types = FALSE)

cat("Caricamento genomic PCs...\n")
pcs_df <- read_csv(pcs_file, show_col_types = FALSE)

cat("Caricamento PVE...\n")
pve_df <- read_csv(pve_file, show_col_types = FALSE)

cat("Caricamento SNP info...\n")
snp_info <- read_csv(snp_info_file, show_col_types = FALSE)

cat("Caricamento matrice SNP...\n")
load(snp_matrix_file)   # carica X_var

cat("Caricamento completato.\n\n")

# -------------------------------
# 4. Dimensioni generali
# -------------------------------
cat("=== DIMENSIONI GENERALI ===\n")
cat("master_alignment_table_with_PCs:", nrow(master_pcs), "x", ncol(master_pcs), "\n")
cat("genomic_PCs_20_paper_style:", nrow(pcs_df), "x", ncol(pcs_df), "\n")
cat("genomic_PCs_variance_explained:", nrow(pve_df), "x", ncol(pve_df), "\n")
cat("SNP_info_modeling_var_gt0:", nrow(snp_info), "x", ncol(snp_info), "\n")
cat("SNP matrix X_var:", nrow(X_var), "x", ncol(X_var), "\n\n")

# -------------------------------
# 5. Coerenza ID genomici
# -------------------------------
master_genotypes <- sort(unique(as.character(master_pcs$Genotype)))
pc_genotypes <- sort(unique(as.character(pcs_df$Genotype)))
snp_genotypes <- sort(rownames(X_var))

cat("=== COERENZA ID GENOTIPI ===\n")
cat("Genotipi unici in master_pcs:", length(master_genotypes), "\n")
cat("Genotipi unici in pcs_df:", length(pc_genotypes), "\n")
cat("Genotipi unici in X_var:", length(snp_genotypes), "\n\n")

cat("Match master vs PCs:", setequal(master_genotypes, pc_genotypes), "\n")
cat("Match master vs SNP matrix:", setequal(master_genotypes, snp_genotypes), "\n\n")

missing_in_pcs <- setdiff(master_genotypes, pc_genotypes)
missing_in_snp <- setdiff(master_genotypes, snp_genotypes)

cat("Genotipi del master mancanti nelle PCs:", length(missing_in_pcs), "\n")
if (length(missing_in_pcs) > 0) print(missing_in_pcs)
cat("\n")

cat("Genotipi del master mancanti nella SNP matrix:", length(missing_in_snp), "\n")
if (length(missing_in_snp) > 0) print(missing_in_snp)
cat("\n")

# -------------------------------
# 6. Controlli missing
# -------------------------------
cat("=== MISSING VALUES ===\n")
cat("Missing in master_pcs:", sum(is.na(master_pcs)), "\n")
cat("Missing in pcs_df:", sum(is.na(pcs_df)), "\n")
cat("Missing in pve_df:", sum(is.na(pve_df)), "\n")
cat("Missing in snp_info:", sum(is.na(snp_info)), "\n")
cat("Missing in X_var:", sum(is.na(X_var)), "\n\n")

# -------------------------------
# 7. Riassunto colonne PC
# -------------------------------
pc_cols_master <- grep("^PC", colnames(master_pcs), value = TRUE)
pc_cols_pcs <- grep("^PC", colnames(pcs_df), value = TRUE)

cat("=== RIASSUNTO PC ===\n")
cat("Numero PC in master_pcs:", length(pc_cols_master), "\n")
cat("Numero PC in pcs_df:", length(pc_cols_pcs), "\n")
cat("Prime 5 PC:", paste(head(pc_cols_master, 5), collapse = ", "), "\n\n")

# -------------------------------
# 8. Riassunto varianza spiegata
# -------------------------------
cat("=== VARIANZA SPIEGATA PCA ===\n")
cat("Prime 10 componenti:\n")
print(head(pve_df, 10))
cat("\n")

cat("Varianza cumulativa prime 5 PC:",
    sum(pve_df$ProportionVarianceExplained[1:min(5, nrow(pve_df))]), "\n")
cat("Varianza cumulativa prime 10 PC:",
    sum(pve_df$ProportionVarianceExplained[1:min(10, nrow(pve_df))]), "\n")
cat("Varianza cumulativa prime 20 PC:",
    sum(pve_df$ProportionVarianceExplained[1:min(20, nrow(pve_df))]), "\n\n")

# -------------------------------
# 9. Riassunto finale dataset modeling
# -------------------------------
cat("=== DATASET FINALE MODELING ===\n")
cat("Righe (Genotype x Envir):", nrow(master_pcs), "\n")
cat("Genotipi unici:", n_distinct(master_pcs$Genotype), "\n")
cat("Environment unici:", n_distinct(master_pcs$Envir), "\n")
cat("Colonne totali:", ncol(master_pcs), "\n\n")

# -------------------------------
# 10. Salvataggio report
# -------------------------------
outdir <- "02_harvest_date/02_input_preparation/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

sink(
  file.path(outdir, "final_check_model_inputs_report.txt")
)

cat("=== STEP C6: FINAL CHECK MODEL INPUTS REPORT ===\n\n")

cat("DIMENSIONI GENERALI\n")
cat("master_alignment_table_with_PCs:", nrow(master_pcs), "x", ncol(master_pcs), "\n")
cat("genomic_PCs_20_paper_style:", nrow(pcs_df), "x", ncol(pcs_df), "\n")
cat("genomic_PCs_variance_explained:", nrow(pve_df), "x", ncol(pve_df), "\n")
cat("SNP_info_modeling_var_gt0:", nrow(snp_info), "x", ncol(snp_info), "\n")
cat("SNP matrix X_var:", nrow(X_var), "x", ncol(X_var), "\n\n")

cat("COERENZA ID GENOTIPI\n")
cat("Genotipi unici in master_pcs:", length(master_genotypes), "\n")
cat("Genotipi unici in pcs_df:", length(pc_genotypes), "\n")
cat("Genotipi unici in X_var:", length(snp_genotypes), "\n")
cat("Match master vs PCs:", setequal(master_genotypes, pc_genotypes), "\n")
cat("Match master vs SNP matrix:", setequal(master_genotypes, snp_genotypes), "\n\n")

cat("Genotipi del master mancanti nelle PCs:", length(missing_in_pcs), "\n")
if (length(missing_in_pcs) > 0) print(missing_in_pcs)
cat("\n")

cat("Genotipi del master mancanti nella SNP matrix:", length(missing_in_snp), "\n")
if (length(missing_in_snp) > 0) print(missing_in_snp)
cat("\n")

cat("MISSING VALUES\n")
cat("Missing in master_pcs:", sum(is.na(master_pcs)), "\n")
cat("Missing in pcs_df:", sum(is.na(pcs_df)), "\n")
cat("Missing in pve_df:", sum(is.na(pve_df)), "\n")
cat("Missing in snp_info:", sum(is.na(snp_info)), "\n")
cat("Missing in X_var:", sum(is.na(X_var)), "\n\n")

cat("RIASSUNTO PC\n")
cat("Numero PC in master_pcs:", length(pc_cols_master), "\n")
cat("Numero PC in pcs_df:", length(pc_cols_pcs), "\n")
cat("Prime 5 PC:", paste(head(pc_cols_master, 5), collapse = ", "), "\n\n")

cat("VARIANZA SPIEGATA PCA\n")
print(head(pve_df, 10))
cat("\n")
cat("Varianza cumulativa prime 5 PC:",
    sum(pve_df$ProportionVarianceExplained[1:min(5, nrow(pve_df))]), "\n")
cat("Varianza cumulativa prime 10 PC:",
    sum(pve_df$ProportionVarianceExplained[1:min(10, nrow(pve_df))]), "\n")
cat("Varianza cumulativa prime 20 PC:",
    sum(pve_df$ProportionVarianceExplained[1:min(20, nrow(pve_df))]), "\n\n")

cat("DATASET FINALE MODELING\n")
cat("Righe (Genotype x Envir):", nrow(master_pcs), "\n")
cat("Genotipi unici:", n_distinct(master_pcs$Genotype), "\n")
cat("Environment unici:", n_distinct(master_pcs$Envir), "\n")
cat("Colonne totali:", ncol(master_pcs), "\n")

sink()

# -------------------------------
# 11. Stampa finale
# -------------------------------
cat("File salvato:\n")
cat("- Output/final_check_model_inputs_report.txt\n")
