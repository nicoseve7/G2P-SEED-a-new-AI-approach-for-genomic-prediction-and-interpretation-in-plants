# ==================================================
# C5_build_SNP_matrix_for_modeling.R
# Costruzione matrice SNP per il modeling
# ==================================================

rm(list = ls())

cat("=== STEP C5: BUILD SNP MATRIX FOR MODELING ===\n\n")

library(snpStats)
library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
bed_file <- "data/raw/genotype/SNPs_final_2022.bed"
bim_file <- "data/raw/genotype/SNPs_final_2022.bim"
fam_file <- "data/raw/genotype/SNPs_final_2022.fam"

# -------------------------------
# 2. Caricamento genomica
# -------------------------------
cat("Caricamento genomica PLINK...\n")
geno <- read.plink(bed_file, bim_file, fam_file)
cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Estrazione matrice raw
# -------------------------------
cat("Estrazione matrice genotipica raw...\n")

X_raw <- geno$genotypes@.Data
class(X_raw) <- "numeric"

geno_ids <- as.character(geno$fam$member)
snp_ids <- as.character(geno$map$snp.name)

rownames(X_raw) <- geno_ids
colnames(X_raw) <- snp_ids

cat("Dimensioni X_raw:", nrow(X_raw), "x", ncol(X_raw), "\n\n")

# -------------------------------
# 4. Ricodifica per modeling
# -------------------------------
cat("Ricodifica SNP 1/2/3 -> 0/1/2 ...\n")
X <- X_raw - 1

cat("Range valori dopo ricodifica:\n")
print(range(X, na.rm = TRUE))
cat("\n")

# -------------------------------
# 5. Controlli missing
# -------------------------------
cat("Calcolo missing...\n")

missing_per_snp <- colSums(is.na(X))
missing_per_genotype <- rowSums(is.na(X))

cat("SNP con almeno un missing:", sum(missing_per_snp > 0), "\n")
cat("Genotipi con almeno un missing:", sum(missing_per_genotype > 0), "\n\n")

# -------------------------------
# 6. Rimozione SNP senza variabilità
# -------------------------------
cat("Rimozione SNP con varianza zero...\n")

snp_var <- apply(X, 2, var, na.rm = TRUE)
keep_snps <- which(!is.na(snp_var) & snp_var > 0)

X_var <- X[, keep_snps, drop = FALSE]
map_var <- geno$map[keep_snps, , drop = FALSE]

cat("Numero SNP iniziali:", ncol(X), "\n")
cat("Numero SNP con varianza > 0:", ncol(X_var), "\n")
cat("Numero SNP rimossi:", ncol(X) - ncol(X_var), "\n\n")

# -------------------------------
# 7. Tabella info SNP
# -------------------------------
snp_info <- map_var %>%
  mutate(snp.name = as.character(snp.name)) %>%
  select(snp.name, chromosome, position, allele.1, allele.2)

# -------------------------------
# 8. Anteprima output
# -------------------------------
cat("Prime righe matrice SNP (prime 10 colonne):\n")
print(X_var[1:min(5, nrow(X_var)), 1:min(10, ncol(X_var)), drop = FALSE])
cat("\n")

cat("Prime righe snp_info:\n")
print(head(snp_info))
cat("\n")

# -------------------------------
# 9. Salvataggio output
# -------------------------------
outdir <- "02_harvest_date/01_common_genomic_preprocessing/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# salviamo in formato RData per non creare csv giganteschi inutili
save(
  X_var,
  file = file.path(outdir, "SNP_matrix_modeling_var_gt0.RData")
)

write_csv(
  snp_info,
  file.path(outdir, "SNP_info_modeling_var_gt0.csv")
)

sink(
  file.path(outdir, "SNP_matrix_modeling_report.txt")
)

cat("=== STEP C5: SNP MATRIX FOR MODELING REPORT ===\n\n")
cat("Numero genotipi:", nrow(X), "\n")
cat("Numero SNP iniziali:", ncol(X), "\n")
cat("Numero SNP con varianza > 0:", ncol(X_var), "\n")
cat("Numero SNP rimossi:", ncol(X) - ncol(X_var), "\n\n")

cat("SNP con almeno un missing:", sum(missing_per_snp > 0), "\n")
cat("Genotipi con almeno un missing:", sum(missing_per_genotype > 0), "\n\n")

cat("Prime righe matrice SNP (prime 10 colonne):\n")
print(X_var[1:min(5, nrow(X_var)), 1:min(10, ncol(X_var)), drop = FALSE])
cat("\n\n")

cat("Prime righe snp_info:\n")
print(head(snp_info))
cat("\n")

sink()

# -------------------------------
# 10. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- ", file.path(outdir, "SNP_matrix_modeling_var_gt0.RData"), "\n")
cat("- ", file.path(outdir, "SNP_info_modeling_var_gt0.csv"), "\n")
cat("- ", file.path(outdir, "SNP_matrix_modeling_report.txt"), "\n")
