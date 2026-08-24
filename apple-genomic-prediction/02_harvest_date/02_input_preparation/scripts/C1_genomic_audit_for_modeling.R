# ==================================================
# C1_genomic_audit_for_modeling.R
# Audit della genomica per il modeling
# ==================================================

rm(list = ls())

cat("=== STEP C1: GENOMIC AUDIT FOR MODELING ===\n\n")

library(snpStats)
library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
bed_file <- "Input/SNPs_final_2022.bed"
bim_file <- "Input/SNPs_final_2022.bim"
fam_file <- "Input/SNPs_final_2022.fam"

pheno_file <- "Output/Harvest_date_processed_final.csv"

# -------------------------------
# 2. Controllo esistenza file
# -------------------------------
cat("Controllo file presenti:\n")
cat("BED  ->", file.exists(bed_file), "\n")
cat("BIM  ->", file.exists(bim_file), "\n")
cat("FAM  ->", file.exists(fam_file), "\n")
cat("PHENO FINAL ->", file.exists(pheno_file), "\n\n")

# -------------------------------
# 3. Caricamento dati
# -------------------------------
cat("Caricamento genomica PLINK...\n")
geno <- read.plink(bed_file, bim_file, fam_file)

cat("Caricamento fenotipo finale...\n")
pheno <- read_csv(pheno_file, show_col_types = FALSE)

cat("Caricamento completato.\n\n")

# -------------------------------
# 4. Dimensioni generali
# -------------------------------
cat("=== DIMENSIONI ===\n")
cat("Numero individui in geno$fam:", nrow(geno$fam), "\n")
cat("Numero SNP in geno$map:", nrow(geno$map), "\n")
cat("Numero righe fenotipo finale:", nrow(pheno), "\n")
cat("Numero genotipi unici fenotipo finale:", n_distinct(pheno$Genotype), "\n\n")

# -------------------------------
# 5. Colonne principali
# -------------------------------
cat("=== COLONNE FAM ===\n")
print(colnames(geno$fam))
cat("\n")

cat("=== PRIME RIGHE FAM ===\n")
print(head(geno$fam))
cat("\n")

cat("=== COLONNE MAP ===\n")
print(colnames(geno$map))
cat("\n")

cat("=== PRIME RIGHE MAP ===\n")
print(head(geno$map))
cat("\n")

cat("=== PRIME RIGHE PHENO FINALE ===\n")
print(head(pheno))
cat("\n")

# -------------------------------
# 6. Match ID genomica vs fenotipo finale
# -------------------------------
pheno_ids <- sort(unique(as.character(pheno$Genotype)))
geno_ids_member <- sort(unique(as.character(geno$fam$member)))
geno_ids_pedigree <- sort(unique(as.character(geno$fam$pedigree)))

match_member <- sum(pheno_ids %in% geno_ids_member)
match_pedigree <- sum(pheno_ids %in% geno_ids_pedigree)

cat("=== MATCH ID ===\n")
cat("Genotipi unici nel fenotipo finale:", length(pheno_ids), "\n")
cat("Match con fam$member:", match_member, "\n")
cat("Match con fam$pedigree:", match_pedigree, "\n\n")

missing_in_geno <- setdiff(pheno_ids, geno_ids_member)
extra_in_geno <- setdiff(geno_ids_member, pheno_ids)

cat("Genotipi del fenotipo finale NON presenti in genomica (member):", length(missing_in_geno), "\n")
if (length(missing_in_geno) > 0) print(missing_in_geno)
cat("\n")

cat("Genotipi della genomica NON presenti nel fenotipo finale:", length(extra_in_geno), "\n")
cat("\n")

# -------------------------------
# 7. Piccola ispezione della matrice genotipica
# -------------------------------
cat("=== ISPEZIONE MATRICE GENOTIPICA ===\n")
geno_matrix_raw <- geno$genotypes@.Data

cat("Dimensioni matrice raw:", dim(geno_matrix_raw)[1], "x", dim(geno_matrix_raw)[2], "\n")
cat("Classe matrice raw:", class(geno_matrix_raw), "\n\n")

# prova a convertire un piccolo blocco a numerico
sub_n_rows <- min(5, nrow(geno_matrix_raw))
sub_n_cols <- min(10, ncol(geno_matrix_raw))

geno_sub <- geno_matrix_raw[1:sub_n_rows, 1:sub_n_cols, drop = FALSE]
geno_sub_num <- geno_sub
class(geno_sub_num) <- "numeric"

cat("Piccolo blocco raw convertito a numeric:\n")
print(geno_sub_num)
cat("\n")

# -------------------------------
# 8. Statistiche semplici su fenotipo per genotype x envir
# -------------------------------
pheno_counts <- pheno %>%
  group_by(Genotype) %>%
  summarise(
    n_envir = n(),
    .groups = "drop"
  )

cat("=== COPERTURA FENOTIPO FINALE PER GENOTIPO ===\n")
cat("Min n_envir per genotipo:", min(pheno_counts$n_envir), "\n")
cat("Median n_envir per genotipo:", median(pheno_counts$n_envir), "\n")
cat("Max n_envir per genotipo:", max(pheno_counts$n_envir), "\n\n")

# -------------------------------
# 9. Salvataggio report
# -------------------------------
dir.create("Output", showWarnings = FALSE)

sink("Output/genomic_audit_for_modeling_report.txt")

cat("=== STEP C1: GENOMIC AUDIT FOR MODELING REPORT ===\n\n")

cat("DIMENSIONI\n")
cat("Numero individui in geno$fam:", nrow(geno$fam), "\n")
cat("Numero SNP in geno$map:", nrow(geno$map), "\n")
cat("Numero righe fenotipo finale:", nrow(pheno), "\n")
cat("Numero genotipi unici fenotipo finale:", n_distinct(pheno$Genotype), "\n\n")

cat("COLONNE FAM\n")
print(colnames(geno$fam))
cat("\n")

cat("PRIME RIGHE FAM\n")
print(head(geno$fam))
cat("\n")

cat("COLONNE MAP\n")
print(colnames(geno$map))
cat("\n")

cat("PRIME RIGHE MAP\n")
print(head(geno$map))
cat("\n")

cat("MATCH ID\n")
cat("Genotipi unici nel fenotipo finale:", length(pheno_ids), "\n")
cat("Match con fam$member:", match_member, "\n")
cat("Match con fam$pedigree:", match_pedigree, "\n\n")

cat("Genotipi del fenotipo finale NON presenti in genomica (member):", length(missing_in_geno), "\n")
if (length(missing_in_geno) > 0) print(missing_in_geno)
cat("\n")

cat("Genotipi della genomica NON presenti nel fenotipo finale:", length(extra_in_geno), "\n\n")

cat("ISPEZIONE MATRICE GENOTIPICA\n")
cat("Dimensioni matrice raw:", dim(geno_matrix_raw)[1], "x", dim(geno_matrix_raw)[2], "\n")
cat("Classe matrice raw:", class(geno_matrix_raw), "\n\n")
cat("Piccolo blocco raw convertito a numeric:\n")
print(geno_sub_num)
cat("\n")

cat("COPERTURA FENOTIPO FINALE PER GENOTIPO\n")
cat("Min n_envir per genotipo:", min(pheno_counts$n_envir), "\n")
cat("Median n_envir per genotipo:", median(pheno_counts$n_envir), "\n")
cat("Max n_envir per genotipo:", max(pheno_counts$n_envir), "\n\n")

sink()

# -------------------------------
# 10. Stampa finale
# -------------------------------
cat("File salvato:\n")
cat("- Output/genomic_audit_for_modeling_report.txt\n")