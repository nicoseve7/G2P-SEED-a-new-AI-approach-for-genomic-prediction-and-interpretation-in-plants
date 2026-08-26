# ===============================
# A1_data_audit.R
# Initial data audit
# ===============================

rm(list = ls()) 

cat("=== STEP A1: DATA AUDIT ===\n\n") 

# -------------------------------
# 1. Pacchetti
# -------------------------------
required_packages <- c("readxl", "snpStats")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(readxl)
library(snpStats)

# -------------------------------
# 2. File paths
# -------------------------------
pheno_file <- "data/raw/phenotype/Pheno_raw.xlsx"
weather_file <- "data/raw/environment/Weather_raw.xlsx"
soil_file <- "data/raw/environment/Soil_raw.xlsx"

bed_file <- "data/raw/genotype/SNPs_final_2022.bed"
bim_file <- "data/raw/genotype/SNPs_final_2022.bim"
fam_file <- "data/raw/genotype/SNPs_final_2022.fam"

# -------------------------------
# 3. File existence check: For each file, it checks whether it actually exists
# -------------------------------
files_to_check <- c(pheno_file, weather_file, soil_file, bed_file, bim_file, fam_file) # This list returns TRUE if the file exists; otherwise, it returns FALSE

cat("Controllo file presenti:\n")
for (f in files_to_check) {
  cat(f, "->", file.exists(f), "\n")
}
cat("\n")

# -------------------------------
# 4. Loading data
# -------------------------------
cat("Caricamento file fenotipico...\n")
pheno <- read_xlsx(pheno_file)
pheno <- as.data.frame(pheno)

cat("Caricamento file meteo...\n")
weather <- read_xlsx(weather_file)
weather <- as.data.frame(weather)

cat("Caricamento file suolo...\n")
soil <- read_xlsx(soil_file)
soil <- as.data.frame(soil)

cat("Caricamento file genomici PLINK...\n")
geno <- read.plink(bed_file, bim_file, fam_file) # geno contains geno$fam = information about individuals, geno$map = information about SNPs, geno$genotypes = genotype matrix

cat("Caricamento completato.\n\n")

# -------------------------------
# 5. Dataset sizes
# -------------------------------
cat("=== DIMENSIONI DATASET ===\n")
cat("Pheno_raw:", nrow(pheno), "righe x", ncol(pheno), "colonne\n")
cat("Weather_raw:", nrow(weather), "righe x", ncol(weather), "colonne\n")
cat("Soil_raw:", nrow(soil), "righe x", ncol(soil), "colonne\n")
cat("Genotype .fam:", nrow(geno$fam), "individui\n")
cat("Genotype .map:", nrow(geno$map), "SNP\n\n")

# -------------------------------
# 6. Column Names
# -------------------------------
cat("=== COLONNE PHENO ===\n")
print(colnames(pheno))
cat("\n")

cat("=== COLONNE WEATHER ===\n")
print(colnames(weather))
cat("\n")

cat("=== COLONNE SOIL ===\n")
print(colnames(soil))
cat("\n")

cat("=== COLONNE FAM ===\n")
print(colnames(geno$fam)) # to understand how IDs are named in the genomic file
cat("\n")

cat("=== PRIME RIGHE FAM ===\n")
print(head(geno$fam))
cat("\n")

# -------------------------------
# 7. Basic Phenotype Assessments
# -------------------------------
cat("=== CONTROLLI FENOTIPICI BASE ===\n")
# Lines 104 and 105 check whether the “Genotype” column exists: if it does, they retrieve all the values in the column, remove duplicates using `unique()`, and count how many different genotypes there are
if ("Genotype" %in% colnames(pheno)) {
  cat("Genotipi unici nel fenotipo:", length(unique(pheno$Genotype)), "\n")
} else {
  cat("ATTENZIONE: colonna 'Genotype' non trovata nel fenotipo.\n")
}
# The following lines check whether the “Envir” column exists; if so, they count how many unique environments there are and print them in sorted order
if ("Envir" %in% colnames(pheno)) {
  cat("Environment unici nel fenotipo:", length(unique(pheno$Envir)), "\n")
  cat("Valori Envir:\n")
  print(sort(unique(pheno$Envir)))
} else {
  cat("ATTENZIONE: colonna 'Envir' non trovata nel fenotipo.\n")
}
cat("\n")

# -------------------------------
# 8. Phenotype-Genotype Match
# -------------------------------
cat("=== MATCH FENOTIPO vs GENOTIPO ===\n")

if ("Genotype" %in% colnames(pheno)) {
  
  pheno_ids <- unique(as.character(pheno$Genotype)) # Prende tutti i genotipi unici dal file fenotipico e li trasforma in testo per confrontarli più facilmente con gli ID del .fam
  
  fam_cols <- colnames(geno$fam) # Prende i nomi di tutte le colonne del .fam
  
  for (fc in fam_cols) {
    fam_ids <- unique(as.character(geno$fam[[fc]]))
    n_match <- sum(pheno_ids %in% fam_ids)
    cat("Match con colonna fam$", fc, ":", n_match, "su", length(pheno_ids), "\n")
  } # For each column in the .fam file: it takes all the unique values in that column, checks how many genotypes of that phenotype also appear there, and prints the number of matches
  
} else {
  cat("Impossibile fare il match: colonna 'Genotype' assente.\n")
}
cat("\n")

# -------------------------------
# 9. Missing values in the phenotype
# -------------------------------
cat("=== MISSING VALUES PHENO ===\n")
missing_counts <- colSums(is.na(pheno)) # Count how many missing values there are in each column of the phenotype. is.na(pheno) creates a table of TRUE/FALSE values. colSums(...) counts how many TRUE values there are in each column. This gives you the number of missing values per column.
print(missing_counts)
cat("\n")

# -------------------------------
# 10. Save report
# -------------------------------
outdir <- "02_harvest_date/01_data_audit/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

sink(file.path(outdir, "audit_report.txt"))

cat("=== DATA AUDIT REPORT ===\n\n")

cat("DIMENSIONI DATASET\n")
cat("Pheno_raw:", nrow(pheno), "x", ncol(pheno), "\n")
cat("Weather_raw:", nrow(weather), "x", ncol(weather), "\n")
cat("Soil_raw:", nrow(soil), "x", ncol(soil), "\n")
cat("Genotype .fam:", nrow(geno$fam), "\n")
cat("Genotype .map:", nrow(geno$map), "\n\n")

cat("COLONNE PHENO\n")
print(colnames(pheno))
cat("\n")

cat("COLONNE WEATHER\n")
print(colnames(weather))
cat("\n")

cat("COLONNE SOIL\n")
print(colnames(soil))
cat("\n")

cat("COLONNE FAM\n")
print(colnames(geno$fam))
cat("\n")

if ("Genotype" %in% colnames(pheno)) {
  cat("Genotipi unici nel fenotipo:", length(unique(pheno$Genotype)), "\n")
}

if ("Envir" %in% colnames(pheno)) {
  cat("Environment unici nel fenotipo:", length(unique(pheno$Envir)), "\n")
  print(sort(unique(pheno$Envir)))
  cat("\n")
}

cat("MISSING VALUES PHENO\n")
print(missing_counts)
cat("\n")

sink()

cat("Report salvato in Output/audit_report.txt\n")
