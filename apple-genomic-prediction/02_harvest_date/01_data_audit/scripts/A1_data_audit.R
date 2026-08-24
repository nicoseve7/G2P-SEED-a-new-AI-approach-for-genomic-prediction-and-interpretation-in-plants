# ===============================
# A1_data_audit.R
# Audit iniziale dei dati
# ===============================

rm(list = ls()) # questo comando pulisce l'ambiente di lavoro R, così altre variabili non interferiscono

cat("=== STEP A1: DATA AUDIT ===\n\n") # stampa nel terminale la scritta

# -------------------------------
# 1. Pacchetti
# -------------------------------
# required_packages <- c("readxl", "snpStats") # crea una lista di pacchetti che ci servono, readx1 serve per leggere file Excel .xlsx, snpStats serve per leggere i file PLINK

# for (pkg in required_packages) {
#   if (!requireNamespace(pkg, quietly = TRUE)) {
#     install.packages(pkg, repos = "https://cloud.r-project.org")
#   }
# } # controlla ogni pacchetto se è stato installato, se non lo è lo installa

library(readxl) # carica i pacchetti in memoria, così possiamo usare le loro funzioni
library(snpStats) # # carica i pacchetti in memoria, così possiamo usare le loro funzioni

# -------------------------------
# 2. Percorsi file
# -------------------------------
pheno_file <- "Input/Pheno_raw.xlsx"
weather_file <- "Input/Weather_raw.xlsx"
soil_file <- "Input/Soil_raw.xlsx"

bed_file <- "Input/SNPs_final_2022.bed"
bim_file <- "Input/SNPs_final_2022.bim"
fam_file <- "Input/SNPs_final_2022.fam"

# -------------------------------
# 3. Controllo esistenza file: per ogni file controlla se esiste davvero
# -------------------------------
files_to_check <- c(pheno_file, weather_file, soil_file, bed_file, bim_file, fam_file) # questa lista resitituisce TRUE se il file c'è, se no FALSE

cat("Controllo file presenti:\n")
for (f in files_to_check) {
  cat(f, "->", file.exists(f), "\n")
}
cat("\n")

# -------------------------------
# 4. Caricamento dati
# -------------------------------
cat("Caricamento file fenotipico...\n") # legge il file Excel del fenotipo e lo trasforma in dataframe
pheno <- read_xlsx(pheno_file)
pheno <- as.data.frame(pheno)

cat("Caricamento file meteo...\n") # legge il file meteo
weather <- read_xlsx(weather_file)
weather <- as.data.frame(weather)

cat("Caricamento file suolo...\n") # legge il file suolo
soil <- read_xlsx(soil_file)
soil <- as.data.frame(soil)

cat("Caricamento file genomici PLINK...\n") # legge i tre file PLINK e li mette in un oggetto R
geno <- read.plink(bed_file, bim_file, fam_file) # geno contiene geno$fam = info sugli individui, geno$map = info sugli SNPs, geno$genotypes = matrice dei genotipi

cat("Caricamento completato.\n\n")

# -------------------------------
# 5. Dimensioni dei dataset
# -------------------------------
cat("=== DIMENSIONI DATASET ===\n")
cat("Pheno_raw:", nrow(pheno), "righe x", ncol(pheno), "colonne\n")
cat("Weather_raw:", nrow(weather), "righe x", ncol(weather), "colonne\n")
cat("Soil_raw:", nrow(soil), "righe x", ncol(soil), "colonne\n")
cat("Genotype .fam:", nrow(geno$fam), "individui\n")
cat("Genotype .map:", nrow(geno$map), "SNP\n\n")

# -------------------------------
# 6. Nomi colonne
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
print(colnames(geno$fam)) # per capire come sono chiamati gli ID nel file genomico
cat("\n")

cat("=== PRIME RIGHE FAM ===\n")
print(head(geno$fam))
cat("\n")

# -------------------------------
# 7. Controlli base sul fenotipo
# -------------------------------
cat("=== CONTROLLI FENOTIPICI BASE ===\n")
# righe 104, 105 sono per controllare se esiste la colonna Genotype: se sì prende tutti i valori della colonna, elimina i duplicati con unique() e conta quanti genotipi diversi ci sono
if ("Genotype" %in% colnames(pheno)) {
  cat("Genotipi unici nel fenotipo:", length(unique(pheno$Genotype)), "\n")
} else {
  cat("ATTENZIONE: colonna 'Genotype' non trovata nel fenotipo.\n")
}
# prossime righe controllano se esiste la colonna Envir, se sì conta quanti environment unici ci sono, li stampa ordinati
if ("Envir" %in% colnames(pheno)) {
  cat("Environment unici nel fenotipo:", length(unique(pheno$Envir)), "\n")
  cat("Valori Envir:\n")
  print(sort(unique(pheno$Envir)))
} else {
  cat("ATTENZIONE: colonna 'Envir' non trovata nel fenotipo.\n")
}
cat("\n")

# -------------------------------
# 8. Match fenotipo-genotipo
# -------------------------------
cat("=== MATCH FENOTIPO vs GENOTIPO ===\n")

if ("Genotype" %in% colnames(pheno)) {
  
  pheno_ids <- unique(as.character(pheno$Genotype)) # Prende tutti i genotipi unici dal file fenotipico e li trasforma in testo per confrontarli più facilmente con gli ID del .fam
  
  fam_cols <- colnames(geno$fam) # Prende i nomi di tutte le colonne del .fam
  
  for (fc in fam_cols) {
    fam_ids <- unique(as.character(geno$fam[[fc]]))
    n_match <- sum(pheno_ids %in% fam_ids)
    cat("Match con colonna fam$", fc, ":", n_match, "su", length(pheno_ids), "\n")
  } # Per ogni colonna del .fam: prende tutti i valori unici di quella colonna, controlla quanti genotipi del fenotipo compaiono anche lì, stampa il numero di match
  
} else {
  cat("Impossibile fare il match: colonna 'Genotype' assente.\n")
}
cat("\n")

# -------------------------------
# 9. Missing values nel fenotipo
# -------------------------------
cat("=== MISSING VALUES PHENO ===\n")
missing_counts <- colSums(is.na(pheno)) # Conta quanti valori mancanti ci sono in ogni colonna del fenotipo. is.na(pheno) crea una tabella di TRUE/FALSE. colSums(...) conta quanti TRUE ci sono in ogni colonna. Quindi ottieni il numero di valori mancanti per colonna.
print(missing_counts)
cat("\n")

# -------------------------------
# 10. Salvataggio report semplice
# -------------------------------
#dir.create("Output", showWarnings = FALSE)

sink("Output/audit_report.txt") # Da questo punto in poi, invece di stampare nel terminale, R scrive tutto in un file di testo. Così avrai un report salvato e non perdi le informazioni.
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

sink() # Chiude il salvataggio nel file e torna a stampare normalmente nel terminale

cat("Report salvato in Output/audit_report.txt\n") # Ti avvisa che il report è stato salvato