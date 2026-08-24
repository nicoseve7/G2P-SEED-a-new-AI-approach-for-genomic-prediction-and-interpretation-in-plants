# ===============================
# A2_id_mapping.R
# Definizione del mapping ID
# ===============================

rm(list = ls())

cat("=== STEP A2: ID MAPPING ===\n\n")

library(readxl)
library(snpStats)

# -------------------------------
# 1. Percorsi file
# -------------------------------
pheno_file <- "Input/Pheno_raw.xlsx"
bed_file <- "Input/SNPs_final_2022.bed"
bim_file <- "Input/SNPs_final_2022.bim"
fam_file <- "Input/SNPs_final_2022.fam"

# -------------------------------
# 2. Caricamento dati/files
# -------------------------------
cat("Caricamento fenotipo...\n")
pheno <- as.data.frame(read_xlsx(pheno_file))

cat("Caricamento file PLINK...\n")
geno <- read.plink(bed_file, bim_file, fam_file)

cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Estrazione ID (li trasformo in testo e li ordino per poterli confrontare in modo pulito)
# -------------------------------
pheno_ids <- sort(unique(as.character(pheno$Genotype))) # prendo tutti i genotipi unici del file fenotipo
pedigree_ids <- sort(unique(as.character(geno$fam$pedigree))) # prendo tutti gli ID unici di pedigree del file genomico
member_ids <- sort(unique(as.character(geno$fam$member))) # tutti gli ID unici di member del file genomico

# -------------------------------
# 4. Confronto pedigree vs member
# -------------------------------
cat("=== CONFRONTO pedigree vs member ===\n")

same_length <- length(geno$fam$pedigree) == length(geno$fam$member) # controllo se hanno la stessa lunghezza
same_values <- all(as.character(geno$fam$pedigree) == as.character(geno$fam$member)) # controlla se hanno esattamente gli stessi valori riga per riga, per capire se sono davvero equivalenti

cat("Stessa lunghezza:", same_length, "\n")
cat("Stessi valori riga per riga:", same_values, "\n\n")

# -------------------------------
# 5. Match fenotipo vs pedigree/member
# -------------------------------
cat("=== MATCH FENOTIPO ===\n")
cat("Genotipi unici nel fenotipo:", length(pheno_ids), "\n")
cat("ID unici in pedigree:", length(pedigree_ids), "\n")
cat("ID unici in member:", length(member_ids), "\n\n")

cat("Match fenotipo con pedigree:", sum(pheno_ids %in% pedigree_ids), "\n")
cat("Match fenotipo con member:", sum(pheno_ids %in% member_ids), "\n\n")

# -------------------------------
# 6. Genotipi del fenotipo senza match (trovo i nomi precisi)
# -------------------------------
pheno_not_in_pedigree <- setdiff(pheno_ids, pedigree_ids)
pheno_not_in_member <- setdiff(pheno_ids, member_ids)

cat("=== GENOTIPI DEL FENOTIPO SENZA MATCH IN pedigree ===\n")
print(pheno_not_in_pedigree)
cat("\n")

cat("=== GENOTIPI DEL FENOTIPO SENZA MATCH IN member ===\n")
print(pheno_not_in_member)
cat("\n")

# -------------------------------
# 7. Genotipi del genomico non presenti nel fenotipo
# -------------------------------
pedigree_not_in_pheno <- setdiff(pedigree_ids, pheno_ids)
member_not_in_pheno <- setdiff(member_ids, pheno_ids)

cat("=== ID genomici in pedigree NON presenti nel fenotipo ===\n")
cat("Numero:", length(pedigree_not_in_pheno), "\n\n")

cat("=== ID genomici in member NON presenti nel fenotipo ===\n")
cat("Numero:", length(member_not_in_pheno), "\n\n")

# -------------------------------
# 8. Scelta ID ufficiale
# -------------------------------
chosen_id <- "member"

cat("=== SCELTA FINALE ===\n")
cat("ID genomico scelto come riferimento:", chosen_id, "\n\n")

# -------------------------------
# 9. Creazione tabella di mapping
# -------------------------------
mapping_df <- data.frame(
  pheno_genotype = pheno_ids,
  in_pedigree = pheno_ids %in% pedigree_ids,
  in_member = pheno_ids %in% member_ids,
  stringsAsFactors = FALSE
)

dir.create("Output", showWarnings = FALSE)
write.csv(mapping_df, "Output/genotype_id_mapping.csv", row.names = FALSE)

# -------------------------------
# 10. Report testuale
# -------------------------------
sink("Output/id_mapping_report.txt")

cat("=== STEP A2: ID MAPPING REPORT ===\n\n")

cat("CONFRONTO pedigree vs member\n")
cat("Stessa lunghezza:", same_length, "\n")
cat("Stessi valori riga per riga:", same_values, "\n\n")

cat("NUMERI GENERALI\n")
cat("Genotipi unici nel fenotipo:", length(pheno_ids), "\n")
cat("ID unici in pedigree:", length(pedigree_ids), "\n")
cat("ID unici in member:", length(member_ids), "\n\n")

cat("MATCH\n")
cat("Match fenotipo con pedigree:", sum(pheno_ids %in% pedigree_ids), "\n")
cat("Match fenotipo con member:", sum(pheno_ids %in% member_ids), "\n\n")

cat("GENOTIPI DEL FENOTIPO SENZA MATCH IN pedigree\n")
print(pheno_not_in_pedigree)
cat("\n")

cat("GENOTIPI DEL FENOTIPO SENZA MATCH IN member\n")
print(pheno_not_in_member)
cat("\n")

cat("ID genomici in pedigree NON presenti nel fenotipo\n")
cat("Numero:", length(pedigree_not_in_pheno), "\n\n")

cat("ID genomici in member NON presenti nel fenotipo\n")
cat("Numero:", length(member_not_in_pheno), "\n\n")

cat("SCELTA FINALE\n")
cat("ID genomico scelto come riferimento:", chosen_id, "\n")

sink()

cat("File salvati:\n")
cat("- Output/genotype_id_mapping.csv\n")
cat("- Output/id_mapping_report.txt\n") 