# ===============================
# A2_id_mapping.R
# ID mapping definition
# ===============================

rm(list = ls())

cat("=== STEP A2: ID MAPPING ===\n\n")

library(readxl)
library(snpStats)

# -------------------------------
# 1. File paths
# -------------------------------
pheno_file <- "data/raw/phenotype/Pheno_raw.xlsx"
bed_file <- "data/raw/genotype/SNPs_final_2022.bed"
bim_file <- "data/raw/genotype/SNPs_final_2022.bim"
fam_file <- "data/raw/genotype/SNPs_final_2022.fam"

# -------------------------------
# 2. Uploading Data/Files
# -------------------------------
cat("Caricamento fenotipo...\n")
pheno <- as.data.frame(read_xlsx(pheno_file))

cat("Caricamento file PLINK...\n")
geno <- read.plink(bed_file, bim_file, fam_file)

cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Extracting IDs (I convert them to text and sort them so I can compare them neatly)
# -------------------------------
pheno_ids <- sort(unique(as.character(pheno$Genotype))) # prendo tutti i genotipi unici del file fenotipo
pedigree_ids <- sort(unique(as.character(geno$fam$pedigree))) # prendo tutti gli ID unici di pedigree del file genomico
member_ids <- sort(unique(as.character(geno$fam$member))) # tutti gli ID unici di member del file genomico

# -------------------------------
# 4. pedigree vs member
# -------------------------------
cat("=== CONFRONTO pedigree vs member ===\n")

same_length <- length(geno$fam$pedigree) == length(geno$fam$member) # controllo se hanno la stessa lunghezza
same_values <- all(as.character(geno$fam$pedigree) == as.character(geno$fam$member)) # Check to see if they have exactly the same values line by line, to determine if they are truly equivalent

cat("Stessa lunghezza:", same_length, "\n")
cat("Stessi valori riga per riga:", same_values, "\n\n")

# -------------------------------
# 5. Phenotype vs. Pedigree/Member Match
# -------------------------------
cat("=== MATCH FENOTIPO ===\n")
cat("Genotipi unici nel fenotipo:", length(pheno_ids), "\n")
cat("ID unici in pedigree:", length(pedigree_ids), "\n")
cat("ID unici in member:", length(member_ids), "\n\n")

cat("Match fenotipo con pedigree:", sum(pheno_ids %in% pedigree_ids), "\n")
cat("Match fenotipo con member:", sum(pheno_ids %in% member_ids), "\n\n")

# -------------------------------
# 6. Genotypes for the phenotype with no matches (I'll find the exact names)
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
# 7. Genomic genotypes not present in the phenotype
# -------------------------------
pedigree_not_in_pheno <- setdiff(pedigree_ids, pheno_ids)
member_not_in_pheno <- setdiff(member_ids, pheno_ids)

cat("=== ID genomici in pedigree NON presenti nel fenotipo ===\n")
cat("Numero:", length(pedigree_not_in_pheno), "\n\n")

cat("=== ID genomici in member NON presenti nel fenotipo ===\n")
cat("Numero:", length(member_not_in_pheno), "\n\n")

# -------------------------------
# 8. Official ID Selection
# -------------------------------
chosen_id <- "member"

cat("=== SCELTA FINALE ===\n")
cat("ID genomico scelto come riferimento:", chosen_id, "\n\n")

# -------------------------------
# 9. Creating a Mapping Table
# -------------------------------
mapping_df <- data.frame(
  pheno_genotype = pheno_ids,
  in_pedigree = pheno_ids %in% pedigree_ids,
  in_member = pheno_ids %in% member_ids,
  stringsAsFactors = FALSE
)

outdir <- "02_harvest_date/01_data_audit/output"
write.csv(
  mapping_df,
  file.path(outdir, "genotype_id_mapping.csv"),
  row.names = FALSE
)

# -------------------------------
# 10. Report
# -------------------------------
sink(file.path(outdir, "id_mapping_report.txt"))

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
