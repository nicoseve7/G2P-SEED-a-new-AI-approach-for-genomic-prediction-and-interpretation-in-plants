# ==================================================
# C3_build_genomic_PCs_paper_style.R
# PCA genomiche paper-inspired con GDS + LD pruning + KING + PCAir
# ==================================================

rm(list = ls())

cat("=== STEP C3: BUILD GENOMIC PCs (PAPER STYLE) ===\n\n")

library(SeqArray)
library(SNPRelate)
library(GENESIS)
library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
bed_file <- "01_common_genomic_preprocessing/input/SNPs_final_2022.bed"
bim_file <- "01_common_genomic_preprocessing/input/SNPs_final_2022.bim"
fam_file <- "01_common_genomic_preprocessing/input/SNPs_final_2022.fam"

outdir <- "01_common_genomic_preprocessing/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

gds_file <- file.path(outdir, "SNPs_final_2022.gds")

# -------------------------------
# 2. Parametri paper-inspired
# -------------------------------
LD_threshold <- 0.975
MAF_threshold <- 0.02
n_pcs_to_save <- 20

# -------------------------------
# 3. Conversione BED -> GDS
# -------------------------------
cat("Conversione BED/BIM/FAM in GDS...\n")

if (!file.exists(gds_file)) {
  seqBED2GDS(
    bed.fn = bed_file,
    fam.fn = fam_file,
    bim.fn = bim_file,
    out.gdsfn = gds_file
  )
  cat("GDS creato:", gds_file, "\n\n")
} else {
  cat("GDS già presente, riuso il file esistente.\n\n")
}

# -------------------------------
# 4. Apertura GDS
# -------------------------------
cat("Apertura GDS...\n")
gds <- seqOpen(gds_file)

sample_ids <- seqGetData(gds, "sample.id")
variant_ids_all <- seqGetData(gds, "variant.id")

cat("Numero sample nel GDS:", length(sample_ids), "\n")
cat("Numero variant nel GDS:", length(variant_ids_all), "\n\n")

# -------------------------------
# 5. LD pruning
# -------------------------------
cat("LD pruning in corso...\n")

snpset <- snpgdsLDpruning(
  gdsobj = gds,
  ld.threshold = LD_threshold,
  autosome.only = FALSE,
  maf = MAF_threshold
)

snpset <- unlist(unname(snpset))

cat("Numero SNP dopo LD pruning:", length(snpset), "\n\n")

# Applica filtro nel GDS
seqSetFilter(gds, variant.id = snpset)

# -------------------------------
# 6. KING-robust kinship
# -------------------------------
cat("Calcolo kinship KING-robust...\n")

ibd_robust <- snpgdsIBDKING(
  gdsobj = gds,
  snp.id = snpset,
  type = "KING-robust",
  num.thread = 1
)

KINGmat <- ibd_robust$kinship
colnames(KINGmat) <- rownames(KINGmat) <- ibd_robust$sample.id

cat("KING matrix calcolata.\n\n")

# -------------------------------
# 7. PCAir
# -------------------------------
cat("Calcolo PCAir...\n")

pcair_obj <- pcair(
  gdsobj = gds,
  kinobj = KINGmat,
  divobj = KINGmat,
  snp.include = snpset,
  eigen.cnt = max(n_pcs_to_save, 10)
)

cat("PCAir completata.\n\n")

# -------------------------------
# 8. Estrazione PC
# -------------------------------
pc_mat <- as.data.frame(pcair_obj$vectors)

# assicura che gli ID dei genotipi siano una colonna esplicita
pc_mat$Genotype <- rownames(pc_mat)

# individua quante componenti ci sono davvero
n_pc_available <- ncol(pc_mat) - 1
n_pc_keep <- min(n_pcs_to_save, n_pc_available)

# tieni solo le prime n componenti + Genotype
pc_df <- pc_mat[, c("Genotype", colnames(pc_mat)[1:n_pc_keep]), drop = FALSE]

# rinomina le colonne delle componenti in PC1, PC2, ...
colnames(pc_df) <- c("Genotype", paste0("PC", seq_len(n_pc_keep)))

# -------------------------------
# 9. Varianza spiegata
# -------------------------------
# In GENESIS/PCAir gli autovalori sono in values
eigenvals <- pcair_obj$values
pve <- eigenvals / sum(eigenvals)

pve_df <- data.frame(
  PC = paste0("PC", seq_along(pve)),
  ProportionVarianceExplained = pve
)

# -------------------------------
# 10. Controlli output
# -------------------------------
cat("Numero genotipi con PC:", nrow(pc_df), "\n")
cat("Numero PC salvate:", n_pc_keep, "\n\n")

cat("Prime righe PC:\n")
print(head(pc_df))
cat("\n")

cat("Prime 10 PVE:\n")
print(head(pve_df, 10))
cat("\n")

# -------------------------------
# 11. Salvataggio output
# -------------------------------
write_csv(pc_df, file.path(outdir, "genomic_PCs_20_paper_style.csv"))
write_csv(pve_df, file.path(outdir, "genomic_PCs_variance_explained_paper_style.csv"))

sink(file.path(outdir, "genomic_PCs_paper_style_report.txt"))

cat("=== STEP C3: GENOMIC PCs PAPER STYLE REPORT ===\n\n")
cat("Numero sample nel GDS:", length(sample_ids), "\n")
cat("Numero variant iniziali:", length(variant_ids_all), "\n")
cat("Numero SNP dopo LD pruning:", length(snpset), "\n")
cat("Numero genotipi con PC:", nrow(pc_df), "\n")
cat("Numero PC salvate:", n_pc_keep, "\n\n")

cat("Prime righe PC:\n")
print(head(pc_df))
cat("\n\n")

cat("Prime 10 componenti - varianza spiegata:\n")
print(head(pve_df, 10))
cat("\n")

sink()

# -------------------------------
# 12. Chiusura GDS
# -------------------------------
seqClose(gds)

# -------------------------------
# 13. Fine
# -------------------------------
cat("File salvati:\n")
cat("- Output/SNPs_final_2022.gds\n")
cat("- Output/genomic_PCs_20_paper_style.csv\n")
cat("- Output/genomic_PCs_variance_explained_paper_style.csv\n")
cat("- Output/genomic_PCs_paper_style_report.txt\n")
