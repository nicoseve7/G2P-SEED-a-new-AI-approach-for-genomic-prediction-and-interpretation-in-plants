# ==================================================
# A5_harvestdate_spatial_correction.R
# Correzione spaziale di Harvest_date con SpATS
# ==================================================

rm(list = ls())

cat("=== STEP A5: HARVEST_DATE SPATIAL CORRECTION ===\n\n")

library(readxl)
library(snpStats)
library(SpATS)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
pheno_file <- "data/raw/phenotype/Pheno_raw.xlsx"
bed_file <- "data/raw/genotype/SNPs_final_2022.bed"
bim_file <- "data/raw/genotype/SNPs_final_2022.bim"
fam_file <- "data/raw/genotype/SNPs_final_2022.fam"

outdir <- "02_harvest_date/02_phenotype_preprocessing/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento fenotipo...\n")
pheno <- as.data.frame(read_xlsx(pheno_file))

cat("Caricamento genomica PLINK...\n")
geno <- read.plink(bed_file, bim_file, fam_file)

cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Teniamo solo colonne utili
# -------------------------------
d <- pheno[, c("Row", "Position", "Genotype", "Management", "Checks",
               "Year", "Country", "Envir", "Harvest_date")]
colnames(d)[ncol(d)] <- "Trait"

# -------------------------------
# 4. Filtro per genotipi con genomica
# -------------------------------
geno_ids <- as.character(geno$fam$member)
d <- d[which(as.character(d$Genotype) %in% geno_ids), ] # tengo genotipi con genomica

cat("Numero righe dopo filtro genomico:", nrow(d), "\n")
cat("Numero genotipi unici dopo filtro genomico:", length(unique(d$Genotype)), "\n\n")

# -------------------------------
# 5. Inizializzazione output
# -------------------------------
trees_all <- NULL
means_all <- NULL

env_list <- sort(unique(d$Envir))

# -------------------------------
# 6. Loop sugli environment (Faccio un environment alla volta: BEL.2018, BEL.2019, ..., ITA.2022. Perchè il paper corregge la spatial heterogeneity separatamente in ogni environment)
# -------------------------------
for (env in env_list) {
  
  cat("Processo environment:", env, "\n")
  
  data_env <- d[d$Envir == env, ]
  
  # rimuovi missing del trait
  data_env <- data_env[!is.na(data_env$Trait), ]
  
  # se troppo pochi dati, salta
  if (nrow(data_env) < 10 || length(unique(data_env$Genotype)) < 2) {
    cat("  -> saltato: dati insufficienti\n")
    next
  }
  
  # prepara coordinate spaziali (uso Row e Position e le trasformo in coordinate numeriche e fattori; questo permette a SpATS di modellare la struttua spaziale del campo)
  data_env$column <- as.numeric(data_env$Row)
  data_env$row <- as.numeric(data_env$Position)
  data_env$C <- as.factor(data_env$column)
  data_env$R <- as.factor(data_env$row)
  data_env$Genotype <- as.factor(as.character(data_env$Genotype))
  
  # ordina per posizione
  data_env <- data_env[order(data_env$Row, data_env$Position), ]
  
  # dataframe base per output tree-level
  trees_env <- data_env[, c("Row", "Position", "Genotype", "Management",
                            "Checks", "Year", "Country", "Envir")]
  
  # dataframe base per output genotype-level
  means_env <- data.frame(
    Genotype = as.character(unique(data_env$Genotype)),
    stringsAsFactors = FALSE
  )
  
  # modello spaziale SpATS: qui il software cerca una superficie spaziale che descriva l'andamento del fenotipo nel campo e in parallelo tiene conto dei genotipi
  spatial_formula <- as.formula(~ PSANOVA(column, row))
  
  fit <- try(
    SpATS(
      response = "Trait",
      spatial = spatial_formula,
      genotype = "Genotype",
      genotype.as.random = TRUE,
      fixed = NULL,
      random = ~ C + R,
      data = data_env,
      control = controlSpATS()
    ),
    silent = TRUE
  )
  
  if (inherits(fit, "try-error")) {
    cat("  -> modello SpATS fallito\n")
    next
  }
  
  # -------------------------------
  # 6A. Valori aggiustati per genotipo
  # -------------------------------
  pred_gen <- predict(fit, which = "Genotype")
  pred_gen <- pred_gen[, c("Genotype", "predicted.values")]
  colnames(pred_gen)[2] <- "Harvest_date_adjusted"
  
  means_env <- merge(means_env, pred_gen, all.x = TRUE)
  means_env$Envir <- env
  
  # -------------------------------
  # 6B. Valori aggiustati per tree
  # -------------------------------
  d_fit <- fit$data
  d_fit$weights <- NULL
  colnames(d_fit)[ncol(d_fit)] <- "Trait_original"
  d_fit$residuals <- fit$residuals
  
  pred_gen$Genotype <- as.character(pred_gen$Genotype)
  d_fit$Genotype <- as.character(d_fit$Genotype)
  
  d_merge <- merge(d_fit, pred_gen, all.x = TRUE)
  
  # se il trait originale era missing, tieni NA
  d_merge[is.na(d_merge$Trait_original), "Harvest_date_adjusted"] <- NA
  
  # adjusted per tree = predicted genotype + residual
  d_merge$Harvest_date_adjusted_tree <- d_merge$Harvest_date_adjusted + d_merge$residuals
  
  # riordina
  d_merge <- d_merge[order(d_merge$Row, d_merge$Position), ]
  
  trees_env$Harvest_date_adjusted_tree <- d_merge$Harvest_date_adjusted_tree
  
  # salva nell'output complessivo
  trees_all <- bind_rows(trees_all, trees_env)
  means_all <- bind_rows(means_all, means_env)
}

# -------------------------------
# 7. Salvataggio output
# -------------------------------
write.csv(
  trees_all,
  file.path(outdir, "harvestdate_adjusted_values_trees.csv"),
  row.names = FALSE
)

write.csv(
  means_all,
  file.path(outdir, "harvestdate_adjusted_values_genotype.csv"),
  row.names = FALSE
)

sink(file.path(outdir, "harvestdate_spatial_correction_report.txt"))

cat("=== STEP A5: HARVEST_DATE SPATIAL CORRECTION REPORT ===\n\n")

cat("Numero righe output tree-level:\n")
print(nrow(trees_all))
cat("\n")

cat("Numero righe output genotype-level:\n")
print(nrow(means_all))
cat("\n")

cat("Prime righe tree-level:\n")
print(head(trees_all))
cat("\n")

cat("Prime righe genotype-level:\n")
print(head(means_all))
cat("\n")

sink()

# -------------------------------
# 8. Stampa finale
# -------------------------------
cat("\nFile salvati:\n")
cat("- ", file.path(outdir, "harvestdate_adjusted_values_trees.csv"), "\n")
cat("- ", file.path(outdir, "harvestdate_adjusted_values_genotype.csv"), "\n")
cat("- ", file.path(outdir, "harvestdate_spatial_correction_report.txt"), "\n")
