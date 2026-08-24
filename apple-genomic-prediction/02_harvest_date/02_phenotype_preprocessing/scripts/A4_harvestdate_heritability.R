# ==========================================
# A4_harvestdate_heritability.R
# Heritability di Harvest_date per environment
# ==========================================

rm(list = ls())

cat("=== STEP A4: HARVEST_DATE HERITABILITY ===\n\n")

library(readxl)
library(snpStats)
library(lme4)

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
cat("Caricamento fenotipo...\n") # carico file fenotipico
pheno <- as.data.frame(read_xlsx(pheno_file)) # carico file genomico

cat("Caricamento genomica PLINK...\n")
geno <- read.plink(bed_file, bim_file, fam_file)

cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Selezione colonne utili (focus Harvest_date)
# -------------------------------
d <- pheno[, c("Envir", "Genotype", "Harvest_date")]
colnames(d)[3] <- "Trait"

# -------------------------------
# 4. Teniamo solo genotipi con genomica
# -------------------------------
geno_ids <- as.character(geno$fam$member)
d <- d[which(as.character(d$Genotype) %in% geno_ids), ]

cat("Numero righe dopo filtro genomico:", nrow(d), "\n")
cat("Numero genotipi unici dopo filtro genomico:", length(unique(d$Genotype)), "\n\n")

# -------------------------------
# 5. Calcolo H² per environment
# -------------------------------
results <- NULL

env_list <- sort(unique(d$Envir))

for (env in env_list) {
  
  cat("Processo environment:", env, "\n")
  
  d_sub <- subset(d, Envir == env)
  
  # rimuovi missing del trait
  d_sub <- d_sub[!is.na(d_sub$Trait), ]
  
  # controlli minimi
  n_obs <- nrow(d_sub)
  n_gen <- length(unique(d_sub$Genotype))
  
  if (n_obs < 2 || n_gen < 2) {
    results <- rbind(results, data.frame(
      Trait = "Harvest_date",
      Envir = env,
      H2 = NA,
      n_obs = n_obs,
      n_genotypes = n_gen,
      mean_reps = NA,
      genov = NA,
      errorv = NA,
      status = "too_few_data",
      stringsAsFactors = FALSE
    ))
    next
  }
  
  # fattori
  d_sub$Genotype <- as.factor(as.character(d_sub$Genotype))
  d_sub$Envir <- as.factor(as.character(d_sub$Envir))
  
  # modello
  fit <- try(lmer(Trait ~ (1 | Genotype), data = d_sub), silent = TRUE)
  
  if (inherits(fit, "try-error")) {
    results <- rbind(results, data.frame(
      Trait = "Harvest_date",
      Envir = env,
      H2 = NA,
      n_obs = n_obs,
      n_genotypes = n_gen,
      mean_reps = NA,
      genov = NA,
      errorv = NA,
      status = "model_failed",
      stringsAsFactors = FALSE
    ))
    next
  }
  
  # componenti di varianza
  vc <- as.data.frame(VarCorr(fit))
  genov <- vc$vcov[vc$grp == "Genotype"]
  errorv <- sigma(fit)^2
  
  # repliche medie per genotipo
  reps_table <- table(d_sub$Genotype)
  mean_reps <- mean(as.numeric(reps_table))
  
  # H²
  H2 <- genov / (genov + errorv / mean_reps)
  
  results <- rbind(results, data.frame(
    Trait = "Harvest_date",
    Envir = env,
    H2 = H2,
    n_obs = n_obs,
    n_genotypes = n_gen,
    mean_reps = mean_reps,
    genov = genov,
    errorv = errorv,
    status = "ok",
    stringsAsFactors = FALSE
  ))
}

# -------------------------------
# 6. Ordinamento risultati
# -------------------------------
results <- results[order(results$H2, decreasing = TRUE, na.last = TRUE), ]

# -------------------------------
# 7. Flag H² < 0.1
# -------------------------------
results$low_H2_flag <- ifelse(!is.na(results$H2) & results$H2 < 0.1, TRUE, FALSE)

# -------------------------------
# 8. Salvataggio output
# -------------------------------
write.csv(
  results,
  file.path(outdir, "harvestdate_heritability_by_environment.csv"),
  row.names = FALSE
)

sink(file.path(outdir, "harvestdate_heritability_report.txt"))

cat("=== STEP A4: HARVEST_DATE HERITABILITY REPORT ===\n\n")
print(results)
cat("\n")

cat("Numero environment con H2 < 0.1:",
    sum(results$low_H2_flag, na.rm = TRUE), "\n")

sink()

# -------------------------------
# 9. Stampa finale
# -------------------------------
cat("\nFile salvati:\n")
cat("- ", file.path(outdir, "harvestdate_heritability_by_environment.csv"), "\n")
cat("- ", file.path(outdir, "harvestdate_heritability_report.txt"), "\n")
