# ==================================================
# A7_harvestdate_phenotypic_model.R
# Phenotypic model finale per Harvest_date
# ==================================================

rm(list = ls())

cat("=== STEP A7: HARVEST_DATE PHENOTYPIC MODEL ===\n\n")

library(readr)
library(dplyr)
library(lme4)

# -------------------------------
# 1. Percorso file
# -------------------------------
outdir <- "02_harvest_date/02_phenotype_preprocessing/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

input_file <- file.path(
  outdir,
  "harvestdate_adjusted_values_trees.csv"
)
# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento dati tree-level aggiustati...\n") # Carichiamo il file tree-level aggiustato. Perché qui vogliamo ancora la struttura completa delle osservazioni
trees <- read_csv(input_file, show_col_types = FALSE)
cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Preparazione dati
# -------------------------------
trees <- trees %>%
  rename(Value = Harvest_date_adjusted_tree) # Rinominiamo la colonna del fenotipo in Value. Questo serve solo a rendere il codice più leggibile

# rimuovi eventuali NA
trees <- trees %>%
  filter(!is.na(Value))

cat("Numero righe usate:", nrow(trees), "\n")
cat("Numero genotipi unici:", n_distinct(trees$Genotype), "\n")
cat("Numero environment unici:", n_distinct(trees$Envir), "\n\n")

trees$Genotype <- as.factor(trees$Genotype)
trees$Envir <- as.factor(trees$Envir) # Togliamo eventuali NA e convertiamo Genotype ed Envir in fattori. Questo è standard per il mixed model

# -------------------------------
# 4. Fit del phenotypic model
# -------------------------------
cat("Fitting model: Value ~ Envir + (1|Genotype) + (1|Genotype:Envir)\n")
fit <- lmer(Value ~ Envir + (1 | Genotype) + (1 | Genotype:Envir), data = trees)
cat("Modello stimato.\n\n")

# -------------------------------
# 5. Estrazione componenti di varianza
# -------------------------------
vc <- as.data.frame(VarCorr(fit)) # Usiamo VarCorr(fit) per estrarre le componenti di varianza. Questo ci restituisce: quanto pesa Genotype, quanto pesa Genotype:Envir, quanto pesa il residuo

# teniamo solo componenti random + residuo
vc_out <- vc[, c("grp", "vcov")]
colnames(vc_out) <- c("Component", "Variance")

# rinomina più leggibile
vc_out$Component[vc_out$Component == "Genotype"] <- "Genotype"
vc_out$Component[vc_out$Component == "Genotype:Envir"] <- "Genotype_by_Envir"
vc_out$Component[vc_out$Component == "Residual"] <- "Residual"

# -------------------------------
# 6. Proporzioni di varianza
# -------------------------------
total_var <- sum(vc_out$Variance)
vc_out$Proportion <- vc_out$Variance / total_var

# aggiungi trait
vc_out$Trait <- "Harvest_date"

# riordina colonne
vc_out <- vc_out[, c("Trait", "Component", "Variance", "Proportion")]

# -------------------------------
# 7. Versione larga
# -------------------------------
vc_wide <- data.frame(
  Trait = "Harvest_date",
  varG = vc_out$Variance[vc_out$Component == "Genotype"],
  varGE = vc_out$Variance[vc_out$Component == "Genotype_by_Envir"],
  varRes = vc_out$Variance[vc_out$Component == "Residual"],
  propG = vc_out$Proportion[vc_out$Component == "Genotype"],
  propGE = vc_out$Proportion[vc_out$Component == "Genotype_by_Envir"],
  propRes = vc_out$Proportion[vc_out$Component == "Residual"]
)

# -------------------------------
# 8. Salvataggio output
# -------------------------------
write_csv(
  vc_out,
  file.path(outdir, "harvestdate_phenotypic_variance_components_long.csv")
)

write_csv(
  vc_wide,
  file.path(outdir, "harvestdate_phenotypic_variance_components_wide.csv")
)

sink(file.path(outdir, "harvestdate_phenotypic_model_report.txt"))

cat("=== STEP A7: HARVEST_DATE PHENOTYPIC MODEL REPORT ===\n\n")
cat("Numero righe usate:", nrow(trees), "\n")
cat("Numero genotipi unici:", n_distinct(trees$Genotype), "\n")
cat("Numero environment unici:", n_distinct(trees$Envir), "\n\n")

cat("Componenti di varianza:\n")
print(vc_out)
cat("\n\n")

cat("Versione wide:\n")
print(vc_wide)
cat("\n")

sink()

# -------------------------------
# 9. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- ", file.path(outdir, "harvestdate_phenotypic_variance_components_long.csv"), "\n")
cat("- ", file.path(outdir, "harvestdate_phenotypic_variance_components_wide.csv"), "\n")
cat("- ", file.path(outdir, "harvestdate_phenotypic_model_report.txt"), "\n")
