# ==================================================
# A6_harvestdate_outlier_detection_paper.R
# Outlier detection paper-inspired con BH-MADR
# ==================================================

rm(list = ls())

cat("=== STEP A6: HARVEST_DATE OUTLIER DETECTION (PAPER METHOD) ===\n\n")

library(readr)
library(dplyr)
library(lme4)
library(multtest)

# --------------------------------------------------
# Funzione outliers dal paper, adattata
# --------------------------------------------------
outliers <- function(mydata, mytrait, mylm) {
  
  # residui del mixed model
  lmer.data <- mylm
  resi <- cbind(residuals(lmer.data, type = "response"))
  
  # re-scaled MAD
  med <- median(resi, na.rm = TRUE)
  MAD <- median(abs(resi - med), na.rm = TRUE)
  re_MAD <- MAD * 1.4826
  
  # protezione nel caso MAD sia zero o NA
  if (is.na(re_MAD) || re_MAD == 0) {
    return(mydata[0, , drop = FALSE])
  }
  
  # residui standardizzati con MAD
  res_MAD <- resi / re_MAD
  
  # p-value grezzi
  rawp <- 2 * (1 - pnorm(abs(res_MAD)))
  
  # riduzione dati senza missing
  dataset_name_1 <- subset(mydata, !is.na(mydata[, mytrait]))
  
  # combina dataset e residui
  rawp2 <- cbind(dataset_name_1, resi, res_MAD, rawp)
  
  # correzione Holm
  res2 <- mt.rawp2adjp(rawp, proc = c("Holm"))
  
  adjp  <- cbind(res2[[1]][,1])
  bholm <- cbind(res2[[1]][,2])
  index <- cbind(res2[[2]])
  
  out_flag <- ifelse(bholm < 0.05, "OUTLIER", ".")
  
  bholm_test <- cbind(adjp, bholm, index, out_flag)
  bholm_test2 <- bholm_test[order(index), ]
  
  colnames(bholm_test2) <- c("rawp", "bholm", "index", "out_flag")
  
  total_m4_data <- cbind(rawp2, bholm_test2)
  
  return(total_m4_data[which(total_m4_data$out_flag != "."), ])
}

# -------------------------------
# 1. Percorsi file
# -------------------------------
outdir <- "02_harvest_date/02_phenotype_preprocessing/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

input_file <- file.path(
  outdir,
  "harvestdate_adjusted_values_genotype.csv"
)
# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento valori aggiustati genotype-level...\n")
pheno_adj <- read_csv(input_file, show_col_types = FALSE)
cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Rinomina per coerenza
# -------------------------------
pheno_adj <- pheno_adj %>%
  rename(Value = Harvest_date_adjusted)

# -------------------------------
# 4. Controlli base
# -------------------------------
cat("Numero righe iniziali:", nrow(pheno_adj), "\n")
cat("Numero genotipi unici:", n_distinct(pheno_adj$Genotype), "\n")
cat("Numero environment unici:", n_distinct(pheno_adj$Envir), "\n\n")

# -------------------------------
# 5. Prepara variabili
# -------------------------------
pheno_adj$Genotype <- as.factor(pheno_adj$Genotype)
pheno_adj$Envir <- as.factor(pheno_adj$Envir)

# -------------------------------
# 6. Fit mixed model
# -------------------------------
cat("Fitting mixed model: Value ~ Envir + (1|Genotype)\n")
mylm <- lmer(Value ~ Envir + (1 | Genotype), data = pheno_adj)
cat("Modello stimato.\n\n")

# -------------------------------
# 7. Trova outlier
# -------------------------------
cat("Ricerca outlier con metodo BH-MADR...\n")
outl <- outliers(pheno_adj, "Value", mylm)
cat("Outlier trovati:", nrow(outl), "\n\n")

# -------------------------------
# 8. Identifica righe da rimuovere
# -------------------------------
pheno_adj$id_row <- 1:nrow(pheno_adj)

if (nrow(outl) > 0) {
  # L'ordine è coerente con i dati usati nel modello senza missing
  outlier_indices <- as.numeric(outl$index)
  pheno_no_outliers <- pheno_adj[-outlier_indices, ]
} else {
  pheno_no_outliers <- pheno_adj
}

# pulizia colonna tecnica
pheno_no_outliers$id_row <- NULL

# -------------------------------
# 9. Salvataggio output
# -------------------------------
write_csv(
  outl,
  file.path(outdir, "harvestdate_outliers_paper_method.csv")
)

write_csv(
  pheno_no_outliers,
  file.path(outdir, "harvestdate_processed_no_outliers.csv")
)

sink(file.path(outdir, "harvestdate_outlier_report_paper_method.txt"))

cat("=== STEP A6: HARVEST_DATE OUTLIER REPORT (PAPER METHOD) ===\n\n")
cat("Numero osservazioni iniziali:", nrow(pheno_adj), "\n")
cat("Numero outlier trovati:", nrow(outl), "\n")
cat("Numero osservazioni finali:", nrow(pheno_no_outliers), "\n\n")

cat("Prime righe outlier:\n")
print(head(outl))
cat("\n")

sink()

# -------------------------------
# 10. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- ", file.path(outdir, "harvestdate_outliers_paper_method.csv"), "\n")
cat("- ", file.path(outdir, "harvestdate_processed_no_outliers.csv"), "\n")
cat("- ", file.path(outdir, "harvestdate_outlier_report_paper_method.txt"), "\n")
