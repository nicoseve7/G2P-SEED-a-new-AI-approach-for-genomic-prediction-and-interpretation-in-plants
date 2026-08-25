# # ==================================================
# # D3_prepare_GB_inputs_from_repo_CV.R
# # Prepara gli input per B_Feature_selection_GB.py
# # usando la CV strategy del repo
# # ==================================================

rm(list = ls())

cat("=== STEP D3: PREPARE GB INPUTS FROM REPO CV (PAPER-INSPIRED) ===\n\n")

library(readr)
library(dplyr)
library(lme4)

# -------------------------------
# 1. Percorsi file
# -------------------------------
pheno_file <- paste0(
  "02_harvest_date/02_phenotype_preprocessing/output/",
  "Harvest_date_processed_final.csv"
)

cv_file <- "data/raw/cv/Harvest_date_CV.csv"

snp_matrix_file <- paste0(
  "01_common_genomic_preprocessing/output/",
  "SNP_matrix_modeling_var_gt0.RData"
)

gb_dir <- "02_harvest_date/05_gradient_boosting/output"
trait_dir <- file.path(gb_dir, "Harvest_date")

dir.create(gb_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(trait_dir, showWarnings = FALSE, recursive = TRUE)
# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento fenotipo finale...\n")
pheno <- read_csv(pheno_file, show_col_types = FALSE)

cat("Caricamento CV del repo...\n")
cv_repo <- read_csv(cv_file, show_col_types = FALSE)

cat("Caricamento matrice SNP...\n")
load(snp_matrix_file)   # carica X_var

cat("Caricamento completato.\n\n")
# -------------------------------
# 3. Chiavi univoche per confronto
# -------------------------------
pheno <- pheno %>%
  mutate(key = paste(Envir, Genotype, sep = "|||"))

cv_repo <- cv_repo %>%
  mutate(key = paste(Envir, Genotype, sep = "|||"))

pheno_keys <- unique(pheno$key)
cv_keys <- unique(cv_repo$key)

only_in_pheno <- setdiff(pheno_keys, cv_keys)
only_in_cv <- setdiff(cv_keys, pheno_keys)

cat("Righe uniche nel fenotipo finale:", length(pheno_keys), "\n")
cat("Righe uniche nella CV repo:", length(cv_keys), "\n")
cat("Righe presenti solo nel fenotipo finale:", length(only_in_pheno), "\n")
cat("Righe presenti solo nella CV repo:", length(only_in_cv), "\n\n")

# -------------------------------
# 4. Intersezione pheno x CV
# -------------------------------
cat("Costruzione intersezione pheno x CV...\n")

cv_aligned <- cv_repo %>%
  semi_join(pheno %>% select(Envir, Genotype), by = c("Envir", "Genotype")) %>%
  arrange(Envir, Genotype)

pheno_aligned <- pheno %>%
  semi_join(cv_repo %>% select(Envir, Genotype), by = c("Envir", "Genotype")) %>%
  arrange(Envir, Genotype)

stopifnot(
  identical(
    paste(pheno_aligned$Envir, pheno_aligned$Genotype, sep = "|||"),
    paste(cv_aligned$Envir, cv_aligned$Genotype, sep = "|||")
  )
)

cat("Righe fenotipo allineato:", nrow(pheno_aligned), "\n")
cat("Righe CV allineata:", nrow(cv_aligned), "\n\n")

# -------------------------------
# 5. Costruzione all.geno da X_var
# -------------------------------
cat("Preparazione all.geno...\n")

geno_df <- as.data.frame(X_var)
geno_df$Genotype <- rownames(X_var)
geno_df <- geno_df %>%
  select(Genotype, everything()) %>%
  arrange(Genotype)

write_csv(geno_df, file.path(gb_dir, "all.geno"))

cat("all.geno salvato.\n")
cat("Dimensioni all.geno:", nrow(geno_df), "x", ncol(geno_df), "\n\n")

# -------------------------------
# 6. Split reali del repo
# -------------------------------
split_cols <- grep("^CV\\d+_Split\\d+$", colnames(cv_aligned), value = TRUE)

cat("Numero split trovati:", length(split_cols), "\n\n")

# -------------------------------
# 7. Costruzione file training per split
#    PAPER-INSPIRED:
#    mixed model training-only:
#    Harvest_date ~ Envir + (1 | Genotype)
#    output = random effect del genotipo
# -------------------------------
cat("Creazione file training paper-inspired per tutti gli split...\n")

split_summary <- list()

for (split in split_cols) {
  cat("Processing", split, "...\n")

  training_df <- pheno_aligned %>%
    bind_cols(cv_aligned %>% select(all_of(split))) %>%
    filter(.data[[split]] == 0) %>%
    select(Genotype, Envir, Harvest_date)

  training_df <- training_df %>%
    mutate(
      Genotype = as.factor(Genotype),
      Envir = as.factor(Envir)
    )

  n_obs_training <- nrow(training_df)
  n_genotypes_training <- n_distinct(training_df$Genotype)
  n_envir_training <- n_distinct(training_df$Envir)

  # fit mixed model
  fit <- lmer(Harvest_date ~ Envir + (1 | Genotype), data = training_df)

  # extract genotype random effects
  ranef_df <- ranef(fit)$Genotype
  ranef_df$Genotype <- rownames(ranef_df)

  # nome colonna compatibile con lo script python
  colnames(ranef_df)[1] <- "Harvest_date"

  out_df <- ranef_df %>%
    select(Genotype, Harvest_date) %>%
    arrange(Genotype)

  out_file <- file.path(trait_dir, paste0(split, ".csv"))
  write_csv(out_df, out_file)

  split_summary[[split]] <- data.frame(
    Split = split,
    n_obs_training = n_obs_training,
    n_genotypes_training = n_genotypes_training,
    n_envir_training = n_envir_training,
    stringsAsFactors = FALSE
  )

  cat("  osservazioni training:", n_obs_training, "\n")
  cat("  genotipi training:", n_genotypes_training, "\n")
}

split_summary_df <- bind_rows(split_summary)

# -------------------------------
# 8. Salvataggi aggiuntivi
# -------------------------------
write_csv(cv_aligned, file.path(gb_dir, "Harvest_date_CV_aligned.csv"))
write_csv(split_summary_df, file.path(gb_dir, "Harvest_date_split_summary.csv"))

# -------------------------------
# 9. Report finale
# -------------------------------
sink(file.path(gb_dir, "GB_inputs_from_repo_CV_report.txt"))

cat("=== STEP D3: GB INPUTS FROM REPO CV REPORT (PAPER-INSPIRED) ===\n\n")

cat("Fenotipo finale righe:", nrow(pheno), "\n")
cat("CV repo righe:", nrow(cv_repo), "\n\n")

cat("Righe presenti solo nel fenotipo finale:", length(only_in_pheno), "\n")
if (length(only_in_pheno) > 0) print(only_in_pheno)
cat("\n")

cat("Righe presenti solo nella CV repo:", length(only_in_cv), "\n")
if (length(only_in_cv) > 0) print(only_in_cv)
cat("\n")

cat("Fenotipo allineato righe:", nrow(pheno_aligned), "\n")
cat("CV allineata righe:", nrow(cv_aligned), "\n\n")

cat("all.geno dimensioni:", nrow(geno_df), "x", ncol(geno_df), "\n\n")

cat("Numero split trovati:", length(split_cols), "\n")
print(split_cols)
cat("\n\n")

cat("Prime righe CV allineata:\n")
print(head(cv_aligned))
cat("\n\n")

cat("Riassunto split training:\n")
print(split_summary_df)
cat("\n")

sink()

cat("File salvati:\n")
cat("- ", file.path(gb_dir, "all.geno"), "\n")
cat("- ", file.path(gb_dir, "Harvest_date_CV_aligned.csv"), "\n")
cat("- ", file.path(trait_dir, "<25 split>.csv"), "\n")
cat("- ", file.path(gb_dir, "Harvest_date_split_summary.csv"), "\n")
cat("- ", file.path(gb_dir, "GB_inputs_from_repo_CV_report.txt"), "\n")
