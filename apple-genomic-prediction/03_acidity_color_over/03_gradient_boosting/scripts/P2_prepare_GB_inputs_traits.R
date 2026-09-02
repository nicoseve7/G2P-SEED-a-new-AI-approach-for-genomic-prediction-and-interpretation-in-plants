# ==================================================
# P2_prepare_GB_inputs_traits.R
#
# Prepares inputs for Gradient Boosting
# for the new traits:
#   - Acidity
#   - Color_over
#
# Uses the Harvest_date CV strategy as a common basis,
# because there is no specific CV for these traits.
#
# For each trait:
#   - reads the final processed phenotype
#   - aligns the phenotype and CV
#   - for each split, uses ONLY the training set
#   - estimates the mixed model:
#       Trait ~ Envir + (1 | Genotype)
#   - extracts the random effect of the genotype
#   - saves a CV*_Split*.csv file for GB
#
# Main outputs:
#   Output/Intermediate/GB_feature_selection/all.geno
#   Output/Intermediate/GB_feature_selection/Acidity/CV*_Split*.csv
#   Output/Intermediate/GB_feature_selection/Color_over/CV*_Split*.csv
# ==================================================

rm(list = ls())

cat("=== P2: PREPARE GB INPUTS FOR NEW TRAITS ===\n\n")

library(readr)
library(dplyr)
library(lme4)

# =============================================================================
# SETTINGS
# =============================================================================

TRAITS <- c("Acidity", "Color_over")

PHENO_BASE_DIR <- "03_acidity_color_over/02_phenotype_preprocessing/output"

CV_FILE <- "data/raw/cv/Harvest_date_CV.csv"

SNP_MATRIX_FILE <- paste0(
  "01_common_genomic_preprocessing/output/",
  "SNP_matrix_modeling_var_gt0.RData"
)

GB_DIR <- "03_acidity_color_over/03_gradient_boosting/output"

dir.create(GB_DIR, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# HELPERS
# =============================================================================

trait_to_file_label <- function(trait) {
  tolower(trait)
}

clean_genotype <- function(x) {
  x <- as.character(x)
  x <- sub("^G_", "", x)
  trimws(x)
}

read_trait_pheno <- function(trait) {
  pheno_file <- file.path(
    PHENO_BASE_DIR,
    trait,
    paste0(trait, "_processed_final.csv")
  )
  
  if (!file.exists(pheno_file)) {
    stop(paste0(
      "File fenotipo processato non trovato:\n",
      pheno_file,
      "\n\nYou must run P1_process_new_traits_pheno.R before"
    ))
  }
  
  pheno <- read_csv(pheno_file, show_col_types = FALSE)
  
  required_cols <- c("Genotype", "Envir", trait)
  missing_cols <- setdiff(required_cols, colnames(pheno))
  
  if (length(missing_cols) > 0) {
    stop(paste0(
      "Nel file ", pheno_file, " mancano colonne: ",
      paste(missing_cols, collapse = ", "),
      "\nColonne trovate: ",
      paste(colnames(pheno), collapse = ", ")
    ))
  }
  
  pheno <- pheno %>%
    mutate(
      Genotype = clean_genotype(Genotype),
      Envir = as.character(Envir)
    )
  
  pheno[[trait]] <- as.numeric(pheno[[trait]])
  
  pheno <- pheno %>%
    filter(!is.na(.data[[trait]]))
  
  return(pheno)
}

prepare_all_geno <- function() {
  all_geno_file <- file.path(GB_DIR, "all.geno")
  
  if (file.exists(all_geno_file)) {
    cat("[INFO] all.geno already exists, keeping existing file:\n")
    cat(all_geno_file, "\n\n")
    return(invisible(NULL))
  }
  
  if (!file.exists(SNP_MATRIX_FILE)) {
    stop(paste0("File SNP matrix non trovato:\n", SNP_MATRIX_FILE))
  }
  
  cat("Caricamento matrice SNP da RData...\n")
  load(SNP_MATRIX_FILE)   # deve caricare X_var
  
  if (!exists("X_var")) {
    stop("Dentro SNP_MATRIX_FILE non trovo l'oggetto X_var.")
  }
  
  cat("Preparazione all.geno...\n")
  
  geno_df <- as.data.frame(X_var)
  geno_df$Genotype <- rownames(X_var)
  geno_df$Genotype <- clean_genotype(geno_df$Genotype)
  
  geno_df <- geno_df %>%
    select(Genotype, everything()) %>%
    arrange(Genotype)
  
  write_csv(geno_df, all_geno_file)
  
  cat("all.geno salvato:\n")
  cat(all_geno_file, "\n")
  cat("Dimensioni all.geno:", nrow(geno_df), "x", ncol(geno_df), "\n\n")
}

fit_training_mixed_model <- function(training_df, trait, split, trait_dir) {
  
  training_df <- training_df %>%
    select(Genotype, Envir, all_of(trait)) %>%
    filter(!is.na(.data[[trait]])) %>%
    mutate(
      Genotype = as.factor(Genotype),
      Envir = as.factor(Envir)
    )
  
  n_obs_training <- nrow(training_df)
  n_genotypes_training <- n_distinct(training_df$Genotype)
  n_envir_training <- n_distinct(training_df$Envir)
  
  if (n_obs_training < 10 || n_genotypes_training < 5 || n_envir_training < 1) {
    warning(paste0(
      "Split ", split, " per trait ", trait,
      " ha pochi dati. n_obs=", n_obs_training,
      ", n_genotypes=", n_genotypes_training,
      ", n_envir=", n_envir_training
    ))
  }
  
  formula_txt <- paste0(trait, " ~ Envir + (1 | Genotype)")
  fit <- lmer(as.formula(formula_txt), data = training_df)
  
  ranef_df <- ranef(fit)$Genotype
  ranef_df$Genotype <- rownames(ranef_df)
  
  # The first column is the random effect.
  # We'll rename it using the trait's name so that P3 can easily read it.
  colnames(ranef_df)[1] <- trait
  
  out_df <- ranef_df %>%
    select(Genotype, all_of(trait)) %>%
    mutate(Genotype = clean_genotype(Genotype)) %>%
    arrange(Genotype)
  
  out_file <- file.path(trait_dir, paste0(split, ".csv"))
  write_csv(out_df, out_file)
  
  return(data.frame(
    Trait = trait,
    Split = split,
    n_obs_training = n_obs_training,
    n_genotypes_training = n_genotypes_training,
    n_envir_training = n_envir_training,
    output_file = out_file,
    status = "ok",
    stringsAsFactors = FALSE
  ))
}

process_one_trait <- function(trait, cv_repo) {
  
  cat("\n", paste(rep("#", 80), collapse = ""), "\n", sep = "")
  cat("Processing trait: ", trait, "\n", sep = "")
  cat(paste(rep("#", 80), collapse = ""), "\n\n", sep = "")
  
  trait_dir <- file.path(GB_DIR, trait)
  dir.create(trait_dir, showWarnings = FALSE, recursive = TRUE)
  
  pheno <- read_trait_pheno(trait)
  
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
  
  # Intersection phenotype x CV
  cv_aligned <- cv_repo %>%
    semi_join(pheno %>% select(Envir, Genotype), by = c("Envir", "Genotype")) %>%
    arrange(Envir, Genotype)
  
  pheno_aligned <- pheno %>%
    semi_join(cv_repo %>% select(Envir, Genotype), by = c("Envir", "Genotype")) %>%
    arrange(Envir, Genotype)
  
  if (!identical(
    paste(pheno_aligned$Envir, pheno_aligned$Genotype, sep = "|||"),
    paste(cv_aligned$Envir, cv_aligned$Genotype, sep = "|||")
  )) {
    stop(paste0("Errore di allineamento pheno/CV per trait: ", trait))
  }
  
  cat("Righe fenotipo allineato:", nrow(pheno_aligned), "\n")
  cat("Righe CV allineata:", nrow(cv_aligned), "\n\n")
  
  split_cols <- grep("^CV\\d+_Split\\d+$", colnames(cv_aligned), value = TRUE)
  
  cat("Numero split trovati:", length(split_cols), "\n")
  print(split_cols)
  cat("\n")
  
  split_summary <- list()
  
  for (split in split_cols) {
    cat("Processing ", trait, " - ", split, "...\n", sep = "")
    
    training_df <- pheno_aligned %>%
      bind_cols(cv_aligned %>% select(all_of(split))) %>%
      filter(.data[[split]] == 0) %>%
      select(Genotype, Envir, all_of(trait))
    
    res <- tryCatch(
      {
        fit_training_mixed_model(
          training_df = training_df,
          trait = trait,
          split = split,
          trait_dir = trait_dir
        )
      },
      error = function(e) {
        warning(paste0("Errore in ", trait, " - ", split, ": ", e$message))
        
        data.frame(
          Trait = trait,
          Split = split,
          n_obs_training = nrow(training_df),
          n_genotypes_training = n_distinct(training_df$Genotype),
          n_envir_training = n_distinct(training_df$Envir),
          output_file = NA,
          status = paste0("failed: ", e$message),
          stringsAsFactors = FALSE
        )
      }
    )
    
    split_summary[[split]] <- res
  }
  
  split_summary_df <- bind_rows(split_summary)
  
  label <- trait_to_file_label(trait)
  
  cv_out_file <- file.path(GB_DIR, paste0(trait, "_CV_aligned.csv"))
  pheno_out_file <- file.path(GB_DIR, paste0(trait, "_pheno_aligned.csv"))
  split_summary_file <- file.path(GB_DIR, paste0(trait, "_split_summary.csv"))
  report_file <- file.path(GB_DIR, paste0("GB_inputs_from_repo_CV_report_", label, ".txt"))
  
  write_csv(cv_aligned, cv_out_file)
  write_csv(pheno_aligned, pheno_out_file)
  write_csv(split_summary_df, split_summary_file)
  
  sink(report_file)
  
  cat("=== P2: GB INPUTS FROM HARVEST_DATE CV REPORT ===\n\n")
  
  cat("Trait:", trait, "\n\n")
  
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
  
  cat("Numero split trovati:", length(split_cols), "\n")
  print(split_cols)
  cat("\n\n")
  
  cat("Prime righe CV allineata:\n")
  print(head(cv_aligned))
  cat("\n\n")
  
  cat("Riassunto split training:\n")
  print(split_summary_df)
  cat("\n\n")
  
  cat("Output principali:\n")
  cat("- ", cv_out_file, "\n", sep = "")
  cat("- ", pheno_out_file, "\n", sep = "")
  cat("- ", split_summary_file, "\n", sep = "")
  cat("- ", trait_dir, "/CV*_Split*.csv\n", sep = "")
  
  sink()
  
  cat("\nFile salvati per ", trait, ":\n", sep = "")
  cat("- ", cv_out_file, "\n", sep = "")
  cat("- ", pheno_out_file, "\n", sep = "")
  cat("- ", split_summary_file, "\n", sep = "")
  cat("- ", report_file, "\n", sep = "")
  cat("- ", trait_dir, "/CV*_Split*.csv\n\n", sep = "")
  
  return(data.frame(
    Trait = trait,
    n_pheno_rows = nrow(pheno),
    n_aligned_rows = nrow(pheno_aligned),
    n_splits = length(split_cols),
    n_completed_splits = sum(split_summary_df$status == "ok"),
    n_failed_splits = sum(split_summary_df$status != "ok"),
    stringsAsFactors = FALSE
  ))
}

# =============================================================================
# MAIN
# =============================================================================

cat("Controllo input principali...\n")

if (!file.exists(CV_FILE)) {
  stop(paste0("CV file non trovato:\n", CV_FILE))
}

if (!file.exists(SNP_MATRIX_FILE)) {
  stop(paste0("SNP matrix file non trovato:\n", SNP_MATRIX_FILE))
}

cat("Caricamento CV comune da Harvest_date...\n")
cv_repo <- read_csv(CV_FILE, show_col_types = FALSE)

cv_repo <- cv_repo %>%
  mutate(
    Genotype = clean_genotype(Genotype),
    Envir = as.character(Envir)
  )

cat("CV rows:", nrow(cv_repo), "\n")
cat("CV unique genotypes:", n_distinct(cv_repo$Genotype), "\n")
cat("CV unique environments:", n_distinct(cv_repo$Envir), "\n\n")

prepare_all_geno()

global_summary <- list()

for (trait in TRAITS) {
  global_summary[[trait]] <- process_one_trait(trait, cv_repo)
}

global_summary_df <- bind_rows(global_summary)

global_summary_file <- file.path(GB_DIR, "P2_GB_inputs_new_traits_global_summary.csv")
write_csv(global_summary_df, global_summary_file)

sink(file.path(GB_DIR, "P2_GB_inputs_new_traits_global_report.txt"))

cat("=== P2 GLOBAL REPORT ===\n\n")
cat("Traits processed:\n")
print(TRAITS)
cat("\n\n")

cat("CV file used:\n")
cat(CV_FILE, "\n\n")

cat("SNP matrix file used:\n")
cat(SNP_MATRIX_FILE, "\n\n")

cat("Global summary:\n")
print(global_summary_df)
cat("\n")

sink()

cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("P2 completed.\n")
cat("Global summary saved:\n")
cat(global_summary_file, "\n")
cat(paste(rep("=", 80), collapse = ""), "\n", sep = "")
