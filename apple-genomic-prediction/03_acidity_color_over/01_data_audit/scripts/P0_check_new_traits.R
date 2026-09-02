# -*- coding: utf-8 -*-

################################################################################
### P0_check_new_traits.R
### Initial audit for Acidity and Color_over
################################################################################

rm(list = ls())

cat("================================================================================\n")
cat("P0 - CHECK NEW TRAITS FOR NO-SOIL PIPELINE\n")
cat("================================================================================\n\n")

library(readxl)
library(dplyr)
library(readr)
library(snpStats)

TRAITS <- c("Acidity", "Color_over")

REPORT_DIR <- "03_acidity_color_over/01_data_audit/output"
dir.create(REPORT_DIR, recursive = TRUE, showWarnings = FALSE)

PHENO_FILE <- "data/raw/phenotype/Pheno_raw.xlsx"

BED_FILE <- "data/raw/genotype/SNPs_final_2022.bed"
BIM_FILE <- "data/raw/genotype/SNPs_final_2022.bim"
FAM_FILE <- "data/raw/genotype/SNPs_final_2022.fam"

SAVE_GLOBAL <- file.path(
  REPORT_DIR,
  "P0_new_traits_summary_global.csv"
)

SAVE_BY_ENV <- file.path(
  REPORT_DIR,
  "P0_new_traits_summary_by_environment.csv"
)

SAVE_MATCH <- file.path(
  REPORT_DIR,
  "P0_new_traits_genotype_match.csv"
)

SAVE_REPORT <- file.path(
  REPORT_DIR,
  "P0_new_traits_audit_report.txt"
)

check_file <- function(path) {
  exists <- file.exists(path)
  cat(path, "->", exists, "\n")
  if (!exists) {
    stop(paste("File non trovato:", path))
  }
}

clean_genotype <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  x <- gsub("^G_", "", x)
  return(x)
}


cat("Controllo file richiesti:\n")
check_file(PHENO_FILE)
check_file(BED_FILE)
check_file(BIM_FILE)
check_file(FAM_FILE)
cat("\n")

cat("Caricamento Pheno_raw...\n")
pheno <- as.data.frame(read_xlsx(PHENO_FILE))

cat("Caricamento PLINK...\n")
geno <- read.plink(BED_FILE, BIM_FILE, FAM_FILE)

cat("Caricamento completato.\n\n")


required_base_cols <- c("Genotype", "Envir")
missing_base_cols <- setdiff(required_base_cols, colnames(pheno))

if (length(missing_base_cols) > 0) {
  stop(paste(
    "Mancano colonne fondamentali nel file fenotipico:",
    paste(missing_base_cols, collapse = ", ")
  ))
}

traits_found <- TRAITS[TRAITS %in% colnames(pheno)]
traits_missing <- setdiff(TRAITS, colnames(pheno))

cat("Trait richiesti:\n")
print(TRAITS)
cat("\n")

cat("Trait trovati nel file:\n")
print(traits_found)
cat("\n")

if (length(traits_missing) > 0) {
  cat("ATTENZIONE: questi trait NON sono stati trovati:\n")
  print(traits_missing)
  cat("\n")
}

if (length(traits_found) == 0) {
  stop("Nessuno dei trait richiesti è presente nel file fenotipico.")
}


pheno$Genotype <- clean_genotype(pheno$Genotype)
pheno$Envir <- trimws(as.character(pheno$Envir))

geno_ids <- clean_genotype(geno$fam$member)


summary_global <- lapply(traits_found, function(tr) {
  df_sub <- pheno[!is.na(pheno[[tr]]), ]

  data.frame(
    Trait = tr,
    non_missing_obs = nrow(df_sub),
    missing_obs = sum(is.na(pheno[[tr]])),
    total_rows = nrow(pheno),
    missing_fraction = round(sum(is.na(pheno[[tr]])) / nrow(pheno), 6),
    unique_genotypes = length(unique(df_sub$Genotype)),
    unique_envir = length(unique(df_sub$Envir)),
    genotypes_matching_genomic = sum(unique(df_sub$Genotype) %in% geno_ids),
    genotypes_not_matching_genomic = sum(!(unique(df_sub$Genotype) %in% geno_ids)),
    min_value = suppressWarnings(min(df_sub[[tr]], na.rm = TRUE)),
    mean_value = suppressWarnings(mean(df_sub[[tr]], na.rm = TRUE)),
    median_value = suppressWarnings(median(df_sub[[tr]], na.rm = TRUE)),
    max_value = suppressWarnings(max(df_sub[[tr]], na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
})

summary_global <- bind_rows(summary_global)
write_csv(summary_global, SAVE_GLOBAL)


summary_by_env <- lapply(traits_found, function(tr) {
  df_sub <- pheno[!is.na(pheno[[tr]]), ]

  if (nrow(df_sub) == 0) {
    return(NULL)
  }

  df_sub %>%
    group_by(Envir) %>%
    summarise(
      non_missing_obs = n(),
      unique_genotypes = n_distinct(Genotype),
      genotypes_matching_genomic = sum(unique(Genotype) %in% geno_ids),
      mean_value = mean(.data[[tr]], na.rm = TRUE),
      median_value = median(.data[[tr]], na.rm = TRUE),
      min_value = min(.data[[tr]], na.rm = TRUE),
      max_value = max(.data[[tr]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(Trait = tr) %>%
    select(Trait, Envir, everything())
})

summary_by_env <- bind_rows(summary_by_env)
summary_by_env <- summary_by_env %>%
  arrange(Trait, Envir)

write_csv(summary_by_env, SAVE_BY_ENV)


match_rows <- lapply(traits_found, function(tr) {
  df_sub <- pheno[!is.na(pheno[[tr]]), ]
  trait_genotypes <- sort(unique(df_sub$Genotype))

  data.frame(
    Trait = tr,
    Genotype = trait_genotypes,
    in_genomic_member = trait_genotypes %in% geno_ids,
    stringsAsFactors = FALSE
  )
})

match_df <- bind_rows(match_rows)
write_csv(match_df, SAVE_MATCH)


sink(SAVE_REPORT)

cat("================================================================================\n")
cat("P0 - NEW TRAITS AUDIT REPORT\n")
cat("================================================================================\n\n")

cat("TRAITS REQUESTED\n")
print(TRAITS)
cat("\n")

cat("TRAITS FOUND\n")
print(traits_found)
cat("\n")

cat("TRAITS MISSING\n")
print(traits_missing)
cat("\n\n")

cat("INPUT FILES\n")
cat("PHENO_FILE:", PHENO_FILE, "\n")
cat("BED_FILE:", BED_FILE, "\n")
cat("BIM_FILE:", BIM_FILE, "\n")
cat("FAM_FILE:", FAM_FILE, "\n\n")

cat("BASIC DATASET DIMENSIONS\n")
cat("Pheno rows:", nrow(pheno), "\n")
cat("Pheno columns:", ncol(pheno), "\n")
cat("Unique pheno genotypes:", length(unique(pheno$Genotype)), "\n")
cat("Unique pheno environments:", length(unique(pheno$Envir)), "\n")
cat("Unique genomic IDs:", length(unique(geno_ids)), "\n\n")

cat("PHENO COLUMNS\n")
print(colnames(pheno))
cat("\n\n")

cat("GLOBAL SUMMARY\n")
print(summary_global)
cat("\n\n")

cat("SUMMARY BY ENVIRONMENT\n")
print(summary_by_env)
cat("\n\n")

cat("GENOTYPE MATCH SUMMARY\n")
print(
  match_df %>%
    group_by(Trait, in_genomic_member) %>%
    summarise(n_genotypes = n(), .groups = "drop")
)
cat("\n\n")

cat("GENOTYPES WITHOUT GENOMIC MATCH BY TRAIT\n")
for (tr in traits_found) {
  cat("\n---", tr, "---\n")
  tmp <- match_df %>%
    filter(Trait == tr, in_genomic_member == FALSE)

  if (nrow(tmp) == 0) {
    cat("All trait genotypes match genomic IDs.\n")
  } else {
    print(tmp$Genotype)
  }
}

sink()


cat("File salvati:\n")
cat("-", SAVE_GLOBAL, "\n")
cat("-", SAVE_BY_ENV, "\n")
cat("-", SAVE_MATCH, "\n")
cat("-", SAVE_REPORT, "\n\n")

cat("Preview global summary:\n")
print(summary_global)

cat("\nP0 completato.\n")
