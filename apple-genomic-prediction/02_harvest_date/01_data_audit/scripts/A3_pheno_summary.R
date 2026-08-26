# ===============================
# A3_pheno_summary.R
# Phenotype summary for trait and environment
# ===============================

rm(list = ls())

cat("=== STEP A3: PHENOTYPE SUMMARY ===\n\n")

library(readxl)
library(dplyr)

# -------------------------------
# 1. File paths
# -------------------------------
pheno_file <- "data/raw/phenotype/Pheno_raw.xlsx"

outdir <- "02_harvest_date/01_data_audit/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
# -------------------------------
# 2. Loading data
# -------------------------------
cat("Caricamento file fenotipico...\n")
pheno <- as.data.frame(read_xlsx(pheno_file))
cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Definition of the trait I want to analyze
# -------------------------------
trait_cols <- c(
  "Harvest_date",
  "Fruit_weight",
  "Fruit_number",
  "Fruit_weight_single",
  "Color_over",
  "Russet_freq_all",
  "Flowering_intensity",
  "Flowering_begin",
  "Acidity",
  "Sugar",
  "Firmness"
)

# Check to see if they actually exist
trait_cols <- trait_cols[trait_cols %in% colnames(pheno)]

cat("Trait trovati nel file:\n")
print(trait_cols)
cat("\n")

# -------------------------------
# 4. Overall Summary by Trait
# -------------------------------
cat("Calcolo riepilogo globale per trait...\n")

summary_global <- lapply(trait_cols, function(tr) {
  df_sub <- pheno[!is.na(pheno[[tr]]), ] # prendo solo le righe dove quel trait non è missing, cioè tengo solo le osservazioni in cui Harvest_date ha un valore
  
  data.frame(
    Trait = tr,
    non_missing_obs = nrow(df_sub),
    unique_genotypes = length(unique(df_sub$Genotype)),
    unique_envir = length(unique(df_sub$Envir)),
    stringsAsFactors = FALSE
  )
}) # non_missing_obs refers to the actual number of rows with available values; unique_genotypes is the number of distinct genotypes that appear among these rows; unique_envir indicates the number of environments in which that trait appears

summary_global <- bind_rows(summary_global)
summary_global <- summary_global[order(-summary_global$non_missing_obs), ]

cat("Riepilogo globale completato.\n\n")

# -------------------------------
# 5. Summary by trait × environment
# -------------------------------
cat("Calcolo riepilogo per trait x environment...\n") # For each trait, I take the non-missing rows, group them by Envir, and then calculate non_missing_obs (the number of observations in that environment for that trait) and unique_genotypes (the number of distinct genotypes in that environment for that trait)

summary_by_env <- lapply(trait_cols, function(tr) {
  df_sub <- pheno[!is.na(pheno[[tr]]), ]
  
  if (nrow(df_sub) == 0) return(NULL)
  
  df_grouped <- df_sub %>%
    group_by(Envir) %>%
    summarise(
      non_missing_obs = n(),
      unique_genotypes = n_distinct(Genotype),
      .groups = "drop"
    )
  
  df_grouped$Trait <- tr
  df_grouped <- df_grouped[, c("Trait", "Envir", "non_missing_obs", "unique_genotypes")]
  
  return(df_grouped)
})

summary_by_env <- bind_rows(summary_by_env)
summary_by_env <- summary_by_env[order(summary_by_env$Trait, summary_by_env$Envir), ]

cat("Riepilogo per trait x environment completato.\n\n")

# -------------------------------
# 6. Save output
# -------------------------------
write.csv(
  summary_global,
  file.path(outdir, "pheno_summary_global.csv"),
  row.names = FALSE
)

write.csv(
  summary_by_env,
  file.path(outdir, "pheno_summary_by_environment.csv"),
  row.names = FALSE
)
# -------------------------------
# 7. Report
# -------------------------------
sink(file.path(outdir, "pheno_summary_report.txt"))

cat("=== STEP A3: PHENOTYPE SUMMARY REPORT ===\n\n")

cat("RIEPILOGO GLOBALE PER TRAIT\n")
print(summary_global)
cat("\n\n")

cat("RIEPILOGO PER TRAIT x ENVIRONMENT\n")
print(summary_by_env)
cat("\n")

sink()

# -------------------------------
# 8. Final print
# -------------------------------
cat("File salvati:\n")
cat("- ", file.path(outdir, "pheno_summary_global.csv"), "\n")
cat("- ", file.path(outdir, "pheno_summary_by_environment.csv"), "\n")
cat("- ", file.path(outdir, "pheno_summary_report.txt"), "\n")
