# ===============================
# A3_pheno_summary.R
# Riassunto del fenotipo per trait e per environment
# ===============================

rm(list = ls())

cat("=== STEP A3: PHENOTYPE SUMMARY ===\n\n")

library(readxl)
library(dplyr)

# -------------------------------
# 1. Percorso file
# -------------------------------
pheno_file <- "Input/Pheno_raw.xlsx" # carico il file fenotipico

# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento file fenotipico...\n")
pheno <- as.data.frame(read_xlsx(pheno_file))
cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Definizione trait che voglio analizzare
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

# controllo che esistano davvero
trait_cols <- trait_cols[trait_cols %in% colnames(pheno)]

cat("Trait trovati nel file:\n")
print(trait_cols)
cat("\n")

# -------------------------------
# 4. Riepilogo globale per trait
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
}) # non_missing_obs sono le righe che ci sono davvero con valore disponibile; unique_genotypes è il numero di genotipi diversi che compaiono tra queste righe; unique_envir è in quanti environment compare quel trait

summary_global <- bind_rows(summary_global)
summary_global <- summary_global[order(-summary_global$non_missing_obs), ]

cat("Riepilogo globale completato.\n\n")

# -------------------------------
# 5. Riepilogo per trait x environment
# -------------------------------
cat("Calcolo riepilogo per trait x environment...\n") # per ogni trait prendo le righe non-missing, le raggruppo per Envir e poi calcolo non_missing_obs (qante osservazioni ci sono in quell'ambiente per quel trait) e unique_genotypes (quanti genotipi diversi ci sono in quell'ambiente per quel trait)

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
# 6. Salvataggio output
# -------------------------------
dir.create("Output", showWarnings = FALSE)

write.csv(summary_global, "Output/pheno_summary_global.csv", row.names = FALSE)
write.csv(summary_by_env, "Output/pheno_summary_by_environment.csv", row.names = FALSE)

# -------------------------------
# 7. Report testuale
# -------------------------------
sink("Output/pheno_summary_report.txt")

cat("=== STEP A3: PHENOTYPE SUMMARY REPORT ===\n\n")

cat("RIEPILOGO GLOBALE PER TRAIT\n")
print(summary_global)
cat("\n\n")

cat("RIEPILOGO PER TRAIT x ENVIRONMENT\n")
print(summary_by_env)
cat("\n")

sink()

# -------------------------------
# 8. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- Output/pheno_summary_global.csv\n")
cat("- Output/pheno_summary_by_environment.csv\n")
cat("- Output/pheno_summary_report.txt\n")