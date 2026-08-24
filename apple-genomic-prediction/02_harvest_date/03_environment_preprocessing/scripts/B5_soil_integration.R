# ==================================================
# B5_soil_integration.R
# Trasformazione e integrazione del suolo per Envir
# ==================================================

rm(list = ls())

cat("=== STEP B5: SOIL INTEGRATION ===\n\n")

library(readxl)
library(readr)
library(dplyr)
library(tidyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
soil_file <- "Input/Soil_raw.xlsx"
periods_file <- "Output/environment_periods_P1_P2.csv"

# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento Soil_raw...\n")
soil <- as.data.frame(read_xlsx(soil_file))

cat("Caricamento environment periods...\n")
periods <- read_csv(periods_file, show_col_types = FALSE)

cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Controlli base
# -------------------------------
cat("Numero righe soil raw:", nrow(soil), "\n")
cat("Prime righe soil:\n")
print(head(soil))
cat("\n")

# -------------------------------
# 4. Soil: formato wide per location
# -------------------------------
cat("Trasformazione soil da long a wide...\n")

soil_wide_location <- soil %>%
  select(Group.1, Variable, x) %>%
  rename(Location = Group.1,
         Value = x) %>%
  pivot_wider(names_from = Variable, values_from = Value)

cat("Trasformazione completata.\n\n")

cat("Numero righe soil wide per location:", nrow(soil_wide_location), "\n")
cat("Prime righe soil wide:\n")
print(head(soil_wide_location))
cat("\n")

# -------------------------------
# 5. Ricavo mappa Envir -> Location
# -------------------------------
periods$Location <- sub("\\..*$", "", periods$Envir)

env_location <- periods %>%
  select(Envir, Location) %>%
  distinct()

# -------------------------------
# 6. Replica soil per environment
# -------------------------------
cat("Replica delle covariate di suolo per environment...\n")

soil_by_envir <- env_location %>%
  left_join(soil_wide_location, by = "Location") %>%
  select(-Location) %>%
  arrange(Envir)

cat("Replica completata.\n\n")

# -------------------------------
# 7. Controlli output
# -------------------------------
cat("Numero environment finali:", nrow(soil_by_envir), "\n")
cat("Prime righe soil_by_envir:\n")
print(head(soil_by_envir))
cat("\n")

cat("Missing nel file soil_by_envir:\n")
print(colSums(is.na(soil_by_envir)))
cat("\n")

# -------------------------------
# 8. Salvataggio output
# -------------------------------
dir.create("Output", showWarnings = FALSE)

write_csv(soil_wide_location, "Output/soil_wide_location.csv")
write_csv(soil_by_envir, "Output/soil_by_envir.csv")

sink("Output/soil_integration_report.txt")

cat("=== STEP B5: SOIL INTEGRATION REPORT ===\n\n")
cat("Numero righe soil raw:", nrow(soil), "\n")
cat("Numero righe soil wide per location:", nrow(soil_wide_location), "\n")
cat("Numero environment finali:", nrow(soil_by_envir), "\n\n")

cat("Soil wide per location:\n")
print(soil_wide_location)
cat("\n\n")

cat("Soil by environment:\n")
print(soil_by_envir)
cat("\n\n")

cat("Missing nel file soil_by_envir:\n")
print(colSums(is.na(soil_by_envir)))
cat("\n")

sink()

# -------------------------------
# 9. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- Output/soil_wide_location.csv\n")
cat("- Output/soil_by_envir.csv\n")
cat("- Output/soil_integration_report.txt\n")