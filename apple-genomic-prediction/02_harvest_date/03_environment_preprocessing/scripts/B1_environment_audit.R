# ==================================================
# B1_environment_audit.R
# Audit iniziale dei file ambientali
# ==================================================

rm(list = ls())

cat("=== STEP B1: ENVIRONMENT AUDIT ===\n\n")

library(readxl)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
weather_file <- "data/raw/environment/Weather_raw.xlsx"
soil_file <- "data/raw/environment/Soil_raw.xlsx"

outdir <- "02_harvest_date/03_environment_preprocessing/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# -------------------------------
# 2. Controllo esistenza file
# -------------------------------
cat("Controllo file presenti:\n")
cat("Weather_raw.xlsx ->", file.exists(weather_file), "\n")
cat("Soil_raw.xlsx    ->", file.exists(soil_file), "\n\n")

# -------------------------------
# 3. Caricamento dati
# -------------------------------
cat("Caricamento weather...\n")
weather <- as.data.frame(read_xlsx(weather_file))

cat("Caricamento soil...\n")
soil <- as.data.frame(read_xlsx(soil_file))

cat("Caricamento completato.\n\n")

# -------------------------------
# 4. Dimensioni dataset
# -------------------------------
cat("=== DIMENSIONI DATASET ===\n")
cat("Weather:", nrow(weather), "righe x", ncol(weather), "colonne\n")
cat("Soil   :", nrow(soil), "righe x", ncol(soil), "colonne\n\n")

# -------------------------------
# 5. Colonne
# -------------------------------
cat("=== COLONNE WEATHER ===\n")
print(colnames(weather))
cat("\n")

cat("=== COLONNE SOIL ===\n")
print(colnames(soil))
cat("\n")

# -------------------------------
# 6. Prime righe
# -------------------------------
cat("=== PRIME RIGHE WEATHER ===\n")
print(head(weather))
cat("\n")

cat("=== PRIME RIGHE SOIL ===\n")
print(head(soil))
cat("\n")

# -------------------------------
# 7. Riassunto WEATHER
# -------------------------------
cat("=== WEATHER SUMMARY ===\n")

if ("Location" %in% colnames(weather)) {
  cat("Location uniche:\n")
  print(sort(unique(weather$Location)))
  cat("\n")
  
  weather_by_loc <- weather %>%
    group_by(Location) %>%
    summarise(n_obs = n(), .groups = "drop")
  
  cat("Numero osservazioni per location:\n")
  print(weather_by_loc)
  cat("\n")
}

if ("Date" %in% colnames(weather)) {
  weather$Date <- as.POSIXct(weather$Date, origin = "1970-01-01", tz = "UTC")
  
  cat("Data minima weather:", as.character(min(weather$Date, na.rm = TRUE)), "\n")
  cat("Data massima weather:", as.character(max(weather$Date, na.rm = TRUE)), "\n\n")
}

cat("Missing values WEATHER:\n")
print(colSums(is.na(weather)))
cat("\n")

# -------------------------------
# 8. Riassunto SOIL
# -------------------------------
cat("=== SOIL SUMMARY ===\n")

if ("Group.1" %in% colnames(soil)) {
  cat("Valori unici Group.1:\n")
  print(sort(unique(soil$Group.1)))
  cat("\n")
}

if ("Group.2" %in% colnames(soil)) {
  cat("Valori unici Group.2:\n")
  print(sort(unique(soil$Group.2)))
  cat("\n")
}

if ("Variable" %in% colnames(soil)) {
  cat("Variabili uniche nel file soil:\n")
  print(sort(unique(soil$Variable)))
  cat("\n")
}

cat("Missing values SOIL:\n")
print(colSums(is.na(soil)))
cat("\n")

# -------------------------------
# 9. Salvataggio report
# -------------------------------
sink(file.path(outdir, "environment_audit_report.txt"))

cat("=== STEP B1: ENVIRONMENT AUDIT REPORT ===\n\n")

cat("DIMENSIONI DATASET\n")
cat("Weather:", nrow(weather), "x", ncol(weather), "\n")
cat("Soil:", nrow(soil), "x", ncol(soil), "\n\n")

cat("COLONNE WEATHER\n")
print(colnames(weather))
cat("\n")

cat("COLONNE SOIL\n")
print(colnames(soil))
cat("\n")

cat("PRIME RIGHE WEATHER\n")
print(head(weather))
cat("\n")

cat("PRIME RIGHE SOIL\n")
print(head(soil))
cat("\n")

if ("Location" %in% colnames(weather)) {
  cat("LOCATION WEATHER\n")
  print(sort(unique(weather$Location)))
  cat("\n")
  
  cat("OSSERVAZIONI WEATHER PER LOCATION\n")
  print(weather_by_loc)
  cat("\n")
}

if ("Date" %in% colnames(weather)) {
  cat("RANGE DATE WEATHER\n")
  cat("Min:", as.character(min(weather$Date, na.rm = TRUE)), "\n")
  cat("Max:", as.character(max(weather$Date, na.rm = TRUE)), "\n\n")
}

cat("MISSING WEATHER\n")
print(colSums(is.na(weather)))
cat("\n")

if ("Group.1" %in% colnames(soil)) {
  cat("GROUP.1 SOIL\n")
  print(sort(unique(soil$Group.1)))
  cat("\n")
}

if ("Group.2" %in% colnames(soil)) {
  cat("GROUP.2 SOIL\n")
  print(sort(unique(soil$Group.2)))
  cat("\n")
}

if ("Variable" %in% colnames(soil)) {
  cat("VARIABLE SOIL\n")
  print(sort(unique(soil$Variable)))
  cat("\n")
}

cat("MISSING SOIL\n")
print(colSums(is.na(soil)))
cat("\n")

sink()

# -------------------------------
# 10. Stampa finale
# -------------------------------
cat("File salvato:\n")
cat("- ", file.path(outdir, "environment_audit_report.txt"), "\n")
