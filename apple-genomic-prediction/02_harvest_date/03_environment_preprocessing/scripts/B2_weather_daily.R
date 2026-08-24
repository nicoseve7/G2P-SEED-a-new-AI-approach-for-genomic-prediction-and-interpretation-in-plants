# ==================================================
# B2_weather_daily.R
# Costruzione del weather giornaliero da hourly data
# ==================================================

rm(list = ls())

cat("=== STEP B2: WEATHER DAILY ===\n\n")

library(readxl)
library(dplyr)
library(readr)

# -------------------------------
# 1. Percorso file
# -------------------------------
weather_file <- "data/raw/environment/Weather_raw.xlsx"

outdir <- "02_harvest_date/03_environment_preprocessing/output"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
# -------------------------------
# 2. Caricamento dati (Carica il file weather hourly)
# -------------------------------
cat("Caricamento weather raw...\n")
weather <- as.data.frame(read_xlsx(weather_file))
cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Controlli base
# -------------------------------
cat("Numero righe iniziali:", nrow(weather), "\n")
cat("Numero location uniche:", length(unique(weather$Location)), "\n\n")

# -------------------------------
# 4. Assicura formato datetime
# -------------------------------
weather$Date <- as.POSIXct(weather$Date, tz = "UTC")

# crea la variabile giorno (senza ora). Prende la colonna Date e ricava il solo giorno:
weather$Day <- as.Date(weather$Date)

# -------------------------------
# 5. Aggregazione giornaliera. Raggruppa per: Location e Day
# -------------------------------
cat("Aggregazione a livello giornaliero...\n")

d1 <- weather %>%
  group_by(Location, Day) %>%
  summarise(
    Temperature_Dmean = mean(Temperature, na.rm = TRUE),
    Humidity_Dmean    = mean(Humidity, na.rm = TRUE),
    Radiation_Dsum    = sum(Radiation, na.rm = TRUE),
    .groups = "drop"
  ) # Calcola media giornaliera della temperatura, media giornaliera dell’umidità, somma giornaliera della radiazione. Usiamo na.rm = TRUE per ignorare i missing e usare i dati disponibili
# radiazione faccio somma giornaliera
cat("Aggregazione completata.\n\n")

# -------------------------------
# 6. Controlli output
# -------------------------------
cat("Numero righe finali:", nrow(d1), "\n")
cat("Prime righe:\n")
print(head(d1))
cat("\n")

cat("Range date giornaliere:\n")
cat("Min:", as.character(min(d1$Day, na.rm = TRUE)), "\n")
cat("Max:", as.character(max(d1$Day, na.rm = TRUE)), "\n\n")

cat("Missing nel daily weather:\n")
print(colSums(is.na(d1)))
cat("\n")

# -------------------------------
# 7. Salvataggio output
# -------------------------------
write_csv(
  d1,
  file.path(outdir, "Weather_daily.csv")
)

sink(file.path(outdir, "weather_daily_report.txt"))

cat("=== STEP B2: WEATHER DAILY REPORT ===\n\n")
cat("Numero righe iniziali:", nrow(weather), "\n")
cat("Numero righe finali:", nrow(d1), "\n\n")

cat("Prime righe:\n")
print(head(d1))
cat("\n")

cat("Range date giornaliere:\n")
cat("Min:", as.character(min(d1$Day, na.rm = TRUE)), "\n")
cat("Max:", as.character(max(d1$Day, na.rm = TRUE)), "\n\n")

cat("Missing nel daily weather:\n")
print(colSums(is.na(d1)))
cat("\n")

sink()

# -------------------------------
# 8. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- ", file.path(outdir, "Weather_daily.csv"), "\n")
cat("- ", file.path(outdir, "weather_daily_report.txt"), "\n")
