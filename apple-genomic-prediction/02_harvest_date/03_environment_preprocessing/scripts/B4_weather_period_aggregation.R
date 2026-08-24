# ==================================================
# B4_weather_period_aggregation.R
# Aggregazione weather per environment e periodo
# ==================================================

rm(list = ls())

cat("=== STEP B4: WEATHER PERIOD AGGREGATION ===\n\n")

library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
weather_daily_file <- "Output/Weather_daily.csv"
periods_file <- "Output/environment_periods_P1_P2.csv"

# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento Weather_daily...\n")
weather_daily <- read_csv(weather_daily_file, show_col_types = FALSE)

cat("Caricamento environment periods...\n")
periods <- read_csv(periods_file, show_col_types = FALSE)

cat("Caricamento completato.\n\n")

# -------------------------------
# 3. Preparazione weather
# -------------------------------
weather_daily$Day <- as.Date(weather_daily$Day)

# crea environment da Location + anno del giorno
weather_daily$Year <- format(weather_daily$Day, "%Y")
weather_daily$Envir <- paste0(weather_daily$Location, ".", weather_daily$Year)

# -------------------------------
# 4. Preparazione periods
# -------------------------------
periods$P1_start <- as.Date(periods$P1_start)
periods$P1_end   <- as.Date(periods$P1_end)
periods$P2_start <- as.Date(periods$P2_start)
periods$P2_end   <- as.Date(periods$P2_end)

# teniamo solo colonne utili
periods_sub <- periods %>%
  select(Envir, P1_start, P1_end, P2_start, P2_end)

# -------------------------------
# 5. Merge weather + periods
# -------------------------------
cat("Merge tra weather daily e periodi...\n")
w <- left_join(weather_daily, periods_sub, by = "Envir")

cat("Numero righe dopo merge:", nrow(w), "\n\n")

# -------------------------------
# 6. Assegna periodo
# -------------------------------
w$Period <- NA_character_

w$Period[w$Day >= w$P1_start & w$Day <= w$P1_end] <- "P1"
w$Period[w$Day >= w$P2_start & w$Day <= w$P2_end] <- "P2"

# tieni solo giorni assegnati a un periodo
w_period <- w %>%
  filter(!is.na(Period))

cat("Numero righe con periodo assegnato:", nrow(w_period), "\n\n")

# -------------------------------
# 7. Aggregazione per Envir e Period
# -------------------------------
cat("Aggregazione weather per environment e periodo...\n")

weather_period_long <- w_period %>%
  group_by(Envir, Period) %>%
  summarise(
    Temperature_Dmean_sum = sum(Temperature_Dmean, na.rm = TRUE),
    Humidity_Dmean_sum    = sum(Humidity_Dmean, na.rm = TRUE),
    Radiation_Dsum_sum    = sum(Radiation_Dsum, na.rm = TRUE),
    .groups = "drop"
  )

cat("Aggregazione completata.\n\n")

# -------------------------------
# 8. Passaggio a wide format
# -------------------------------
p1 <- weather_period_long %>%
  filter(Period == "P1") %>%
  select(-Period) %>%
  rename(
    Temperature_Dmean_P1sum = Temperature_Dmean_sum,
    Humidity_Dmean_P1sum    = Humidity_Dmean_sum,
    Radiation_Dsum_P1sum    = Radiation_Dsum_sum
  )

p2 <- weather_period_long %>%
  filter(Period == "P2") %>%
  select(-Period) %>%
  rename(
    Temperature_Dmean_P2sum = Temperature_Dmean_sum,
    Humidity_Dmean_P2sum    = Humidity_Dmean_sum,
    Radiation_Dsum_P2sum    = Radiation_Dsum_sum
  )

weather_period_wide <- full_join(p1, p2, by = "Envir") %>%
  arrange(Envir)

# -------------------------------
# 9. Controlli output
# -------------------------------
cat("Numero environment finali:", nrow(weather_period_wide), "\n")
cat("Prime righe output wide:\n")
print(head(weather_period_wide))
cat("\n")

cat("Missing nel file wide:\n")
print(colSums(is.na(weather_period_wide)))
cat("\n")

# -------------------------------
# 10. Salvataggio output
# -------------------------------
dir.create("Output", showWarnings = FALSE)

write_csv(weather_period_long, "Output/weather_period_aggregation_long.csv")
write_csv(weather_period_wide, "Output/weather_period_aggregation_wide.csv")

sink("Output/weather_period_aggregation_report.txt")

cat("=== STEP B4: WEATHER PERIOD AGGREGATION REPORT ===\n\n")
cat("Numero righe weather daily:", nrow(weather_daily), "\n")
cat("Numero righe con periodo assegnato:", nrow(w_period), "\n")
cat("Numero environment finali:", nrow(weather_period_wide), "\n\n")

cat("Output long:\n")
print(weather_period_long)
cat("\n\n")

cat("Output wide:\n")
print(weather_period_wide)
cat("\n\n")

cat("Missing nel file wide:\n")
print(colSums(is.na(weather_period_wide)))
cat("\n")

sink()

# -------------------------------
# 11. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- Output/weather_period_aggregation_long.csv\n")
cat("- Output/weather_period_aggregation_wide.csv\n")
cat("- Output/weather_period_aggregation_report.txt\n")