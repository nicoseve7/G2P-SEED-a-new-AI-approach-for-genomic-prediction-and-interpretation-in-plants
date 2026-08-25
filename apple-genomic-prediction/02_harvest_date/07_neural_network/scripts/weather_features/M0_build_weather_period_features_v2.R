# Questo usa:

# Input/base_files/Weather_daily.csv
# Input/base_files/environment_periods_P1_P2.csv

# e produce feature per:

# P1
# P2

# con statistiche:

# sum
# mean
# sd
# min
# max

# ==================================================
# M0_build_weather_period_features_v2.R
# Costruzione feature meteo P1/P2 più ricche
# ==================================================

rm(list = ls())

cat("=== M0: BUILD WEATHER PERIOD FEATURES V2 ===\n\n")

library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
weather_daily_file <- "Input/base_files/Weather_daily.csv"
periods_file       <- "Input/base_files/environment_periods_P1_P2.csv"

out_csv_file    <- "Input/derived/weather_period_features_v2.csv"
out_rdata_file  <- "Input/derived/weather_period_features_v2.RData"
report_file     <- "Output/reports/weather_period_features_v2_report.txt"

dir.create("Input/derived", recursive = TRUE, showWarnings = FALSE)
dir.create("Output/reports", recursive = TRUE, showWarnings = FALSE)

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
weather_daily$Year <- format(weather_daily$Day, "%Y")
weather_daily$Envir <- paste0(weather_daily$Location, ".", weather_daily$Year)

# -------------------------------
# 4. Preparazione periods
# -------------------------------
periods$P1_start <- as.Date(periods$P1_start)
periods$P1_end   <- as.Date(periods$P1_end)
periods$P2_start <- as.Date(periods$P2_start)
periods$P2_end   <- as.Date(periods$P2_end)

periods_sub <- periods %>%
  select(Envir, P1_start, P1_end, P2_start, P2_end)

# -------------------------------
# 5. Merge weather + periods
# -------------------------------
w <- left_join(weather_daily, periods_sub, by = "Envir")

# -------------------------------
# 6. Assegna periodo
# -------------------------------
w$Period <- NA_character_

w$Period[w$Day >= w$P1_start & w$Day <= w$P1_end] <- "P1"
w$Period[w$Day >= w$P2_start & w$Day <= w$P2_end] <- "P2"

w_period <- w %>% filter(!is.na(Period))

cat("Numero righe con periodo assegnato:", nrow(w_period), "\n\n")

# -------------------------------
# 7. Aggregazione ricca per Envir e Period
# -------------------------------
weather_period_long_v2 <- w_period %>%
  group_by(Envir, Period) %>%
  summarise(
    n_days = n(),

    Temperature_Dmean_sum  = sum(Temperature_Dmean, na.rm = TRUE),
    Temperature_Dmean_mean = mean(Temperature_Dmean, na.rm = TRUE),
    Temperature_Dmean_sd   = sd(Temperature_Dmean, na.rm = TRUE),
    Temperature_Dmean_min  = min(Temperature_Dmean, na.rm = TRUE),
    Temperature_Dmean_max  = max(Temperature_Dmean, na.rm = TRUE),

    Humidity_Dmean_sum  = sum(Humidity_Dmean, na.rm = TRUE),
    Humidity_Dmean_mean = mean(Humidity_Dmean, na.rm = TRUE),
    Humidity_Dmean_sd   = sd(Humidity_Dmean, na.rm = TRUE),
    Humidity_Dmean_min  = min(Humidity_Dmean, na.rm = TRUE),
    Humidity_Dmean_max  = max(Humidity_Dmean, na.rm = TRUE),

    Radiation_Dsum_sum  = sum(Radiation_Dsum, na.rm = TRUE),
    Radiation_Dsum_mean = mean(Radiation_Dsum, na.rm = TRUE),
    Radiation_Dsum_sd   = sd(Radiation_Dsum, na.rm = TRUE),
    Radiation_Dsum_min  = min(Radiation_Dsum, na.rm = TRUE),
    Radiation_Dsum_max  = max(Radiation_Dsum, na.rm = TRUE),

    .groups = "drop"
  )

# -------------------------------
# 8. Wide format
# -------------------------------
p1 <- weather_period_long_v2 %>%
  filter(Period == "P1") %>%
  select(-Period)

colnames(p1)[-1] <- paste0(colnames(p1)[-1], "_P1")

p2 <- weather_period_long_v2 %>%
  filter(Period == "P2") %>%
  select(-Period)

colnames(p2)[-1] <- paste0(colnames(p2)[-1], "_P2")

weather_period_wide_v2 <- full_join(p1, p2, by = "Envir") %>%
  arrange(Envir)

# -------------------------------
# 9. Salvataggio
# -------------------------------
write_csv(weather_period_wide_v2, out_csv_file)

weather_period_matrix_v2 <- as.data.frame(weather_period_wide_v2)
rownames(weather_period_matrix_v2) <- weather_period_matrix_v2$Envir
weather_period_matrix_v2$Envir <- NULL
weather_period_matrix_v2 <- as.matrix(weather_period_matrix_v2)

save(weather_period_matrix_v2, file = out_rdata_file)

# -------------------------------
# 10. Report
# -------------------------------
sink(report_file)

cat("=== M0: WEATHER PERIOD FEATURES V2 REPORT ===\n\n")
cat("Numero righe weather_daily:", nrow(weather_daily), "\n")
cat("Numero righe con periodo assegnato:", nrow(w_period), "\n")
cat("Numero environment finali:", nrow(weather_period_wide_v2), "\n")
cat("Numero colonne finali:", ncol(weather_period_wide_v2), "\n\n")

cat("Prime righe output:\n")
print(head(weather_period_wide_v2))
cat("\n\n")

cat("Missing per colonna:\n")
print(colSums(is.na(weather_period_wide_v2)))
cat("\n")

sink()

cat("File salvati:\n")
cat("-", out_csv_file, "\n")
cat("-", out_rdata_file, "\n")
cat("-", report_file, "\n")