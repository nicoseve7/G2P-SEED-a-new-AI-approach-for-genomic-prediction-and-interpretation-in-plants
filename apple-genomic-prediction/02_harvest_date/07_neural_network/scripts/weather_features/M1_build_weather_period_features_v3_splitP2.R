# Qui facciamo:

# P1 invariato
# P2 divisa in:
# P2a
# P2b

# e per ciascun periodo calcoliamo:

# sum
# mean
# sd
# min
# max

# ==================================================
# M1_build_weather_period_features_v3_splitP2.R
# Costruzione feature meteo con P1 + P2a + P2b
# ==================================================

rm(list = ls())

cat("=== M1: BUILD WEATHER PERIOD FEATURES V3 (split P2) ===\n\n")

library(readr)
library(dplyr)

# -------------------------------
# 1. Percorsi file
# -------------------------------
weather_daily_file <- paste0(
  "02_harvest_date/03_environment_preprocessing/output/",
  "Weather_daily.csv"
)

periods_file <- paste0(
  "02_harvest_date/03_environment_preprocessing/output/",
  "environment_periods_P1_P2.csv"
)

outdir <- "02_harvest_date/07_neural_network/output/weather_features"
report_dir <- file.path(outdir, "reports")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

out_csv_file <- file.path(
  outdir,
  "weather_period_features_v3_splitP2.csv"
)

out_rdata_file <- file.path(
  outdir,
  "weather_period_features_v3_splitP2.RData"
)

report_file <- file.path(
  report_dir,
  "weather_period_features_v3_splitP2_report.txt"
)
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

# midpoint di P2
periods$P2_mid <- periods$P2_start + floor(as.numeric(periods$P2_end - periods$P2_start) / 2)

periods_sub <- periods %>%
  select(Envir, P1_start, P1_end, P2_start, P2_mid, P2_end)

# -------------------------------
# 5. Merge weather + periods
# -------------------------------
w <- left_join(weather_daily, periods_sub, by = "Envir")

# -------------------------------
# 6. Assegna periodo
# -------------------------------
w$Period <- NA_character_

w$Period[w$Day >= w$P1_start & w$Day <= w$P1_end] <- "P1"
w$Period[w$Day >= w$P2_start & w$Day <= w$P2_mid] <- "P2a"
w$Period[w$Day >  w$P2_mid   & w$Day <= w$P2_end] <- "P2b"

w_period <- w %>% filter(!is.na(Period))

cat("Numero righe con periodo assegnato:", nrow(w_period), "\n\n")

# -------------------------------
# 7. Aggregazione ricca per Envir e Period
# -------------------------------
weather_period_long_v3 <- w_period %>%
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
make_period_df <- function(df, period_name) {
  x <- df %>%
    filter(Period == period_name) %>%
    select(-Period)

  colnames(x)[-1] <- paste0(colnames(x)[-1], "_", period_name)
  x
}

p1  <- make_period_df(weather_period_long_v3, "P1")
p2a <- make_period_df(weather_period_long_v3, "P2a")
p2b <- make_period_df(weather_period_long_v3, "P2b")

weather_period_wide_v3 <- full_join(p1, p2a, by = "Envir") %>%
  full_join(p2b, by = "Envir") %>%
  arrange(Envir)

# -------------------------------
# 9. Salvataggio
# -------------------------------
write_csv(weather_period_wide_v3, out_csv_file)

weather_period_matrix_v3 <- as.data.frame(weather_period_wide_v3)
rownames(weather_period_matrix_v3) <- weather_period_matrix_v3$Envir
weather_period_matrix_v3$Envir <- NULL
weather_period_matrix_v3 <- as.matrix(weather_period_matrix_v3)

save(weather_period_matrix_v3, file = out_rdata_file)

# -------------------------------
# 10. Report
# -------------------------------
sink(report_file)

cat("=== M1: WEATHER PERIOD FEATURES V3 (split P2) REPORT ===\n\n")
cat("Numero righe weather_daily:", nrow(weather_daily), "\n")
cat("Numero righe con periodo assegnato:", nrow(w_period), "\n")
cat("Numero environment finali:", nrow(weather_period_wide_v3), "\n")
cat("Numero colonne finali:", ncol(weather_period_wide_v3), "\n\n")

cat("Prime righe output:\n")
print(head(weather_period_wide_v3))
cat("\n\n")

cat("Missing per colonna:\n")
print(colSums(is.na(weather_period_wide_v3)))
cat("\n")

sink()

cat("File salvati:\n")
cat("-", out_csv_file, "\n")
cat("-", out_rdata_file, "\n")
cat("-", report_file, "\n")
