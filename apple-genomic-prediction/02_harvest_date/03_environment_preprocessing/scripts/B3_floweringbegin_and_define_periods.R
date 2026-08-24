# ==========================================================
# B3_floweringbegin_and_define_periods.R
# Recupero Flowering_begin + definizione periodi P1/P2
# ==========================================================

rm(list = ls())

cat("=== STEP B3: FLOWERING_BEGIN + DEFINE PERIODS P1/P2 ===\n\n")

library(readxl)
library(readr)
library(dplyr)
library(snpStats)
library(lme4)
library(SpATS)
library(multtest)

# ----------------------------------------------------------
# Funzione outliers paper-inspired (BH-MADR)
# ----------------------------------------------------------
outliers_bhmadr <- function(mydata, mytrait, mylm) {
  resi <- cbind(residuals(mylm, type = "response"))

  med <- median(resi, na.rm = TRUE)
  MAD <- median(abs(resi - med), na.rm = TRUE)
  re_MAD <- MAD * 1.4826

  if (is.na(re_MAD) || re_MAD == 0) {
    return(mydata[0, , drop = FALSE])
  }

  res_MAD <- resi / re_MAD
  rawp <- 2 * (1 - pnorm(abs(res_MAD)))

  dataset_name_1 <- subset(mydata, !is.na(mydata[, mytrait]))
  rawp2 <- cbind(dataset_name_1, resi, res_MAD, rawp)

  res2 <- mt.rawp2adjp(rawp, proc = c("Holm"))
  adjp  <- cbind(res2[[1]][, 1])
  bholm <- cbind(res2[[1]][, 2])
  index <- cbind(res2[[2]])

  out_flag <- ifelse(bholm < 0.05, "OUTLIER", ".")
  bholm_test <- cbind(adjp, bholm, index, out_flag)
  bholm_test2 <- bholm_test[order(index), ]
  colnames(bholm_test2) <- c("rawp", "bholm", "index", "out_flag")

  total_m4_data <- cbind(rawp2, bholm_test2)
  total_m4_data[which(total_m4_data$out_flag != "."), ]
}

# ----------------------------------------------------------
# Utility
# ----------------------------------------------------------
day_to_date <- function(day_of_year, year) {
  as.Date(day_of_year - 1, origin = paste0(year, "-01-01"))
}

# ----------------------------------------------------------
# 1. File input
# ----------------------------------------------------------
pheno_raw_file <- "Input/Pheno_raw.xlsx"
bed_file       <- "Input/SNPs_final_2022.bed"
bim_file       <- "Input/SNPs_final_2022.bim"
fam_file       <- "Input/SNPs_final_2022.fam"
harvest_file   <- "Output/Harvest_date_processed_final.csv"

# ----------------------------------------------------------
# 2. Caricamento dati
# ----------------------------------------------------------
cat("Caricamento Pheno_raw...\n")
pheno_raw <- as.data.frame(read_xlsx(pheno_raw_file))

cat("Caricamento PLINK...\n")
geno <- read.plink(bed_file, bim_file, fam_file)

cat("Caricamento Harvest_date finale...\n")
harvest_final <- read_csv(harvest_file, show_col_types = FALSE)

cat("Caricamento completato.\n\n")

dir.create("Output", showWarnings = FALSE)

# ----------------------------------------------------------
# 3. Preparazione Flowering_begin raw
# ----------------------------------------------------------
cat("Preparazione dati Flowering_begin...\n")

fb <- pheno_raw[, c("Row", "Position", "Genotype", "Management", "Checks",
                    "Year", "Country", "Envir", "Flowering_begin")]
colnames(fb)[ncol(fb)] <- "Trait"

geno_ids <- as.character(geno$fam$member)
fb <- fb[which(as.character(fb$Genotype) %in% geno_ids), ]

cat("Righe Flowering_begin dopo filtro genomico:", nrow(fb), "\n")
cat("Genotipi unici:", length(unique(fb$Genotype)), "\n\n")

# ----------------------------------------------------------
# 4. H2 per environment per Flowering_begin
# ----------------------------------------------------------
cat("Calcolo H2 per environment per Flowering_begin...\n")

h2_results <- NULL
env_list <- sort(unique(fb$Envir))

for (env in env_list) {
  d_sub <- subset(fb, Envir == env)
  d_sub <- d_sub[!is.na(d_sub$Trait), ]

  n_obs <- nrow(d_sub)
  n_gen <- length(unique(d_sub$Genotype))

  if (n_obs < 2 || n_gen < 2) {
    h2_results <- rbind(h2_results, data.frame(
      Envir = env, H2 = NA, n_obs = n_obs, n_genotypes = n_gen,
      mean_reps = NA, genov = NA, errorv = NA, status = "too_few_data",
      stringsAsFactors = FALSE
    ))
    next
  }

  d_sub$Genotype <- as.factor(as.character(d_sub$Genotype))

  fit <- try(lmer(Trait ~ (1 | Genotype), data = d_sub), silent = TRUE)

  if (inherits(fit, "try-error")) {
    h2_results <- rbind(h2_results, data.frame(
      Envir = env, H2 = NA, n_obs = n_obs, n_genotypes = n_gen,
      mean_reps = NA, genov = NA, errorv = NA, status = "model_failed",
      stringsAsFactors = FALSE
    ))
    next
  }

  vc <- as.data.frame(VarCorr(fit))
  genov <- vc$vcov[vc$grp == "Genotype"]
  errorv <- sigma(fit)^2
  mean_reps <- mean(as.numeric(table(d_sub$Genotype)))

  H2 <- genov / (genov + errorv / mean_reps)

  h2_results <- rbind(h2_results, data.frame(
    Envir = env, H2 = H2, n_obs = n_obs, n_genotypes = n_gen,
    mean_reps = mean_reps, genov = genov, errorv = errorv, status = "ok",
    stringsAsFactors = FALSE
  ))
}

h2_results$low_H2_flag <- ifelse(!is.na(h2_results$H2) & h2_results$H2 < 0.1, TRUE, FALSE)

write_csv(h2_results, "Output/Flowering_begin_heritability_by_environment.csv")

cat("Environment Flowering_begin con H2 < 0.1:\n")
print(h2_results$Envir[h2_results$low_H2_flag %in% TRUE])
cat("\n")

# ----------------------------------------------------------
# 5. Correzione spaziale SpATS per Flowering_begin
#    Escludiamo gli environment low-H2 dal ramo genotype-level
#    ma teniamo nota che ESP.2020 verrà comunque gestito dopo
# ----------------------------------------------------------
cat("Correzione spaziale SpATS per Flowering_begin...\n")

fb_for_spats <- fb
bad_env <- h2_results$Envir[h2_results$low_H2_flag %in% TRUE]
fb_for_spats <- fb_for_spats[!(fb_for_spats$Envir %in% bad_env), ]

means_all <- NULL
trees_all <- NULL

env_list_spats <- sort(unique(fb_for_spats$Envir))

for (env in env_list_spats) {
  cat("  SpATS environment:", env, "\n")

  data_env <- fb_for_spats[fb_for_spats$Envir == env, ]
  data_env <- data_env[!is.na(data_env$Trait), ]

  if (nrow(data_env) < 10 || length(unique(data_env$Genotype)) < 2) {
    cat("    -> saltato: dati insufficienti\n")
    next
  }

  data_env$column <- as.numeric(data_env$Row)
  data_env$row <- as.numeric(data_env$Position)
  data_env$C <- as.factor(data_env$column)
  data_env$R <- as.factor(data_env$row)
  data_env$Genotype <- as.factor(as.character(data_env$Genotype))

  data_env <- data_env[order(data_env$Row, data_env$Position), ]

  trees_env <- data_env[, c("Row", "Position", "Genotype", "Management",
                            "Checks", "Year", "Country", "Envir")]

  means_env <- data.frame(
    Genotype = as.character(unique(data_env$Genotype)),
    stringsAsFactors = FALSE
  )

  fit <- try(
    SpATS(
      response = "Trait",
      spatial = as.formula(~ PSANOVA(column, row)),
      genotype = "Genotype",
      genotype.as.random = TRUE,
      fixed = NULL,
      random = ~ C + R,
      data = data_env,
      control = controlSpATS()
    ),
    silent = TRUE
  )

  if (inherits(fit, "try-error")) {
    cat("    -> modello SpATS fallito\n")
    next
  }

  pred_gen <- predict(fit, which = "Genotype")
  pred_gen <- pred_gen[, c("Genotype", "predicted.values")]
  colnames(pred_gen)[2] <- "Flowering_begin_adjusted"

  means_env <- merge(means_env, pred_gen, all.x = TRUE)
  means_env$Envir <- env
  means_all <- bind_rows(means_all, means_env)

  d_fit <- fit$data
  d_fit$weights <- NULL
  colnames(d_fit)[ncol(d_fit)] <- "Trait_original"
  d_fit$residuals <- fit$residuals

  pred_gen$Genotype <- as.character(pred_gen$Genotype)
  d_fit$Genotype <- as.character(d_fit$Genotype)
  d_merge <- merge(d_fit, pred_gen, all.x = TRUE)

  d_merge[is.na(d_merge$Trait_original), "Flowering_begin_adjusted"] <- NA
  d_merge$Flowering_begin_adjusted_tree <- d_merge$Flowering_begin_adjusted + d_merge$residuals
  d_merge <- d_merge[order(d_merge$Row, d_merge$Position), ]

  trees_env$Flowering_begin_adjusted_tree <- d_merge$Flowering_begin_adjusted_tree
  trees_all <- bind_rows(trees_all, trees_env)
}

write_csv(means_all, "Output/Flowering_begin_adjusted_values_genotype.csv")
write_csv(trees_all, "Output/Flowering_begin_adjusted_values_trees.csv")

# ----------------------------------------------------------
# 6. Outlier detection paper-inspired su genotype-level
# ----------------------------------------------------------
cat("\nOutlier detection su Flowering_begin genotype-level...\n")

fb_adj <- means_all %>%
  rename(Value = Flowering_begin_adjusted)

fb_adj$Genotype <- as.factor(fb_adj$Genotype)
fb_adj$Envir <- as.factor(fb_adj$Envir)

mylm <- lmer(Value ~ Envir + (1 | Genotype), data = fb_adj)
outl <- outliers_bhmadr(fb_adj, "Value", mylm)

cat("Outlier Flowering_begin trovati:", nrow(outl), "\n\n")

if (nrow(outl) > 0) {
  outlier_indices <- as.numeric(outl$index)
  fb_proc <- fb_adj[-outlier_indices, ]
} else {
  fb_proc <- fb_adj
}

fb_proc <- fb_proc %>%
  select(Genotype, Envir, Value) %>%
  rename(Flowering_begin = Value)

write_csv(outl, "Output/Flowering_begin_outliers_paper_method.csv")
write_csv(fb_proc, "Output/Flowering_begin_processed_final.csv")

# ----------------------------------------------------------
# 7. Definizione periodi P1/P2
#    Usiamo Flowering_begin processato + Harvest_date processato
# ----------------------------------------------------------
cat("Definizione dei periodi P1/P2...\n")

flower_q <- fb_proc %>%
  group_by(Envir) %>%
  summarise(
    FD = round(quantile(Flowering_begin, probs = 0.9, na.rm = TRUE)),
    .groups = "drop"
  )

harvest_q <- harvest_final %>%
  group_by(Envir) %>%
  summarise(
    HD = round(quantile(Harvest_date, probs = 0.9, na.rm = TRUE)),
    .groups = "drop"
  )

dates <- full_join(flower_q, harvest_q, by = "Envir")

# eccezioni paper-inspired
dates$FD[dates$Envir == "ESP.2020"] <- 91
dates$FD[dates$Envir == "ESP.2022"] <- 96

dates$Year <- as.integer(sub(".*\\.", "", dates$Envir))

dates$FDdate <- day_to_date(dates$FD, dates$Year)
dates$HDdate <- day_to_date(dates$HD, dates$Year)

ndays <- 80

dates$P1_start <- dates$FDdate - ndays
dates$P1_end   <- dates$FDdate

dates$P2_start <- dates$FDdate + 1
dates$P2_end   <- dates$HDdate

write_csv(dates, "Output/environment_periods_P1_P2.csv")

# ----------------------------------------------------------
# 8. Report finale
# ----------------------------------------------------------
sink("Output/floweringbegin_and_periods_report.txt")

cat("=== STEP B3: FLOWERING_BEGIN + PERIODS REPORT ===\n\n")

cat("FLOWERING_BEGIN H2 PER ENVIRONMENT\n")
print(h2_results)
cat("\n\n")

cat("FLOWERING_BEGIN OUTLIER SUMMARY\n")
cat("Numero outlier trovati:", nrow(outl), "\n")
cat("Numero righe finali Flowering_begin:", nrow(fb_proc), "\n\n")

cat("PERIODI P1/P2\n")
print(dates)
cat("\n")

sink()

cat("File salvati:\n")
cat("- Output/Flowering_begin_heritability_by_environment.csv\n")
cat("- Output/Flowering_begin_adjusted_values_genotype.csv\n")
cat("- Output/Flowering_begin_adjusted_values_trees.csv\n")
cat("- Output/Flowering_begin_outliers_paper_method.csv\n")
cat("- Output/Flowering_begin_processed_final.csv\n")
cat("- Output/environment_periods_P1_P2.csv\n")
cat("- Output/floweringbegin_and_periods_report.txt\n")