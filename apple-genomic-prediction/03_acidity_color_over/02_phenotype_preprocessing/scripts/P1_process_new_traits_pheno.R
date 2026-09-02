# -*- coding: utf-8 -*-

################################################################################
### P1_process_new_traits_pheno.R
### Multi-trait phenotype preprocessing for no-soil network
###
### For each trait:
###   1. Heritability by environment
###   2. Spatial correction with SpATS
###   3. Outlier detection with BH-MADR / Holm
###   4. Phenotypic variance components
###   5. Final processed phenotype table
###
### Traits:
###   - Acidity
###   - Color_over
################################################################################

rm(list = ls())

cat("================================================================================\n")
cat("P1 - PROCESS NEW TRAITS PHENOTYPES\n")
cat("================================================================================\n\n")

library(readxl)
library(readr)
library(dplyr)
library(snpStats)
library(lme4)
library(SpATS)
library(multtest)

# =============================================================================
# SETTINGS
# =============================================================================

TRAITS <- c("Acidity", "Color_over")

PROJECT_DIR <- "."
OUTPUT_DIR <- file.path(PROJECT_DIR, "Output")
PHENO_OUT_DIR <- file.path(OUTPUT_DIR, "01_pheno_processed")
dir.create(PHENO_OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PHENO_FILE <- file.path("Input", "Pheno_raw.xlsx")

BED_FILE <- file.path("Input", "SNPs_final_2022.bed")
BIM_FILE <- file.path("Input", "SNPs_final_2022.bim")
FAM_FILE <- file.path("Input", "SNPs_final_2022.fam")

MIN_ROWS_SPATS <- 10
LOW_H2_THRESHOLD <- 0.1


# =============================================================================
# HELPERS
# =============================================================================

clean_genotype <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  x <- gsub("^G_", "", x)
  return(x)
}

trait_safe_name <- function(trait) {
  x <- trait
  x <- gsub("[^A-Za-z0-9_]+", "_", x)
  return(x)
}

check_file <- function(path) {
  if (!file.exists(path)) {
    stop(paste("File non trovato:", path))
  }
}

bh_madr_outliers <- function(mydata, value_col) {
  # mydata deve già essere senza NA nel value_col
  # e contenere .row_id, Genotype, Envir, Value

  if (!all(c(".row_id", "Genotype", "Envir", value_col) %in% colnames(mydata))) {
    stop("bh_madr_outliers: mancano colonne richieste.")
  }

  d <- mydata %>%
    filter(!is.na(.data[[value_col]])) %>%
    mutate(
      Genotype = as.factor(Genotype),
      Envir = as.factor(Envir)
    )

  if (nrow(d) < 5 || n_distinct(d$Genotype) < 2 || n_distinct(d$Envir) < 2) {
    return(d[0, , drop = FALSE])
  }

  fit <- try(
    lmer(as.formula(paste0(value_col, " ~ Envir + (1 | Genotype)")), data = d),
    silent = TRUE
  )

  if (inherits(fit, "try-error")) {
    return(d[0, , drop = FALSE])
  }

  resi <- residuals(fit, type = "response")

  med <- median(resi, na.rm = TRUE)
  MAD <- median(abs(resi - med), na.rm = TRUE)
  re_MAD <- MAD * 1.4826

  if (is.na(re_MAD) || re_MAD == 0) {
    return(d[0, , drop = FALSE])
  }

  res_MAD <- resi / re_MAD
  rawp <- 2 * (1 - pnorm(abs(res_MAD)))

  res_adj <- mt.rawp2adjp(rawp, proc = c("Holm"))

  # res_adj[[2]] contiene l'ordine originale dopo ordinamento interno
  adj_table <- data.frame(
    rawp = res_adj[[1]][, 1],
    bholm = res_adj[[1]][, 2],
    index = res_adj[[2]],
    stringsAsFactors = FALSE
  )

  adj_table <- adj_table[order(adj_table$index), ]

  out <- d
  out$residual <- as.numeric(resi)
  out$res_MAD <- as.numeric(res_MAD)
  out$rawp <- adj_table$rawp
  out$bholm <- adj_table$bholm
  out$out_flag <- ifelse(out$bholm < 0.05, "OUTLIER", ".")

  out <- out %>% filter(out_flag == "OUTLIER")

  return(out)
}

calculate_heritability_by_env <- function(d, trait_name, trait_dir) {
  cat("\n[", trait_name, "] Heritability by environment...\n", sep = "")

  results <- NULL
  env_list <- sort(unique(d$Envir))

  for (env in env_list) {
    d_sub <- subset(d, Envir == env)
    d_sub <- d_sub[!is.na(d_sub$Trait), ]

    n_obs <- nrow(d_sub)
    n_gen <- length(unique(d_sub$Genotype))

    if (n_obs < 2 || n_gen < 2) {
      results <- rbind(results, data.frame(
        Trait = trait_name,
        Envir = env,
        H2 = NA,
        n_obs = n_obs,
        n_genotypes = n_gen,
        mean_reps = NA,
        genov = NA,
        errorv = NA,
        status = "too_few_data",
        stringsAsFactors = FALSE
      ))
      next
    }

    d_sub$Genotype <- as.factor(as.character(d_sub$Genotype))

    fit <- try(lmer(Trait ~ (1 | Genotype), data = d_sub), silent = TRUE)

    if (inherits(fit, "try-error")) {
      results <- rbind(results, data.frame(
        Trait = trait_name,
        Envir = env,
        H2 = NA,
        n_obs = n_obs,
        n_genotypes = n_gen,
        mean_reps = NA,
        genov = NA,
        errorv = NA,
        status = "model_failed",
        stringsAsFactors = FALSE
      ))
      next
    }

    vc <- as.data.frame(VarCorr(fit))
    genov <- vc$vcov[vc$grp == "Genotype"]
    errorv <- sigma(fit)^2
    mean_reps <- mean(as.numeric(table(d_sub$Genotype)))

    H2 <- genov / (genov + errorv / mean_reps)

    results <- rbind(results, data.frame(
      Trait = trait_name,
      Envir = env,
      H2 = H2,
      n_obs = n_obs,
      n_genotypes = n_gen,
      mean_reps = mean_reps,
      genov = genov,
      errorv = errorv,
      status = "ok",
      stringsAsFactors = FALSE
    ))
  }

  results <- results[order(results$H2, decreasing = TRUE, na.last = TRUE), ]
  results$low_H2_flag <- ifelse(!is.na(results$H2) & results$H2 < LOW_H2_THRESHOLD, TRUE, FALSE)

  write_csv(results, file.path(trait_dir, paste0(trait_name, "_heritability_by_environment.csv")))

  return(results)
}

run_spatial_correction <- function(d, trait_name, trait_dir) {
  cat("\n[", trait_name, "] Spatial correction with SpATS...\n", sep = "")

  trees_all <- NULL
  means_all <- NULL
  failed_envs <- data.frame()

  env_list <- sort(unique(d$Envir))

  for (env in env_list) {
    cat("  Environment:", env, "\n")

    data_env <- d[d$Envir == env, ]
    data_env <- data_env[!is.na(data_env$Trait), ]

    if (nrow(data_env) < MIN_ROWS_SPATS || length(unique(data_env$Genotype)) < 2) {
      cat("    -> skipped: insufficient data\n")
      failed_envs <- rbind(failed_envs, data.frame(
        Trait = trait_name,
        Envir = env,
        reason = "insufficient_data",
        n_rows = nrow(data_env),
        n_genotypes = length(unique(data_env$Genotype)),
        stringsAsFactors = FALSE
      ))
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
      cat("    -> SpATS failed\n")
      failed_envs <- rbind(failed_envs, data.frame(
        Trait = trait_name,
        Envir = env,
        reason = "spats_failed",
        n_rows = nrow(data_env),
        n_genotypes = length(unique(data_env$Genotype)),
        stringsAsFactors = FALSE
      ))
      next
    }

    pred_gen <- predict(fit, which = "Genotype")
    pred_gen <- pred_gen[, c("Genotype", "predicted.values")]
    colnames(pred_gen)[2] <- paste0(trait_name, "_adjusted")

    means_env <- merge(means_env, pred_gen, all.x = TRUE)
    means_env$Envir <- env

    d_fit <- fit$data
    d_fit$weights <- NULL
    colnames(d_fit)[ncol(d_fit)] <- "Trait_original"
    d_fit$residuals <- fit$residuals

    pred_gen$Genotype <- as.character(pred_gen$Genotype)
    d_fit$Genotype <- as.character(d_fit$Genotype)

    d_merge <- merge(d_fit, pred_gen, all.x = TRUE)

    adjusted_col <- paste0(trait_name, "_adjusted")
    tree_col <- paste0(trait_name, "_adjusted_tree")

    d_merge[is.na(d_merge$Trait_original), adjusted_col] <- NA
    d_merge[[tree_col]] <- d_merge[[adjusted_col]] + d_merge$residuals

    d_merge <- d_merge[order(d_merge$Row, d_merge$Position), ]

    trees_env[[tree_col]] <- d_merge[[tree_col]]

    trees_all <- bind_rows(trees_all, trees_env)
    means_all <- bind_rows(means_all, means_env)
  }

  write_csv(trees_all, file.path(trait_dir, paste0(trait_name, "_adjusted_values_trees.csv")))
  write_csv(means_all, file.path(trait_dir, paste0(trait_name, "_adjusted_values_genotype.csv")))
  write_csv(failed_envs, file.path(trait_dir, paste0(trait_name, "_spats_failed_or_skipped_envs.csv")))

  return(list(
    trees_all = trees_all,
    means_all = means_all,
    failed_envs = failed_envs
  ))
}

run_outlier_detection <- function(means_all, trait_name, trait_dir) {
  cat("\n[", trait_name, "] Outlier detection BH-MADR...\n", sep = "")

  adjusted_col <- paste0(trait_name, "_adjusted")

  pheno_adj <- means_all %>%
    rename(Value = all_of(adjusted_col)) %>%
    mutate(
      .row_id = row_number(),
      Genotype = as.character(Genotype),
      Envir = as.character(Envir)
    )

  outl <- bh_madr_outliers(pheno_adj, "Value")

  cat("  Outliers found:", nrow(outl), "\n")

  if (nrow(outl) > 0) {
    pheno_no_outliers <- pheno_adj %>%
      filter(!(.row_id %in% outl$.row_id))
  } else {
    pheno_no_outliers <- pheno_adj
  }

  pheno_no_outliers <- pheno_no_outliers %>%
    select(-.row_id)

  write_csv(outl, file.path(trait_dir, paste0(trait_name, "_outliers_paper_method.csv")))
  write_csv(pheno_no_outliers, file.path(trait_dir, paste0(trait_name, "_processed_no_outliers.csv")))

  return(list(
    outliers = outl,
    processed_no_outliers = pheno_no_outliers
  ))
}

run_phenotypic_model <- function(trees_all, trait_name, trait_dir) {
  cat("\n[", trait_name, "] Phenotypic variance model...\n", sep = "")

  tree_col <- paste0(trait_name, "_adjusted_tree")

  trees <- trees_all %>%
    rename(Value = all_of(tree_col)) %>%
    filter(!is.na(Value))

  if (nrow(trees) < 10 || n_distinct(trees$Genotype) < 2 || n_distinct(trees$Envir) < 2) {
    cat("  Skipped phenotypic model: insufficient data.\n")

    empty_long <- data.frame(
      Trait = trait_name,
      Component = character(),
      Variance = numeric(),
      Proportion = numeric()
    )

    empty_wide <- data.frame(
      Trait = trait_name,
      varG = NA,
      varGE = NA,
      varRes = NA,
      propG = NA,
      propGE = NA,
      propRes = NA
    )

    write_csv(empty_long, file.path(trait_dir, paste0(trait_name, "_phenotypic_variance_components_long.csv")))
    write_csv(empty_wide, file.path(trait_dir, paste0(trait_name, "_phenotypic_variance_components_wide.csv")))

    return(list(long = empty_long, wide = empty_wide))
  }

  trees$Genotype <- as.factor(trees$Genotype)
  trees$Envir <- as.factor(trees$Envir)

  fit <- try(
    lmer(Value ~ Envir + (1 | Genotype) + (1 | Genotype:Envir), data = trees),
    silent = TRUE
  )

  if (inherits(fit, "try-error")) {
    cat("  Phenotypic model failed.\n")

    empty_long <- data.frame(
      Trait = trait_name,
      Component = character(),
      Variance = numeric(),
      Proportion = numeric()
    )

    empty_wide <- data.frame(
      Trait = trait_name,
      varG = NA,
      varGE = NA,
      varRes = NA,
      propG = NA,
      propGE = NA,
      propRes = NA
    )

    write_csv(empty_long, file.path(trait_dir, paste0(trait_name, "_phenotypic_variance_components_long.csv")))
    write_csv(empty_wide, file.path(trait_dir, paste0(trait_name, "_phenotypic_variance_components_wide.csv")))

    return(list(long = empty_long, wide = empty_wide))
  }

  vc <- as.data.frame(VarCorr(fit))
  vc_out <- vc[, c("grp", "vcov")]
  colnames(vc_out) <- c("Component", "Variance")

  vc_out$Component[vc_out$Component == "Genotype"] <- "Genotype"
  vc_out$Component[vc_out$Component == "Genotype:Envir"] <- "Genotype_by_Envir"
  vc_out$Component[vc_out$Component == "Residual"] <- "Residual"

  total_var <- sum(vc_out$Variance)
  vc_out$Proportion <- vc_out$Variance / total_var
  vc_out$Trait <- trait_name
  vc_out <- vc_out[, c("Trait", "Component", "Variance", "Proportion")]

  get_var <- function(component) {
    val <- vc_out$Variance[vc_out$Component == component]
    if (length(val) == 0) return(NA)
    return(val)
  }

  get_prop <- function(component) {
    val <- vc_out$Proportion[vc_out$Component == component]
    if (length(val) == 0) return(NA)
    return(val)
  }

  vc_wide <- data.frame(
    Trait = trait_name,
    varG = get_var("Genotype"),
    varGE = get_var("Genotype_by_Envir"),
    varRes = get_var("Residual"),
    propG = get_prop("Genotype"),
    propGE = get_prop("Genotype_by_Envir"),
    propRes = get_prop("Residual")
  )

  write_csv(vc_out, file.path(trait_dir, paste0(trait_name, "_phenotypic_variance_components_long.csv")))
  write_csv(vc_wide, file.path(trait_dir, paste0(trait_name, "_phenotypic_variance_components_wide.csv")))

  return(list(long = vc_out, wide = vc_wide))
}

finalize_trait <- function(processed_no_outliers, trait_name, trait_dir) {
  cat("\n[", trait_name, "] Finalizing processed phenotype...\n", sep = "")

  final_df <- processed_no_outliers %>%
    rename(!!trait_name := Value) %>%
    select(Genotype, Envir, all_of(trait_name))

  final_file <- file.path(trait_dir, paste0(trait_name, "_processed_final.csv"))
  write_csv(final_df, final_file)

  cat("  Saved final phenotype:", final_file, "\n")
  cat("  Rows:", nrow(final_df), "\n")
  cat("  Unique genotypes:", n_distinct(final_df$Genotype), "\n")
  cat("  Unique environments:", n_distinct(final_df$Envir), "\n")

  return(final_df)
}


# =============================================================================
# MAIN
# =============================================================================

for (f in c(PHENO_FILE, BED_FILE, BIM_FILE, FAM_FILE)) {
  check_file(f)
}

cat("Loading phenotype file...\n")
pheno <- as.data.frame(read_xlsx(PHENO_FILE))

cat("Loading PLINK genotype files...\n")
geno <- read.plink(BED_FILE, BIM_FILE, FAM_FILE)

geno_ids <- clean_genotype(geno$fam$member)

required_cols <- c(
  "Row", "Position", "Genotype", "Management", "Checks",
  "Year", "Country", "Envir"
)

missing_required <- setdiff(required_cols, colnames(pheno))
if (length(missing_required) > 0) {
  stop(paste("Mancano colonne richieste in Pheno_raw:", paste(missing_required, collapse = ", ")))
}

traits_found <- TRAITS[TRAITS %in% colnames(pheno)]
traits_missing <- setdiff(TRAITS, colnames(pheno))

if (length(traits_missing) > 0) {
  cat("WARNING - missing traits in Pheno_raw:\n")
  print(traits_missing)
}

if (length(traits_found) == 0) {
  stop("Nessun trait richiesto è presente in Pheno_raw.")
}

global_report_rows <- list()

for (trait in traits_found) {

  trait_name <- trait_safe_name(trait)
  trait_dir <- file.path(PHENO_OUT_DIR, trait_name)
  dir.create(trait_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n\n")
  cat("================================================================================\n")
  cat("PROCESSING TRAIT:", trait, "\n")
  cat("SAFE NAME:", trait_name, "\n")
  cat("================================================================================\n")

  d <- pheno[, c(required_cols, trait)]
  colnames(d)[ncol(d)] <- "Trait"

  d$Genotype <- clean_genotype(d$Genotype)
  d$Envir <- trimws(as.character(d$Envir))

  d <- d[which(d$Genotype %in% geno_ids), ]

  cat("Rows after genomic filter:", nrow(d), "\n")
  cat("Unique genotypes after genomic filter:", length(unique(d$Genotype)), "\n")
  cat("Unique environments:", length(unique(d$Envir)), "\n")
  cat("Non-missing trait rows:", sum(!is.na(d$Trait)), "\n")

  h2_results <- calculate_heritability_by_env(d, trait_name, trait_dir)

  # Escludiamo gli environment con H2 < 0.1 prima della correzione SpATS
  bad_env <- h2_results$Envir[h2_results$low_H2_flag %in% TRUE]

  cat("\n[", trait_name, "] Low-H2 environments excluded from downstream processing:\n", sep = "")
  print(bad_env)

  d_for_spats <- d[!(d$Envir %in% bad_env), ]

  cat("[", trait_name, "] Rows before low-H2 filtering: ", nrow(d), "\n", sep = "")
  cat("[", trait_name, "] Rows after low-H2 filtering: ", nrow(d_for_spats), "\n", sep = "")
  cat("[", trait_name, "] Environments after low-H2 filtering: ", length(unique(d_for_spats$Envir)), "\n", sep = "")

  spats_res <- run_spatial_correction(d_for_spats, trait_name, trait_dir)

  outlier_res <- run_outlier_detection(
    means_all = spats_res$means_all,
    trait_name = trait_name,
    trait_dir = trait_dir
  )

  pheno_model_res <- run_phenotypic_model(
    trees_all = spats_res$trees_all,
    trait_name = trait_name,
    trait_dir = trait_dir
  )

  final_df <- finalize_trait(
    processed_no_outliers = outlier_res$processed_no_outliers,
    trait_name = trait_name,
    trait_dir = trait_dir
  )

  # Trait-specific report
  report_file <- file.path(trait_dir, paste0(trait_name, "_processing_report.txt"))

  sink(report_file)

  cat("================================================================================\n")
  cat("P1 TRAIT PROCESSING REPORT\n")
  cat("================================================================================\n\n")

  cat("Trait original:", trait, "\n")
  cat("Trait safe name:", trait_name, "\n\n")

  cat("Rows after genomic filter:", nrow(d), "\n")
  cat("Unique genotypes after genomic filter:", length(unique(d$Genotype)), "\n")
  cat("Unique environments:", length(unique(d$Envir)), "\n")
  cat("Non-missing trait rows:", sum(!is.na(d$Trait)), "\n\n")

  cat("HERITABILITY BY ENVIRONMENT\n")
  print(h2_results)
  cat("\n\n")

  cat("LOW H2 ENVIRONMENTS\n")
  print(h2_results$Envir[h2_results$low_H2_flag %in% TRUE])
  cat("\n\n")
  cat("DATA AFTER LOW-H2 FILTERING\n")
  cat("Rows before low-H2 filtering:", nrow(d), "\n")
  cat("Rows after low-H2 filtering:", nrow(d_for_spats), "\n")
  cat("Environments before low-H2 filtering:", length(unique(d$Envir)), "\n")
  cat("Environments after low-H2 filtering:", length(unique(d_for_spats$Envir)), "\n")
  cat("\n\n")

  cat("SPATS FAILED/SKIPPED ENVIRONMENTS\n")
  print(spats_res$failed_envs)
  cat("\n\n")

  cat("OUTLIERS\n")
  cat("Number of outliers:", nrow(outlier_res$outliers), "\n")
  print(head(outlier_res$outliers))
  cat("\n\n")

  cat("PHENOTYPIC VARIANCE COMPONENTS\n")
  print(pheno_model_res$long)
  cat("\n\n")

  cat("FINAL PHENOTYPE\n")
  cat("Rows:", nrow(final_df), "\n")
  cat("Unique genotypes:", n_distinct(final_df$Genotype), "\n")
  cat("Unique environments:", n_distinct(final_df$Envir), "\n")
  print(head(final_df))
  cat("\n")

  sink()

  global_report_rows[[trait_name]] <- data.frame(
    Trait = trait_name,
    original_trait_name = trait,
    # rows_after_genomic_filter = nrow(d),
    # non_missing_rows = sum(!is.na(d$Trait)),
    # n_genotypes_input = length(unique(d$Genotype[!is.na(d$Trait)])),
    # n_environments_input = length(unique(d$Envir[!is.na(d$Trait)])),
    rows_after_genomic_filter = nrow(d),
    non_missing_rows_before_H2_filter = sum(!is.na(d$Trait)),
    rows_after_low_H2_filter = nrow(d_for_spats),
    non_missing_rows_after_H2_filter = sum(!is.na(d_for_spats$Trait)),
    n_genotypes_input_before_H2_filter = length(unique(d$Genotype[!is.na(d$Trait)])),
    n_environments_input_before_H2_filter = length(unique(d$Envir[!is.na(d$Trait)])),
    n_genotypes_after_H2_filter = length(unique(d_for_spats$Genotype[!is.na(d_for_spats$Trait)])),
    n_environments_after_H2_filter = length(unique(d_for_spats$Envir[!is.na(d_for_spats$Trait)])),
    n_h2_env_ok = sum(h2_results$status == "ok"),
    n_low_H2_env = sum(h2_results$low_H2_flag, na.rm = TRUE),
    n_spats_failed_or_skipped = nrow(spats_res$failed_envs),
    n_outliers = nrow(outlier_res$outliers),
    final_rows = nrow(final_df),
    final_genotypes = n_distinct(final_df$Genotype),
    final_environments = n_distinct(final_df$Envir),
    stringsAsFactors = FALSE
  )
}

global_summary <- bind_rows(global_report_rows)
write_csv(global_summary, file.path(PHENO_OUT_DIR, "P1_all_traits_processing_summary.csv"))

cat("\n================================================================================\n")
cat("P1 COMPLETED\n")
cat("================================================================================\n")
cat("Global summary saved in:\n")
cat(file.path(PHENO_OUT_DIR, "P1_all_traits_processing_summary.csv"), "\n")
print(global_summary)