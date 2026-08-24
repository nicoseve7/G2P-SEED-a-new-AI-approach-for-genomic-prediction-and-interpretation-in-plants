# ==================================================
# B7_build_W_paper_style.R
# Costruzione della versione ridotta paper-style di W
# ==================================================

rm(list = ls())

cat("=== STEP B7: BUILD W PAPER STYLE ===\n\n")

library(readr)
library(dplyr)

# -------------------------------
# 1. Percorso file
# -------------------------------
input_file <- "Output/W_environment_full.csv"

# -------------------------------
# 2. Caricamento dati
# -------------------------------
cat("Caricamento W completa...\n")
W_full <- read_csv(input_file, show_col_types = FALSE)
cat("Caricamento completato.\n\n")

cat("Numero righe:", nrow(W_full), "\n")
cat("Numero colonne:", ncol(W_full), "\n\n")

# -------------------------------
# 3. Colonne da rimuovere (paper-style)
# -------------------------------
cols_to_remove <- c(
  "pH.lower",
  "pH.upper",
  "Manganese.lower",
  "Manganese.upper",
  "Copper.lower",
  "Copper.upper"
)

cat("Colonne da rimuovere:\n")
print(cols_to_remove)
cat("\n")

# controllo presenza colonne
missing_cols <- setdiff(cols_to_remove, colnames(W_full))
if (length(missing_cols) > 0) {
  cat("ATTENZIONE: colonne non trovate:\n")
  print(missing_cols)
  cat("\n")
}

# -------------------------------
# 4. Costruzione W ridotta
# -------------------------------
W_paper <- W_full %>%
  select(-any_of(cols_to_remove))

cat("Numero colonne dopo riduzione:", ncol(W_paper), "\n\n")

cat("Prime righe W_paper:\n")
print(head(W_paper))
cat("\n")

cat("Missing in W_paper:\n")
print(colSums(is.na(W_paper)))
cat("\n")

# -------------------------------
# 5. Matrice numerica
# -------------------------------
W_paper_matrix <- as.data.frame(W_paper)
rownames(W_paper_matrix) <- W_paper_matrix$Envir
W_paper_matrix$Envir <- NULL
W_paper_matrix <- as.matrix(W_paper_matrix)

# -------------------------------
# 6. Salvataggio output
# -------------------------------
dir.create("Output", showWarnings = FALSE)

write_csv(W_paper, "Output/W_environment_paper_style.csv")
save(W_paper_matrix, file = "Output/W_environment_paper_style.RData")

sink("Output/W_environment_paper_style_report.txt")

cat("=== STEP B7: W PAPER STYLE REPORT ===\n\n")
cat("Numero righe W full:", nrow(W_full), "\n")
cat("Numero colonne W full:", ncol(W_full), "\n")
cat("Numero colonne W paper-style:", ncol(W_paper), "\n\n")

cat("Colonne rimosse:\n")
print(cols_to_remove)
cat("\n\n")

cat("Prime righe W paper-style:\n")
print(head(W_paper))
cat("\n\n")

cat("Missing in W paper-style:\n")
print(colSums(is.na(W_paper)))
cat("\n")

sink()

# -------------------------------
# 7. Stampa finale
# -------------------------------
cat("File salvati:\n")
cat("- Output/W_environment_paper_style.csv\n")
cat("- Output/W_environment_paper_style.RData\n")
cat("- Output/W_environment_paper_style_report.txt\n")