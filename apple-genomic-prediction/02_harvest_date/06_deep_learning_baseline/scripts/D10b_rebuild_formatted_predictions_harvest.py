# -*- coding: utf-8 -*-

import os
import pandas as pd

trait = "Harvest_date"

wide_file = os.path.join(
    "Output", "DeepLearning_Harvest", "Predictions", f"{trait}_predictions_wide.csv"
)
cv_file = os.path.join("Input", "CV1_Strategy", "Harvest_date_CV.csv")
out_file = os.path.join(
    "Output", "DeepLearning_Harvest", "Predictions", f"{trait}_predictions_formatted.csv"
)

print("Loading wide predictions...")
wide = pd.read_csv(wide_file)

print("Loading CV template...")
cv = pd.read_csv(cv_file)

# chiave comune
wide["ID_key"] = wide[["Envir", "Genotype"]].astype(str).agg("-".join, axis=1)
cv["ID_key"] = cv[["Envir", "Genotype"]].astype(str).agg("-".join, axis=1)

# colonne split
split_cols = [c for c in cv.columns if c.startswith("CV")]

# teniamo solo le colonne utili dal wide
wide_keep = ["Envir", "Genotype", "Observed"] + split_cols
wide = wide[wide_keep + ["ID_key"]]

# melt CV
cv_long = cv[["Envir", "Genotype", "ID_key"] + split_cols].melt(
    id_vars=["Envir", "Genotype", "ID_key"],
    var_name="CV",
    value_name="Testing"
)

# melt predictions
pred_long = wide[["Envir", "Genotype", "Observed", "ID_key"] + split_cols].melt(
    id_vars=["Envir", "Genotype", "Observed", "ID_key"],
    var_name="CV",
    value_name="Predicted"
)

# merge pulito
formatted = cv_long.merge(
    pred_long[["ID_key", "CV", "Predicted", "Observed"]],
    on=["ID_key", "CV"],
    how="left"
)

# aggiungi Run e Fold
formatted["Run"] = formatted["CV"].str.extract(r"CV(\d)_Split")[0].astype(int)
formatted["Fold"] = formatted["CV"].str.extract(r"Split(\d)")[0].astype(int)

# ordine colonne finale
formatted = formatted[
    ["Envir", "Genotype", "Testing", "Run", "Fold", "Predicted", "Observed"]
].copy()

formatted.to_csv(out_file, index=False)

print("Saved corrected formatted predictions to:")
print(out_file)
print("Shape:", formatted.shape)
print(formatted.head())