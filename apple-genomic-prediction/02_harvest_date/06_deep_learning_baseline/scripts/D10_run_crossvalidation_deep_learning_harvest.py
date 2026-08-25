# -*- coding: utf-8 -*-

################################################################################
### D10_run_crossvalidation_deep_learning_harvest.py
### Adattamento di E_Run_Crossvalidation_DeepLearning.py per Harvest_date
################################################################################

import os
import sys

print("PYTHON:", sys.executable)
print("Starting D10...")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

print("TensorFlow imported")
print("TensorFlow version:", tf.__version__)

import time
import numpy as np
import pandas as pd

from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from math import sqrt

from deep_model import model
print("deep_model imported")

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# ------------------------------------------------------------------------------
# 2. Fixed settings
# ------------------------------------------------------------------------------
trait = "Harvest_date"
batch_size = 256
epochs = 1500
lr = 0.0001

data_numpy_dir = os.path.join(
    "02_harvest_date",
    "06_deep_learning_baseline",
    "output",
    "numpy_arrays_harvest"
)

save_dir = os.path.join(
    "02_harvest_date",
    "06_deep_learning_baseline",
    "output",
    "DeepLearning_Harvest"
)

models_dir = os.path.join(save_dir, "Models", trait)
pred_dir = os.path.join(save_dir, "Predictions")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# 3. Load shared arrays
# ------------------------------------------------------------------------------
print("Loading shared arrays...")
weather_all = np.load(os.path.join(data_numpy_dir, "weather.npy")).astype("float32")
pca_all = np.load(os.path.join(data_numpy_dir, "pca.npy")).astype("float32")
soil_all = np.load(os.path.join(data_numpy_dir, "soil.npy")).astype("float32")

print("weather_all shape:", weather_all.shape)
print("pca_all shape:", pca_all.shape)
print("soil_all shape:", soil_all.shape)

# ------------------------------------------------------------------------------
# 4. Load phenotype metadata
# ------------------------------------------------------------------------------
print("Loading sample metadata...")
pheno = pd.read_csv(os.path.join(data_numpy_dir, "sample_metadata_harvest.csv"))

# no missing expected, but keep repo logic style
nan_mask_trait = np.isnan(pheno[trait])
pheno_subset_trait = pheno[~nan_mask_trait].reset_index(drop=True)

# ------------------------------------------------------------------------------
# 5. Load and align CV template
# ------------------------------------------------------------------------------
print("Loading CV strategy...")
cv_template = pd.read_csv(
    os.path.join(
        "data",
        "raw",
        "cv",
        "Harvest_date_CV.csv"
    )
)
cv_template["ID_template"] = cv_template[["Envir", "Genotype"]].astype(str).agg("-".join, axis=1)
cv_template = cv_template.set_index("ID_template")

pheno_subset_trait["ID_pheno"] = pheno_subset_trait[["Envir", "Genotype"]].astype(str).agg("-".join, axis=1)
pheno_subset_trait = pheno_subset_trait.set_index("ID_pheno")

cv_template = cv_template.reindex(pheno_subset_trait.index)

if cv_template.isna().sum().sum() > 0:
    raise ValueError("CV template contains missing rows after reindexing. Check alignment.")

# response
y = pheno_subset_trait[trait].to_numpy().reshape(-1, 1)

# ------------------------------------------------------------------------------
# 6. Scaling (faithful to repo logic: fit on full trait dataset)
# ------------------------------------------------------------------------------
print("Scaling arrays...")

weather_trait = weather_all[~nan_mask_trait]
weather_2D = weather_trait.reshape(weather_trait.shape[0], weather_trait.shape[1] * weather_trait.shape[2])

scaler_weather = MinMaxScaler(feature_range=(0, 1))
scaler_weather = scaler_weather.fit(weather_2D)
weather_2D_sc = scaler_weather.transform(weather_2D)
weather_sc = weather_2D_sc.reshape(weather_trait.shape[0], weather_trait.shape[1], weather_trait.shape[2])

pca_trait = pca_all[~nan_mask_trait]
scaler_pca = MinMaxScaler(feature_range=(0, 1))
scaler_pca = scaler_pca.fit(pca_trait)
pca_sc = scaler_pca.transform(pca_trait)

soil_trait = soil_all[~nan_mask_trait]
scaler_soil = MinMaxScaler(feature_range=(0, 1))
scaler_soil = scaler_soil.fit(soil_trait)
soil_sc = scaler_soil.transform(soil_trait)

scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y = scaler_y.fit(y)
y_sc = scaler_y.transform(y)

# ------------------------------------------------------------------------------
# 7. Prepare outputs
# ------------------------------------------------------------------------------
metrics_cols = ["Split", "r", "time", "r2", "RMSE", "MAE"]
df_metrics = pd.DataFrame(columns=metrics_cols)

predictions_full = cv_template.copy()
split_cols = cv_template.columns[2:].tolist()

stop_early = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=35,
    restore_best_weights=True
)

# ------------------------------------------------------------------------------
# 8. Cross-validation loop
# ------------------------------------------------------------------------------
for split in split_cols:
    print(f"\nProcessing split {split}")

    snp_all = np.load(
        os.path.join(data_numpy_dir, "snps", f"SNP_{trait}_{split}.npy")
    ).astype("float32")

    time_0 = time.time()

    tf.keras.backend.clear_session()

    test_index = cv_template[split] == 1
    train_index = cv_template[split] == 0

    weather_train, weather_test = weather_sc[train_index], weather_sc[test_index]
    pca_train, pca_test = pca_sc[train_index], pca_sc[test_index]
    snp_train, snp_test = snp_all[train_index], snp_all[test_index]
    soil_train, soil_test = soil_sc[train_index], soil_sc[test_index]
    y_train, y_test = y_sc[train_index], y_sc[test_index]

    print("  train shapes:",
          weather_train.shape, pca_train.shape, snp_train.shape, soil_train.shape, y_train.shape)
    print("  test shapes:",
          weather_test.shape, pca_test.shape, snp_test.shape, soil_test.shape, y_test.shape)

    time_steps = weather_train.shape[1]
    n_pca = pca_train.shape[1]
    n_snp = snp_train.shape[1]
    n_soil = soil_train.shape[1]

    pred_model = model(time_steps, n_pca, n_snp, n_soil)
    pred_model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=lr)
    )

    pred_model.fit(
        [weather_train, pca_train, snp_train, soil_train],
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        shuffle=False,
        callbacks=[stop_early]
    )

    # predict test
    y_pred = pred_model.predict([weather_test, pca_test, snp_test, soil_test], verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred).reshape(y_test.shape[0],)

    time_1 = time.time()
    elapsed_time = time_1 - time_0

    y_true = scaler_y.inverse_transform(y_test).reshape(y_test.shape[0],)

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r = pearsonr(y_true, y_pred)[0]

    df_metrics.loc[len(df_metrics)] = [split, r, elapsed_time, r2, rmse, mae]

    # save model
    model_path = os.path.join(models_dir, f"model_{split}.keras")
    pred_model.save(model_path)

    # predict full dataset for storage
    y_pred_all = pred_model.predict([weather_sc, pca_sc, snp_all, soil_sc], verbose=0)
    y_pred_all = scaler_y.inverse_transform(y_pred_all).reshape(y_pred_all.shape[0],)
    predictions_full[split] = y_pred_all

# ------------------------------------------------------------------------------
# 9. Save metrics
# ------------------------------------------------------------------------------
mean_rmse = df_metrics["RMSE"].mean()
mean_mae = df_metrics["MAE"].mean()
mean_r = df_metrics["r"].mean()
mean_t = df_metrics["time"].mean()
mean_r2 = df_metrics["r2"].mean()

df_metrics.loc[len(df_metrics)] = ["Mean", mean_r, mean_t, mean_r2, mean_rmse, mean_mae]

metrics_file = os.path.join(save_dir, f"{trait}_metrics_splits.csv")
df_metrics.to_csv(metrics_file, index=False)
print("\nSaved metrics to:", metrics_file)

# ------------------------------------------------------------------------------
# 10. Format predictions to match repo style
# ------------------------------------------------------------------------------
df_melted_cv = cv_template.reset_index(drop=False).melt(
    id_vars=["Envir", "Genotype"],
    var_name="CV",
    value_name="Testing"
)

df_melted_cv["Run"] = df_melted_cv["CV"].str.extract(r"CV(\d)_Split")[0]
df_melted_cv["Fold"] = df_melted_cv["CV"].str.extract(r"Split(\d)")[0]
df_melted_cv["Run"] = pd.to_numeric(df_melted_cv["Run"])
df_melted_cv["Fold"] = pd.to_numeric(df_melted_cv["Fold"])

df_observed = pheno_subset_trait.reset_index(drop=False)[["Envir", "Genotype", trait]]
df_observed.columns = ["Envir", "Genotype", "Observed"]

predictions_full_obs = pd.merge(
    predictions_full.reset_index(drop=False),
    df_observed,
    how="left",
    on=["Envir", "Genotype"]
)

df_melted_pred_obs = predictions_full_obs.melt(
    id_vars=["Envir", "Genotype", "Observed"],
    var_name="CV",
    value_name="Predicted"
)

df_melted_cv["Predicted"] = df_melted_pred_obs["Predicted"]
df_melted_cv["Observed"] = df_melted_pred_obs["Observed"]

df_result = df_melted_cv.drop(columns=["CV"])

pred_file = os.path.join(pred_dir, f"{trait}_predictions_formatted.csv")
df_result.to_csv(pred_file, index=False)
print("Saved formatted predictions to:", pred_file)

# also save full wide predictions
pred_wide_file = os.path.join(pred_dir, f"{trait}_predictions_wide.csv")
predictions_full_obs.to_csv(pred_wide_file, index=False)
print("Saved wide predictions to:", pred_wide_file)

print("\nDeep learning cross-validation completed successfully.")
