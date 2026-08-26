# -*- coding: utf-8 -*-

################################################################################
### H1_compute_shap_no_soil_inputs.py
### SHAP computation for V3 no-soil multi-input model
################################################################################

import os
import sys
import gc
from pathlib import Path

print("PYTHON:", sys.executable)
print("Starting SHAP computation for V3 no-soil...")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import shap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices("GPU")
print("GPUs visible to TensorFlow:", gpus)

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Could not set memory growth for {gpu}: {e}")


# =============================================================================
# SETTINGS
# =============================================================================

TRAIT = "Harvest_date"
MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"

OUT_DIR = Path("Output")

META_FILE = Path("../Output/Intermediate/numpy_arrays_harvest/sample_metadata_harvest.csv")
CV_FILE = Path("../Input/CV1_Strategy/Harvest_date_CV.csv")

# In senza_suolo this should be local
INNER_SPLITS_FILE = Path("Output/datasets/inner_validation_splits_harvest.csv")

GENO_DIR = Path("../Output/Intermediate/geno_files/Harvest_date")
NPY_DIR = Path("../Output/Intermediate/numpy_arrays_harvest")
WEATHER_EXP_FILE = Path("../esperimento_meteo/Output/numpy_arrays_weather_exp/weather_period_features_v3.npy")

MODEL_DIR = OUT_DIR / "models"
SPLIT_INPUTS_DIR = Path("Output/biologic_objects/split_inputs")

SAVE_DIR = OUT_DIR / "Interpretation" / TRAIT / "SHAP_arrays"
SAVE_SPLITWISE_DIR = OUT_DIR / "Interpretation" / TRAIT / "Splitwise_tables"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_SPLITWISE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILS
# =============================================================================

def atomic_save_npy(path, array):
    tmp_path = str(path) + ".tmp.npy"
    np.save(tmp_path, array)
    os.replace(tmp_path, path)


def load_base_tables():
    meta = pd.read_csv(META_FILE)
    meta = meta[["Envir", "Genotype", TRAIT]].copy()
    meta["Envir"] = meta["Envir"].astype(str)
    meta["Genotype"] = meta["Genotype"].astype(str)
    meta["ID_key"] = meta["Envir"] + "-" + meta["Genotype"]

    cv = pd.read_csv(CV_FILE)
    cv["Envir"] = cv["Envir"].astype(str)
    cv["Genotype"] = cv["Genotype"].astype(str)
    cv["ID_key"] = cv["Envir"] + "-" + cv["Genotype"]

    cv = cv.set_index("ID_key").loc[meta["ID_key"]].reset_index()
    split_cols = [c for c in cv.columns if c.startswith("CV")]

    return meta, cv, split_cols


def load_inner_splits():
    inner = pd.read_csv(INNER_SPLITS_FILE)
    inner["Envir"] = inner["Envir"].astype(str)
    inner["Genotype"] = inner["Genotype"].astype(str)
    inner["ID_key"] = inner["ID_key"].astype(str)
    return inner


def prepare_split_dataframe(meta: pd.DataFrame, cv: pd.DataFrame, inner: pd.DataFrame, split_name: str):
    geno_file = GENO_DIR / f"geno_{split_name}.csv"
    Xgeno = pd.read_csv(geno_file)

    first_col = Xgeno.columns[0]
    Xgeno[first_col] = Xgeno[first_col].astype(str).str.replace("^G_", "", regex=True)
    Xgeno = Xgeno.rename(columns={first_col: "Genotype"})
    Xgeno["Genotype"] = Xgeno["Genotype"].astype(str)

    df = meta.merge(Xgeno, on="Genotype", how="left")
    df["Testing"] = cv[split_name].values

    inner_cols = [
        "ID_key",
        f"{split_name}_role",
        f"{split_name}_Testing",
        f"{split_name}_Validation",
        f"{split_name}_Subtrain",
    ]

    df = df.merge(inner[inner_cols], on="ID_key", how="left")
    df["Split"] = split_name

    return df


@tf.keras.utils.register_keras_serializable(package="Custom")
class MaskedDense(tf.keras.layers.Layer):
    def __init__(self, units, mask_matrix, activation=None, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.mask_matrix_np = np.asarray(mask_matrix, dtype=np.float32)
        self.mask_matrix = tf.constant(self.mask_matrix_np, dtype=tf.float32)
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        masked_kernel = self.kernel * self.mask_matrix
        out = tf.linalg.matmul(inputs, masked_kernel) + self.bias

        if self.activation is not None:
            out = self.activation(out)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "mask_matrix": self.mask_matrix_np.tolist(),
            "activation": tf.keras.activations.serialize(self.activation),
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        mask_matrix = np.array(config.pop("mask_matrix"), dtype=np.float32)
        activation = tf.keras.activations.deserialize(config.pop("activation"))
        kernel_regularizer = tf.keras.regularizers.deserialize(config.pop("kernel_regularizer"))

        return cls(
            mask_matrix=mask_matrix,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            **config
        )


def normalize_multiinput_shap_output(shap_values):
    """
    Expected for multi-input single-output model:
    list with one item per input branch.

    For no-soil model:
      [weather, pca, mapped, unmapped]
    """
    if not isinstance(shap_values, list):
        raise ValueError(f"Expected SHAP output as list for multi-input model, got {type(shap_values)}")

    normalized = []

    for arr in shap_values:
        arr = np.asarray(arr)

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]

        if arr.ndim != 2:
            raise ValueError(f"Unexpected branch SHAP shape: {arr.shape}")

        normalized.append(arr)

    return normalized


# =============================================================================
# SPLIT DATA
# =============================================================================

def get_split_data(split_name: str):
    meta, cv, _ = load_base_tables()
    inner = load_inner_splits()

    weather_all = np.load(WEATHER_EXP_FILE).astype("float32")
    pca_all = np.load(NPY_DIR / "pca.npy").astype("float32")

    df = prepare_split_dataframe(meta, cv, inner, split_name)

    split_dir = SPLIT_INPUTS_DIR / split_name

    mapped_snps = pd.read_csv(split_dir / "mapped_snps.csv")["SNP"].astype(str).tolist()
    unmapped_snps = pd.read_csv(split_dir / "unmapped_snps.csv")["SNP"].astype(str).tolist()

    weather_cols = [f"WeatherV3_{i+1}" for i in range(weather_all.shape[1])]
    pca_cols = [f"PCA_{i+1}" for i in range(pca_all.shape[1])]

    X_mapped = df[mapped_snps].apply(pd.to_numeric, errors="coerce")
    X_unmapped = df[unmapped_snps].apply(pd.to_numeric, errors="coerce")

    subtrain_mask = df[f"{split_name}_Subtrain"] == 1
    test_mask = df["Testing"] == 1

    sub_idx = df.index[subtrain_mask]
    test_idx = df.index[test_mask]

    # Fill missing SNP values using subtrain means only
    mapped_means = X_mapped.loc[sub_idx].mean()
    unmapped_means = X_unmapped.loc[sub_idx].mean()

    X_mapped_sub = X_mapped.loc[sub_idx].fillna(mapped_means)
    X_mapped_test = X_mapped.loc[test_idx].fillna(mapped_means)

    X_unmapped_sub = X_unmapped.loc[sub_idx].fillna(unmapped_means)
    X_unmapped_test = X_unmapped.loc[test_idx].fillna(unmapped_means)

    # Scalers fitted on subtrain only
    mapped_scaler = StandardScaler()
    unmapped_scaler = StandardScaler()
    weather_scaler = StandardScaler()
    pca_scaler = StandardScaler()

    weather_sub = weather_scaler.fit_transform(weather_all[sub_idx]).astype("float32")
    weather_test = weather_scaler.transform(weather_all[test_idx]).astype("float32")

    pca_sub = pca_scaler.fit_transform(pca_all[sub_idx]).astype("float32")
    pca_test = pca_scaler.transform(pca_all[test_idx]).astype("float32")

    mapped_sub = mapped_scaler.fit_transform(X_mapped_sub.values).astype("float32")
    mapped_test = mapped_scaler.transform(X_mapped_test.values).astype("float32")

    unmapped_sub = unmapped_scaler.fit_transform(X_unmapped_sub.values).astype("float32")
    unmapped_test = unmapped_scaler.transform(X_unmapped_test.values).astype("float32")

    return {
        "background": [
            weather_sub,
            pca_sub,
            mapped_sub,
            unmapped_sub,
        ],
        "test_inputs": [
            weather_test,
            pca_test,
            mapped_test,
            unmapped_test,
        ],
        "weather_cols": weather_cols,
        "pca_cols": pca_cols,
        "mapped_cols": mapped_snps,
        "unmapped_cols": unmapped_snps,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    _, _, split_cols = load_base_tables()

    custom_objects = {"MaskedDense": MaskedDense}

    for split in split_cols:
        print("\n" + "=" * 80)
        print(f"Processing SHAP for {split}")
        print("=" * 80)

        out_weather = SAVE_DIR / f"{split}_SHAP_weather.npy"
        out_pca = SAVE_DIR / f"{split}_SHAP_pca.npy"
        out_mapped = SAVE_DIR / f"{split}_SHAP_mapped.npy"
        out_unmapped = SAVE_DIR / f"{split}_SHAP_unmapped.npy"

        out_branch_csv = SAVE_SPLITWISE_DIR / f"{split}_branch_importance.csv"
        out_allsnp_csv = SAVE_SPLITWISE_DIR / f"{split}_all_snp_mean_abs_SHAP.csv"

        expected_outputs = [
            out_weather,
            out_pca,
            out_mapped,
            out_unmapped,
            out_branch_csv,
            out_allsnp_csv,
        ]

        if all(p.exists() for p in expected_outputs):
            print(f"Outputs already exist for {split}, skipping.")
            continue

        model_path = MODEL_DIR / f"model_{split}.keras"

        if not model_path.exists():
            print(f"Model not found for {split}, skipping.")
            continue

        split_data = get_split_data(split)

        background = split_data["background"]
        test_inputs = split_data["test_inputs"]

        tf.keras.backend.clear_session()
        gc.collect()

        print("Loading model:", model_path)
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        print("Model input names:")
        for inp in model.inputs:
            print(" -", inp.name, inp.shape)

        print("Background shapes:")
        for x in background:
            print(x.shape)

        print("Test input shapes:")
        for x in test_inputs:
            print(x.shape)

        print("Building SHAP GradientExplainer...")
        explainer = shap.GradientExplainer(model, background)

        print("Computing SHAP values on full test set...")
        shap_values = explainer.shap_values(test_inputs)
        shap_values = normalize_multiinput_shap_output(shap_values)

        if len(shap_values) != 4:
            raise ValueError(
                f"Expected 4 SHAP arrays for no-soil model "
                f"[weather, pca, mapped, unmapped], got {len(shap_values)}"
            )

        shap_weather, shap_pca, shap_mapped, shap_unmapped = shap_values

        # Save arrays
        atomic_save_npy(out_weather, shap_weather.astype("float32"))
        atomic_save_npy(out_pca, shap_pca.astype("float32"))
        atomic_save_npy(out_mapped, shap_mapped.astype("float32"))
        atomic_save_npy(out_unmapped, shap_unmapped.astype("float32"))

        # Splitwise tables
        def save_mean_abs(arr, names, filename, colname):
            mean_abs = np.mean(np.abs(arr), axis=0)

            df = pd.DataFrame({
                "Split": split,
                colname: names,
                "mean_abs_SHAP": mean_abs
            }).sort_values(
                "mean_abs_SHAP",
                ascending=False
            ).reset_index(drop=True)

            df["Rank"] = np.arange(1, len(df) + 1)
            df.to_csv(filename, index=False)

        save_mean_abs(
            shap_weather,
            split_data["weather_cols"],
            SAVE_SPLITWISE_DIR / f"{split}_weather_mean_abs_SHAP.csv",
            "Feature"
        )

        save_mean_abs(
            shap_pca,
            split_data["pca_cols"],
            SAVE_SPLITWISE_DIR / f"{split}_pca_mean_abs_SHAP.csv",
            "Feature"
        )

        save_mean_abs(
            shap_mapped,
            split_data["mapped_cols"],
            SAVE_SPLITWISE_DIR / f"{split}_mapped_snp_mean_abs_SHAP.csv",
            "SNP"
        )

        save_mean_abs(
            shap_unmapped,
            split_data["unmapped_cols"],
            SAVE_SPLITWISE_DIR / f"{split}_unmapped_snp_mean_abs_SHAP.csv",
            "SNP"
        )

        # All SNP table
        mapped_df = pd.read_csv(SAVE_SPLITWISE_DIR / f"{split}_mapped_snp_mean_abs_SHAP.csv")
        mapped_df["MappedStatus"] = "mapped"

        unmapped_df = pd.read_csv(SAVE_SPLITWISE_DIR / f"{split}_unmapped_snp_mean_abs_SHAP.csv")
        unmapped_df["MappedStatus"] = "unmapped"

        all_snp_df = (
            pd.concat([mapped_df, unmapped_df], ignore_index=True)
            .sort_values("mean_abs_SHAP", ascending=False)
            .reset_index(drop=True)
        )

        all_snp_df["Rank"] = np.arange(1, len(all_snp_df) + 1)
        all_snp_df.to_csv(out_allsnp_csv, index=False)

        # Branch importance WITHOUT soil
        branch_df = pd.DataFrame({
            "Split": [split] * 4,
            "Branch": [
                "Weather",
                "PCA",
                "Mapped_SNP",
                "Unmapped_SNP",
            ],
            "mean_abs_SHAP": [
                np.mean(np.abs(shap_weather)),
                np.mean(np.abs(shap_pca)),
                np.mean(np.abs(shap_mapped)),
                np.mean(np.abs(shap_unmapped)),
            ]
        })

        branch_df.to_csv(out_branch_csv, index=False)

        print(f"Saved SHAP outputs for {split}")

        del model, explainer, shap_values
        tf.keras.backend.clear_session()
        gc.collect()

    print("\nSHAP computation for V3 no-soil completed.")
    print("Array folder:", SAVE_DIR)
    print("Splitwise table folder:", SAVE_SPLITWISE_DIR)


if __name__ == "__main__":
    main()