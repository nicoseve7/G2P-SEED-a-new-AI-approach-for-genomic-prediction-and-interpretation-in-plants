# -*- coding: utf-8 -*-

################################################################################
### H0_refit_and_save_no_soil_models.py
### Refit V3 no-soil split-wise and save .keras models for SHAP interpretation
################################################################################

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# =============================================================================
# SETTINGS
# =============================================================================

TRAIT = "Harvest_date"
MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"

OUT_DIR = (
    Path("02_harvest_date")
    / "07_neural_network"
    / "output"
)

META_FILE = (
    Path("02_harvest_date")
    / "06_deep_learning_baseline"
    / "output"
    / "numpy_arrays_harvest"
    / "sample_metadata_harvest.csv"
)

CV_FILE = (
    Path("data")
    / "raw"
    / "cv"
    / "Harvest_date_CV.csv"
)

INNER_SPLITS_FILE = (
    OUT_DIR
    / "datasets"
    / "inner_validation_splits_harvest.csv"
)

GENO_DIR = (
    Path("02_harvest_date")
    / "05_gradient_boosting"
    / "output"
    / "geno_files"
    / "Harvest_date"
)

NPY_DIR = (
    Path("02_harvest_date")
    / "06_deep_learning_baseline"
    / "output"
    / "numpy_arrays_harvest"
)

SPLIT_INPUTS_DIR = (
    OUT_DIR
    / "biologic_objects"
    / "split_inputs"
)

WEATHER_EXP_FILE = (
    OUT_DIR
    / "weather_features"
    / "weather_period_features_v3.npy"
)

BEST_PARAMS_FILE = OUT_DIR / "tuning" / "best_params_Harvest_date_no_soil.json"

GLOBAL_SEED = 42

UNMAPPED_HIDDEN_UNITS = 16
BIO_HIDDEN_UNITS = 8

MAX_EPOCHS = 500
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 20
SCALE_TARGET = True

FALLBACK_PARAMS = {
    "learning_rate": 0.001,
    "l2_lambda": 0.0,
    "fusion_hidden_units": 32,
    "dropout_rate": 0.3,
}


# =============================================================================
# BASIC UTILS
# =============================================================================

def make_dirs():
    for d in [
        OUT_DIR / "models",
        OUT_DIR / "metrics",
        OUT_DIR / "predictions",
        OUT_DIR / "loss_history",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_best_params():
    if BEST_PARAMS_FILE.exists():
        with open(BEST_PARAMS_FILE, "r", encoding="utf-8") as f:
            params = json.load(f)
        print(f"[INFO] Loaded best params from: {BEST_PARAMS_FILE}")
        return params

    print(f"[WARNING] Best params file not found: {BEST_PARAMS_FILE}")
    print("[WARNING] Using FALLBACK_PARAMS:")
    print(FALLBACK_PARAMS)
    return FALLBACK_PARAMS


# =============================================================================
# DATA LOADING
# =============================================================================

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


def load_shared_arrays():
    weather_exp_all = np.load(WEATHER_EXP_FILE).astype("float32")
    pca_all = np.load(NPY_DIR / "pca.npy").astype("float32")
    return weather_exp_all, pca_all


def load_split_objects(split_name: str):
    split_dir = SPLIT_INPUTS_DIR / split_name

    mapped_snps = pd.read_csv(split_dir / "mapped_snps.csv")["SNP"].astype(str).tolist()
    unmapped_snps = pd.read_csv(split_dir / "unmapped_snps.csv")["SNP"].astype(str).tolist()
    all_genes = pd.read_csv(split_dir / "all_genes.csv")["Gene"].astype(str).tolist()
    snp_to_gene_edges = pd.read_csv(split_dir / "snp_to_gene_edges.csv", dtype=str)

    return mapped_snps, unmapped_snps, all_genes, snp_to_gene_edges


# =============================================================================
# BIOLOGICAL MASK
# =============================================================================

def build_sparse_weight_matrix(input_names, output_names, edge_df, input_col, output_col):
    input_to_idx = {x: i for i, x in enumerate(input_names)}
    output_to_idx = {x: i for i, x in enumerate(output_names)}

    mat = np.zeros((len(input_names), len(output_names)), dtype=np.float32)

    for _, row in edge_df.iterrows():
        inp = str(row[input_col])
        out = str(row[output_col])

        if inp in input_to_idx and out in output_to_idx:
            mat[input_to_idx[inp], output_to_idx[out]] = 1.0

    return mat


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


# =============================================================================
# MODEL WITHOUT SOIL
# =============================================================================

def build_model(
    n_weather_exp,
    n_pca,
    n_mapped_snps,
    n_unmapped_snps,
    n_genes,
    snp_gene_mask,
    learning_rate,
    l2_lambda,
    fusion_hidden_units,
    dropout_rate,
):
    reg = tf.keras.regularizers.L2(l2_lambda) if l2_lambda > 0 else None

    weather_exp_input = tf.keras.layers.Input(shape=(n_weather_exp,), name="Weather_Exp_Input")
    pca_input = tf.keras.layers.Input(shape=(n_pca,), name="PCA_Input")
    mapped_input = tf.keras.layers.Input(shape=(n_mapped_snps,), name="Mapped_SNP_Input")
    unmapped_input = tf.keras.layers.Input(shape=(n_unmapped_snps,), name="Unmapped_SNP_Input")

    # Weather branch
    xw = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg, name="weather_dense_64")(weather_exp_input)
    xw = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg, name="weather_dense_32")(xw)
    xw = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg, name="weather_dense_16")(xw)
    xw = tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg, name="weather_embedding")(xw)

    # PCA branch
    xp = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg, name="pca_dense_128_a")(pca_input)
    xp = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg, name="pca_dense_128_b")(xp)
    xp = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg, name="pca_dense_64")(xp)
    xp = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg, name="pca_dense_32")(xp)
    xp = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg, name="pca_dense_16")(xp)
    xp = tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg, name="pca_embedding")(xp)

    # Mapped SNP -> gene branch
    gene_embedding = MaskedDense(
        units=n_genes,
        mask_matrix=snp_gene_mask,
        activation="relu",
        kernel_regularizer=None,
        name="snp_to_gene"
    )(mapped_input)

    # Unmapped SNP branch
    unmapped_embedding = tf.keras.layers.Dense(
        UNMAPPED_HIDDEN_UNITS,
        activation="relu",
        kernel_regularizer=reg,
        name="unmapped_hidden"
    )(unmapped_input)

    bio_concat = tf.keras.layers.Concatenate(name="bio_concat")(
        [gene_embedding, unmapped_embedding]
    )

    bio_embedding = tf.keras.layers.Dense(
        BIO_HIDDEN_UNITS,
        activation="relu",
        kernel_regularizer=reg,
        name="bio_hidden"
    )(bio_concat)

    # Global fusion without soil
    all_concat = tf.keras.layers.Concatenate(name="global_concat")(
        [xw, xp, bio_embedding]
    )

    fusion_hidden = tf.keras.layers.Dense(
        fusion_hidden_units,
        activation="relu",
        kernel_regularizer=reg,
        name="fusion_hidden"
    )(all_concat)

    if dropout_rate > 0:
        fusion_hidden = tf.keras.layers.Dropout(dropout_rate, name="fusion_dropout")(fusion_hidden)

    output = tf.keras.layers.Dense(1, activation="linear", name="output")(fusion_hidden)

    model = tf.keras.Model(
        inputs=[
            weather_exp_input,
            pca_input,
            mapped_input,
            unmapped_input,
        ],
        outputs=output,
        name=MODEL_NAME
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )

    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    make_dirs()
    set_all_seeds(GLOBAL_SEED)

    params = load_best_params()

    print("\n" + "=" * 80)
    print("H0 - REFIT AND SAVE NO-SOIL MODELS")
    print("=" * 80)
    print("Model:", MODEL_NAME)
    print("Using params:")
    print(params)
    print("Soil branch: REMOVED")
    print("=" * 80)

    meta, cv, split_cols = load_base_tables()
    inner = load_inner_splits()
    weather_exp_all, pca_all = load_shared_arrays()

    for split_name in split_cols:
        print(f"\nProcessing {split_name}")

        model_path = OUT_DIR / "models" / f"model_{split_name}.keras"

        if model_path.exists():
            print("Model already exists, skipping.")
            continue

        df = prepare_split_dataframe(meta, cv, inner, split_name)

        mapped_snps, unmapped_snps, all_genes, snp_to_gene_edges = load_split_objects(split_name)

        X_mapped = df[mapped_snps].apply(pd.to_numeric, errors="coerce")
        X_unmapped = df[unmapped_snps].apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(df[TRAIT], errors="coerce")

        subtrain_mask = df[f"{split_name}_Subtrain"] == 1
        val_mask = df[f"{split_name}_Validation"] == 1

        sub_idx = df.index[subtrain_mask]
        val_idx = df.index[val_mask]

        X_mapped_sub = X_mapped.loc[sub_idx].copy()
        X_mapped_val = X_mapped.loc[val_idx].copy()

        X_unmapped_sub = X_unmapped.loc[sub_idx].copy()
        X_unmapped_val = X_unmapped.loc[val_idx].copy()

        mapped_means = X_mapped_sub.mean()
        unmapped_means = X_unmapped_sub.mean()

        X_mapped_sub = X_mapped_sub.fillna(mapped_means)
        X_mapped_val = X_mapped_val.fillna(mapped_means)

        X_unmapped_sub = X_unmapped_sub.fillna(unmapped_means)
        X_unmapped_val = X_unmapped_val.fillna(unmapped_means)

        mapped_scaler = StandardScaler()
        unmapped_scaler = StandardScaler()
        weather_scaler = StandardScaler()
        pca_scaler = StandardScaler()

        weather_sub = weather_scaler.fit_transform(weather_exp_all[sub_idx])
        weather_val = weather_scaler.transform(weather_exp_all[val_idx])

        pca_sub = pca_scaler.fit_transform(pca_all[sub_idx])
        pca_val = pca_scaler.transform(pca_all[val_idx])

        mapped_sub = mapped_scaler.fit_transform(X_mapped_sub.values)
        mapped_val = mapped_scaler.transform(X_mapped_val.values)

        unmapped_sub = unmapped_scaler.fit_transform(X_unmapped_sub.values)
        unmapped_val = unmapped_scaler.transform(X_unmapped_val.values)

        y_sub = y.loc[sub_idx].to_numpy(dtype=float)
        y_val = y.loc[val_idx].to_numpy(dtype=float)

        if SCALE_TARGET:
            y_scaler = StandardScaler()
            y_sub_sc = y_scaler.fit_transform(y_sub.reshape(-1, 1)).ravel()
            y_val_sc = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        else:
            y_sub_sc = y_sub
            y_val_sc = y_val

        snp_gene_mask = build_sparse_weight_matrix(
            mapped_snps,
            all_genes,
            snp_to_gene_edges,
            "SNP",
            "Gene"
        )

        tf.keras.backend.clear_session()

        model = build_model(
            n_weather_exp=weather_sub.shape[1],
            n_pca=pca_sub.shape[1],
            n_mapped_snps=len(mapped_snps),
            n_unmapped_snps=len(unmapped_snps),
            n_genes=len(all_genes),
            snp_gene_mask=snp_gene_mask,
            learning_rate=params["learning_rate"],
            l2_lambda=params["l2_lambda"],
            fusion_hidden_units=params["fusion_hidden_units"],
            dropout_rate=params["dropout_rate"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0
            )
        ]

        history = model.fit(
            [
                weather_sub,
                pca_sub,
                mapped_sub,
                unmapped_sub,
            ],
            y_sub_sc,
            validation_data=(
                [
                    weather_val,
                    pca_val,
                    mapped_val,
                    unmapped_val,
                ],
                y_val_sc
            ),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=callbacks
        )

        model.save(model_path)
        print(f"Saved model: {model_path}")

        hist_df = pd.DataFrame({
            "epoch": np.arange(1, len(history.history["loss"]) + 1),
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "train_mae": history.history["mae"],
            "val_mae": history.history["val_mae"],
            "Split": split_name,
            "Model": MODEL_NAME,
        })

        hist_df.to_csv(
            OUT_DIR / "loss_history" / f"loss_history_{split_name}_H0_refit.csv",
            index=False
        )

    print("\nH0 completed.")


if __name__ == "__main__":
    main()
