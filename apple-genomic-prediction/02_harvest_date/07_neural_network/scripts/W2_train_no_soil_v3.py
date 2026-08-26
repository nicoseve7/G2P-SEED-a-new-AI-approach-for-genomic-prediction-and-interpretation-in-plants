# -*- coding: utf-8 -*-

################################################################################
### W2_train_no_soil_v3.py
###
### Train finale sui 25 split della rete V3 SENZA ramo suolo.
###
### Architettura:
###   Weather expanded branch
###   PCA branch
###   mapped SNP -> gene -> ReLU branch
###   unmapped SNP -> hidden branch
###   concatenation -> fusion hidden -> output
###
### Input rimossi:
###   Soil branch
###
### Da eseguire da:
###   dalpaper/senza_suolo/
################################################################################

import os
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# SETTINGS
# =============================================================================

MODE = "train"

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

GLOBAL_SEED = 42

UNMAPPED_HIDDEN_UNITS = 16
BIO_HIDDEN_UNITS = 8

MAX_EPOCHS = 500
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 20
SCALE_TARGET = True

BEST_PARAMS_FILE = OUT_DIR / "tuning" / "best_params_Harvest_date_no_soil.json"

FALLBACK_PARAMS = {
    "learning_rate": 0.001,
    "l2_lambda": 0.0,
    "fusion_hidden_units": 32,
    "dropout_rate": 0.2,
}


# =============================================================================
# BASIC UTILS
# =============================================================================

def make_dirs():
    subdirs = [
        OUT_DIR / "datasets",
        OUT_DIR / "tuning",
        OUT_DIR / "models",
        OUT_DIR / "predictions",
        OUT_DIR / "metrics",
        OUT_DIR / "loss_history",
        OUT_DIR / "grafici" / "per_split",
        OUT_DIR / "grafici" / "summary",
    ]

    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


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


def prepare_split_dataframe(
    meta: pd.DataFrame,
    cv: pd.DataFrame,
    inner: pd.DataFrame,
    split_name: str
):
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

    return {
        "mapped_snps": mapped_snps,
        "unmapped_snps": unmapped_snps,
        "all_genes": all_genes,
        "snp_to_gene_edges": snp_to_gene_edges,
    }


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
    def __init__(
        self,
        units,
        mask_matrix,
        activation=None,
        kernel_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.mask_matrix_np = np.array(mask_matrix, dtype=np.float32)
        self.mask_matrix = tf.constant(self.mask_matrix_np, dtype=tf.float32)
        self.activation = tf.keras.activations.get(activation)
        self.activation_name = tf.keras.activations.serialize(self.activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_regularizer_config = tf.keras.regularizers.serialize(self.kernel_regularizer)

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
            "activation": self.activation_name,
            "kernel_regularizer": self.kernel_regularizer_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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

    weather_exp_input = tf.keras.layers.Input(
        shape=(n_weather_exp,),
        name="Weather_Exp_Input"
    )

    pca_input = tf.keras.layers.Input(
        shape=(n_pca,),
        name="PCA_Input"
    )

    mapped_input = tf.keras.layers.Input(
        shape=(n_mapped_snps,),
        name="Mapped_SNP_Input"
    )

    unmapped_input = tf.keras.layers.Input(
        shape=(n_unmapped_snps,),
        name="Unmapped_SNP_Input"
    )

    # Weather expanded branch
    xw = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=reg,
        name="weather_dense_64"
    )(weather_exp_input)

    xw = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=reg,
        name="weather_dense_32"
    )(xw)

    xw = tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=reg,
        name="weather_dense_16"
    )(xw)

    xw = tf.keras.layers.Dense(
        8,
        activation="relu",
        kernel_regularizer=reg,
        name="weather_embedding"
    )(xw)

    # PCA branch
    xp = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=reg,
        name="pca_dense_128_a"
    )(pca_input)

    xp = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=reg,
        name="pca_dense_128_b"
    )(xp)

    xp = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=reg,
        name="pca_dense_64"
    )(xp)

    xp = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=reg,
        name="pca_dense_32"
    )(xp)

    xp = tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=reg,
        name="pca_dense_16"
    )(xp)

    xp = tf.keras.layers.Dense(
        8,
        activation="relu",
        kernel_regularizer=reg,
        name="pca_embedding"
    )(xp)

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

    # Global fusion WITHOUT soil
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
        fusion_hidden = tf.keras.layers.Dropout(
            dropout_rate,
            name="fusion_dropout"
        )(fusion_hidden)

    output = tf.keras.layers.Dense(
        1,
        activation="linear",
        name="output"
    )(fusion_hidden)

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
# RUN ONE SPLIT
# =============================================================================

def run_one_split(split_name: str, params: dict, save_outputs: bool = False):
    set_all_seeds(GLOBAL_SEED)

    meta, cv, _ = load_base_tables()
    inner = load_inner_splits()

    weather_exp_all, pca_all = load_shared_arrays()
    split_obj = load_split_objects(split_name)

    df = prepare_split_dataframe(meta, cv, inner, split_name)

    mapped_snps = split_obj["mapped_snps"]
    unmapped_snps = split_obj["unmapped_snps"]
    all_genes = split_obj["all_genes"]
    snp_to_gene_edges = split_obj["snp_to_gene_edges"]

    X_mapped = df[mapped_snps].apply(pd.to_numeric, errors="coerce")
    X_unmapped = df[unmapped_snps].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TRAIT], errors="coerce")

    subtrain_mask = df[f"{split_name}_Subtrain"] == 1
    val_mask = df[f"{split_name}_Validation"] == 1
    test_mask = df["Testing"] == 1

    subtrain_idx = df.index[subtrain_mask]
    val_idx = df.index[val_mask]
    test_idx = df.index[test_mask]

    # SNP split
    X_mapped_subtrain = X_mapped.loc[subtrain_idx].copy()
    X_mapped_val = X_mapped.loc[val_idx].copy()
    X_mapped_test = X_mapped.loc[test_idx].copy()

    X_unmapped_subtrain = X_unmapped.loc[subtrain_idx].copy()
    X_unmapped_val = X_unmapped.loc[val_idx].copy()
    X_unmapped_test = X_unmapped.loc[test_idx].copy()

    # Shared arrays split
    weather_exp_subtrain = weather_exp_all[subtrain_idx]
    weather_exp_val = weather_exp_all[val_idx]
    weather_exp_test = weather_exp_all[test_idx]

    pca_subtrain = pca_all[subtrain_idx]
    pca_val = pca_all[val_idx]
    pca_test = pca_all[test_idx]

    y_subtrain = y.loc[subtrain_idx].to_numpy(dtype=float)
    y_val = y.loc[val_idx].to_numpy(dtype=float)
    y_test = y.loc[test_idx].to_numpy(dtype=float)

    # Missing SNP imputation from subtrain only
    mapped_means = X_mapped_subtrain.mean()
    unmapped_means = X_unmapped_subtrain.mean()

    X_mapped_subtrain = X_mapped_subtrain.fillna(mapped_means)
    X_mapped_val = X_mapped_val.fillna(mapped_means)
    X_mapped_test = X_mapped_test.fillna(mapped_means)

    X_unmapped_subtrain = X_unmapped_subtrain.fillna(unmapped_means)
    X_unmapped_val = X_unmapped_val.fillna(unmapped_means)
    X_unmapped_test = X_unmapped_test.fillna(unmapped_means)

    # Scaling from subtrain only
    mapped_scaler = StandardScaler()
    X_mapped_subtrain_sc = mapped_scaler.fit_transform(X_mapped_subtrain.values)
    X_mapped_val_sc = mapped_scaler.transform(X_mapped_val.values)
    X_mapped_test_sc = mapped_scaler.transform(X_mapped_test.values)

    unmapped_scaler = StandardScaler()
    X_unmapped_subtrain_sc = unmapped_scaler.fit_transform(X_unmapped_subtrain.values)
    X_unmapped_val_sc = unmapped_scaler.transform(X_unmapped_val.values)
    X_unmapped_test_sc = unmapped_scaler.transform(X_unmapped_test.values)

    weather_scaler = StandardScaler()
    weather_exp_sub_sc = weather_scaler.fit_transform(weather_exp_subtrain)
    weather_exp_val_sc = weather_scaler.transform(weather_exp_val)
    weather_exp_test_sc = weather_scaler.transform(weather_exp_test)

    pca_scaler = StandardScaler()
    pca_sub_sc = pca_scaler.fit_transform(pca_subtrain)
    pca_val_sc = pca_scaler.transform(pca_val)
    pca_test_sc = pca_scaler.transform(pca_test)

    if SCALE_TARGET:
        y_scaler = StandardScaler()
        y_subtrain_sc = y_scaler.fit_transform(y_subtrain.reshape(-1, 1)).ravel()
        y_val_sc = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test_sc = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_scaler = None
        y_subtrain_sc = y_subtrain
        y_val_sc = y_val
        y_test_sc = y_test

    # Sparse SNP -> gene mask
    snp_gene_mask = build_sparse_weight_matrix(
        input_names=mapped_snps,
        output_names=all_genes,
        edge_df=snp_to_gene_edges,
        input_col="SNP",
        output_col="Gene"
    )

    # Build model
    model = build_model(
        n_weather_exp=weather_exp_sub_sc.shape[1],
        n_pca=pca_sub_sc.shape[1],
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
            verbose=0,
        )
    ]

    history = model.fit(
        [
            weather_exp_sub_sc,
            pca_sub_sc,
            X_mapped_subtrain_sc,
            X_unmapped_subtrain_sc,
        ],
        y_subtrain_sc,
        validation_data=(
            [
                weather_exp_val_sc,
                pca_val_sc,
                X_mapped_val_sc,
                X_unmapped_val_sc,
            ],
            y_val_sc,
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
    )

    y_pred_test_sc = model.predict(
        [
            weather_exp_test_sc,
            pca_test_sc,
            X_mapped_test_sc,
            X_unmapped_test_sc,
        ],
        verbose=0
    ).ravel()

    if y_scaler is not None:
        y_pred_test = y_scaler.inverse_transform(
            y_pred_test_sc.reshape(-1, 1)
        ).ravel()
    else:
        y_pred_test = y_pred_test_sc

    rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    r = pearson_r(y_test, y_pred_test)

    best_epoch_idx = int(np.argmin(history.history["val_loss"]))

    result = {
        "split": split_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "r": r,
        "best_epoch": best_epoch_idx + 1,
        "best_val_loss": float(history.history["val_loss"][best_epoch_idx]),
        "best_val_mae": float(history.history["val_mae"][best_epoch_idx]),
        "history": history.history,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "mapped_snps": mapped_snps,
        "unmapped_snps": unmapped_snps,
        "all_genes": all_genes,
        "model": model,
    }

    if save_outputs:
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
            OUT_DIR / "loss_history" / f"loss_history_{split_name}.csv",
            index=False
        )

        metrics_df = pd.DataFrame([{
            "Split": split_name,
            "Model": MODEL_NAME,
            "RMSE": rmse,
            "MAE": mae,
            "r2": r2,
            "r": r,
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "best_val_mae": result["best_val_mae"],
            "n_mapped_snps": len(mapped_snps),
            "n_unmapped_snps": len(unmapped_snps),
            "n_genes": len(all_genes),
        }])

        metrics_df.to_csv(
            OUT_DIR / "metrics" / f"metrics_{split_name}.csv",
            index=False
        )

        # Salva il modello per eventuale SHAP/interpretazione successiva.
        model_path = OUT_DIR / "models" / f"model_{split_name}.keras"
        model.save(model_path)

    return result


# =============================================================================
# PARAMS
# =============================================================================

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
# PARTIAL OUTPUTS
# =============================================================================

def save_partial_training_outputs(all_metrics, all_preds):
    metrics_df = pd.DataFrame(all_metrics)

    if len(metrics_df) > 0:
        mean_row = pd.DataFrame([{
            "Split": "Mean",
            "Model": MODEL_NAME,
            "RMSE": metrics_df["RMSE"].mean(),
            "MAE": metrics_df["MAE"].mean(),
            "r2": metrics_df["r2"].mean(),
            "r": metrics_df["r"].mean(),
            "best_epoch": metrics_df["best_epoch"].mean(),
            "best_val_loss": metrics_df["best_val_loss"].mean(),
            "best_val_mae": metrics_df["best_val_mae"].mean(),
            "n_mapped_snps": metrics_df["n_mapped_snps"].mean(),
            "n_unmapped_snps": metrics_df["n_unmapped_snps"].mean(),
            "n_genes": metrics_df["n_genes"].mean(),
        }])

        metrics_out = pd.concat([metrics_df, mean_row], ignore_index=True)

        metrics_out.to_csv(
            OUT_DIR / "metrics" / f"metrics_{TRAIT}_{MODEL_NAME}.csv",
            index=False
        )

    if len(all_preds) > 0:
        preds_out = pd.concat(all_preds, ignore_index=True)

        preds_out.to_csv(
            OUT_DIR / "predictions" / f"predictions_{TRAIT}_{MODEL_NAME}_all_splits.csv",
            index=False
        )


# =============================================================================
# TRAINING
# =============================================================================

def run_training():
    meta, cv, split_cols = load_base_tables()
    params = load_best_params()

    print("\n" + "=" * 80)
    print("Running MODE = train")
    print("Model:", MODEL_NAME)
    print("Soil branch: REMOVED")
    print("Using params:")
    print(params)
    print(f"Number of splits: {len(split_cols)}")
    print("=" * 80)

    existing_metrics_file = OUT_DIR / "metrics" / f"metrics_{TRAIT}_{MODEL_NAME}.csv"
    existing_preds_file = OUT_DIR / "predictions" / f"predictions_{TRAIT}_{MODEL_NAME}_all_splits.csv"

    all_metrics = []
    all_preds = []
    completed_splits = set()

    # Resume logic
    if existing_metrics_file.exists():
        old_metrics = pd.read_csv(existing_metrics_file)
        old_metrics = old_metrics[old_metrics["Split"] != "Mean"].copy()

        if len(old_metrics) > 0:
            all_metrics = old_metrics.to_dict(orient="records")
            completed_splits = set(old_metrics["Split"].astype(str).tolist())

            print(f"[RESUME] Existing metrics found: {existing_metrics_file}")
            print(f"[RESUME] Completed splits: {len(completed_splits)}")

    if existing_preds_file.exists():
        old_preds = pd.read_csv(existing_preds_file)

        if len(old_preds) > 0:
            for _, sub in old_preds.groupby("Split"):
                all_preds.append(sub.copy())

            print(f"[RESUME] Existing predictions found: {existing_preds_file}")

    for split_name in split_cols:
        if split_name in completed_splits:
            print(f"[SKIP] Already completed split: {split_name}")
            continue

        print("\n" + "-" * 80)
        print(f"Training final model on split: {split_name}")
        print("-" * 80)

        res = run_one_split(
            split_name=split_name,
            params=params,
            save_outputs=True,
        )

        metric_row = {
            "Split": split_name,
            "Model": MODEL_NAME,
            "RMSE": res["rmse"],
            "MAE": res["mae"],
            "r2": res["r2"],
            "r": res["r"],
            "best_epoch": res["best_epoch"],
            "best_val_loss": res["best_val_loss"],
            "best_val_mae": res["best_val_mae"],
            "n_mapped_snps": len(res["mapped_snps"]),
            "n_unmapped_snps": len(res["unmapped_snps"]),
            "n_genes": len(res["all_genes"]),
        }

        all_metrics.append(metric_row)

        # Build predictions dataframe
        meta_now, cv_now, _ = load_base_tables()
        inner_now = load_inner_splits()

        df = prepare_split_dataframe(meta_now, cv_now, inner_now, split_name)
        test_mask = df["Testing"] == 1

        pred_df = df.loc[
            test_mask,
            ["Envir", "Genotype", TRAIT, "Testing", "Split"]
        ].copy()

        pred_df = pred_df.rename(columns={TRAIT: "Observed"})
        pred_df["Predicted"] = res["y_pred_test"]
        pred_df["Residual"] = pred_df["Observed"] - pred_df["Predicted"]
        pred_df["Model"] = MODEL_NAME

        all_preds.append(pred_df)

        save_partial_training_outputs(all_metrics, all_preds)

        print(
            f"[SPLIT DONE] {split_name} | "
            f"RMSE={res['rmse']:.4f}, "
            f"MAE={res['mae']:.4f}, "
            f"R2={res['r2']:.4f}, "
            f"r={res['r']:.4f}, "
            f"best_epoch={res['best_epoch']}"
        )

    # Final summary
    metrics_df = pd.DataFrame(all_metrics)
    metrics_no_mean = metrics_df[metrics_df["Split"] != "Mean"].copy()

    print("\n" + "=" * 80)
    print("TRAINING FINISHED")
    print("=" * 80)

    if len(metrics_no_mean) > 0:
        print("Final mean metrics across completed splits:")
        print(
            metrics_no_mean[["RMSE", "MAE", "r2", "r", "best_epoch"]]
            .mean()
            .to_string()
        )

        summary_path = OUT_DIR / "metrics" / f"summary_{TRAIT}_{MODEL_NAME}.txt"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("NO-SOIL V3 TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"MODEL_NAME: {MODEL_NAME}\n")
            f.write(f"TRAIT: {TRAIT}\n")
            f.write("Soil branch: REMOVED\n\n")
            f.write("Best params:\n")
            f.write(json.dumps(params, indent=2))
            f.write("\n\n")
            f.write("Mean metrics across completed splits:\n")
            f.write(
                metrics_no_mean[["RMSE", "MAE", "r2", "r", "best_epoch"]]
                .mean()
                .to_string()
            )
            f.write("\n\n")
            f.write("Std metrics across completed splits:\n")
            f.write(
                metrics_no_mean[["RMSE", "MAE", "r2", "r", "best_epoch"]]
                .std(ddof=1)
                .to_string()
            )
            f.write("\n\n")
            f.write("Per-split metrics:\n")
            f.write(metrics_no_mean.to_string(index=False))

        print(f"\nSaved summary:")
        print(summary_path)

    print("\nSaved metrics:")
    print(OUT_DIR / "metrics" / f"metrics_{TRAIT}_{MODEL_NAME}.csv")

    print("\nSaved predictions:")
    print(OUT_DIR / "predictions" / f"predictions_{TRAIT}_{MODEL_NAME}_all_splits.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    make_dirs()
    set_all_seeds(GLOBAL_SEED)

    if MODE == "train":
        run_training()
    else:
        raise ValueError("This script is only for MODE='train'.")


if __name__ == "__main__":
    main()
