# -*- coding: utf-8 -*-

################################################################################
### Q1_tune_no_soil_newtraits.py
###
### Tuning della rete V3 SENZA ramo suolo per:
###   - Acidity
###   - Color_over
###
### Architettura:
###   Weather V3 expanded branch
###   PCA branch
###   mapped SNP -> gene -> ReLU branch
###   unmapped SNP -> hidden branch
###   concatenation -> fusion hidden -> output
###
### Input:
###   Output/Intermediate/numpy_arrays_newtraits/<Trait>/
###       sample_metadata_<Trait>.csv
###       pca.npy
###       weather_period_features_v3.npy
###
###   Output/Intermediate/geno_files/<Trait>/geno_CV*_Split*.csv
###
###   Output/biologic_objects/<Trait>/split_inputs/<Split>/
###       mapped_snps.csv
###       unmapped_snps.csv
###       all_genes.csv
###       snp_to_gene_edges.csv
###
### Output:
###   Output/02_no_soil_model/<Trait>/tuning/
################################################################################

import os
import json
import math
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"

BASE_OUT_DIR = Path("Output/02_no_soil_model")

NPY_BASE_DIR = Path("Output/Intermediate/numpy_arrays_newtraits")
GENO_BASE_DIR = Path("Output/Intermediate/geno_files")
BIO_BASE_DIR = Path("Output/biologic_objects")

CV_FILE = Path("Input/CV1_Strategy/Harvest_date_CV.csv")

# Prima prova a usare inner splits già costruiti nella pipeline senza_suolo.
# Se non li trova, lo script crea inner split deterministici dentro il train.
INNER_SPLITS_CANDIDATES = [
    Path("Output/datasets/inner_validation_splits_newtraits.csv"),
    Path("../senza_suolo/Output/datasets/inner_validation_splits_harvest.csv"),
    Path("../Output/datasets/inner_validation_splits_harvest.csv"),
]

GLOBAL_SEED = 42

# Stessa logica di tuning della rete senza_suolo.
# Se è troppo lento, puoi ridurre questa griglia.
GRID = {
    "learning_rate": [0.001, 0.0005],
    "l2_lambda": [0.0, 1e-5, 1e-4],
    "fusion_hidden_units": [16, 32, 64],
    "dropout_rate": [0.1, 0.2, 0.3],
}

UNMAPPED_HIDDEN_UNITS = 16
BIO_HIDDEN_UNITS = 8

MAX_EPOCHS = 500
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 20
SCALE_TARGET = True

TUNING_SPLITS = [
    "CV1_Split1",
    "CV2_Split2",
    "CV3_Split3",
    "CV4_Split4",
    "CV5_Split5",
]

FALLBACK_PARAMS = {
    "learning_rate": 0.001,
    "l2_lambda": 0.0,
    "fusion_hidden_units": 32,
    "dropout_rate": 0.2,
}


# =============================================================================
# BASIC UTILS
# =============================================================================

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


def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File non trovato:\n{path}")


def make_trait_dirs(trait: str):
    out_dir = BASE_OUT_DIR / trait

    subdirs = [
        out_dir,
        out_dir / "datasets",
        out_dir / "tuning",
        out_dir / "models",
        out_dir / "predictions",
        out_dir / "metrics",
        out_dir / "loss_history",
        out_dir / "grafici" / "per_split",
        out_dir / "grafici" / "summary",
    ]

    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)

    return out_dir


# =============================================================================
# DATA LOADING
# =============================================================================

def load_base_tables(trait: str):
    meta_file = NPY_BASE_DIR / trait / f"sample_metadata_{trait}.csv"
    check_file(meta_file)
    check_file(CV_FILE)

    meta = pd.read_csv(meta_file)
    meta = meta[["Envir", "Genotype", trait]].copy()

    meta["Envir"] = meta["Envir"].astype(str).str.strip()
    meta["Genotype"] = meta["Genotype"].astype(str).str.replace("^G_", "", regex=True).str.strip()
    meta["ID_key"] = meta["Envir"] + "-" + meta["Genotype"]

    cv = pd.read_csv(CV_FILE)
    cv["Envir"] = cv["Envir"].astype(str).str.strip()
    cv["Genotype"] = cv["Genotype"].astype(str).str.replace("^G_", "", regex=True).str.strip()
    cv["ID_key"] = cv["Envir"] + "-" + cv["Genotype"]

    split_cols = [c for c in cv.columns if c.startswith("CV")]

    # Allinea la CV alle righe del fenotipo del trait.
    cv_indexed = cv.set_index("ID_key")

    missing_keys = sorted(set(meta["ID_key"]) - set(cv_indexed.index))
    if len(missing_keys) > 0:
        raise ValueError(
            f"Nel trait {trait}, alcune righe non sono presenti nella CV strategy.\n"
            f"Esempi ID_key mancanti: {missing_keys[:20]}"
        )

    cv_aligned = cv_indexed.loc[meta["ID_key"]].reset_index()

    if cv_aligned.shape[0] != meta.shape[0]:
        raise ValueError(
            f"CV alignment mismatch per {trait}: "
            f"meta={meta.shape[0]}, cv_aligned={cv_aligned.shape[0]}"
        )

    return meta, cv_aligned, split_cols


def find_existing_inner_splits_file():
    for path in INNER_SPLITS_CANDIDATES:
        if path.exists():
            return path
    return None


def build_fallback_inner_splits(meta: pd.DataFrame, cv: pd.DataFrame, split_cols):
    """
    Fallback se non troviamo il file inner_validation_splits_harvest.csv.

    Per ogni split:
      - Testing = CV == 1
      - tra i non-test, crea Validation deterministica circa 20%
      - il resto è Subtrain

    Nota: se esiste il file inner splits vecchio, è meglio usare quello,
    perché mantiene più continuità con la pipeline precedente.
    """
    print("[WARNING] Nessun file inner split esistente trovato.")
    print("[WARNING] Creo inner validation split deterministici dentro Q1.")

    inner = meta[["Envir", "Genotype", "ID_key"]].copy()

    rng_base = np.random.default_rng(GLOBAL_SEED)

    for split_name in split_cols:
        testing = cv[split_name].values.astype(int)

        train_idx = np.where(testing == 0)[0]
        test_idx = np.where(testing == 1)[0]

        rng = np.random.default_rng(GLOBAL_SEED + abs(hash(split_name)) % 100000)
        shuffled_train = train_idx.copy()
        rng.shuffle(shuffled_train)

        n_val = max(1, int(round(0.20 * len(shuffled_train))))
        val_idx = set(shuffled_train[:n_val].tolist())
        train_idx_set = set(train_idx.tolist())

        validation = np.array([1 if i in val_idx else 0 for i in range(len(meta))], dtype=int)
        subtrain = np.array([1 if (i in train_idx_set and i not in val_idx) else 0 for i in range(len(meta))], dtype=int)

        inner[f"{split_name}_Testing"] = testing
        inner[f"{split_name}_Validation"] = validation
        inner[f"{split_name}_Subtrain"] = subtrain

        role = []
        for i in range(len(meta)):
            if testing[i] == 1:
                role.append("Testing")
            elif validation[i] == 1:
                role.append("Validation")
            elif subtrain[i] == 1:
                role.append("Subtrain")
            else:
                role.append("Unused")

        inner[f"{split_name}_role"] = role

    return inner


# def load_inner_splits_for_trait(trait: str, meta: pd.DataFrame, cv: pd.DataFrame, split_cols):
#     inner_file = find_existing_inner_splits_file()

#     if inner_file is None:
#         return build_fallback_inner_splits(meta, cv, split_cols)

#     print(f"[INFO] Uso inner split file: {inner_file}")

#     inner_all = pd.read_csv(inner_file)

#     if "ID_key" not in inner_all.columns:
#         if {"Envir", "Genotype"}.issubset(inner_all.columns):
#             inner_all["Envir"] = inner_all["Envir"].astype(str).str.strip()
#             inner_all["Genotype"] = inner_all["Genotype"].astype(str).str.replace("^G_", "", regex=True).str.strip()
#             inner_all["ID_key"] = inner_all["Envir"] + "-" + inner_all["Genotype"]
#         else:
#             raise ValueError(
#                 f"Il file inner split non contiene ID_key né Envir/Genotype:\n{inner_file}"
#             )

#     inner_all["Envir"] = inner_all["Envir"].astype(str).str.strip()
#     inner_all["Genotype"] = inner_all["Genotype"].astype(str).str.replace("^G_", "", regex=True).str.strip()
#     inner_all["ID_key"] = inner_all["ID_key"].astype(str).str.strip()

    # needed_cols = ["Envir", "Genotype", "ID_key"]

    # for split_name in split_cols:
    #     needed_cols.extend([
    #         f"{split_name}_role",
    #         f"{split_name}_Testing",
    #         f"{split_name}_Validation",
    #         f"{split_name}_Subtrain",
    #     ])

    # missing_cols = [c for c in needed_cols if c not in inner_all.columns]

    # if len(missing_cols) > 0:
    #     raise ValueError(
    #         f"Nel file inner split mancano colonne richieste.\n"
    #         f"Esempi mancanti: {missing_cols[:20]}\n"
    #         f"File: {inner_file}"
    #     )

    # inner_indexed = inner_all[needed_cols].drop_duplicates(subset=["ID_key"]).set_index("ID_key")

    # missing_keys = sorted(set(meta["ID_key"]) - set(inner_indexed.index))

    # if len(missing_keys) > 0:
    #     raise ValueError(
    #         f"Per il trait {trait}, alcune righe non hanno inner split.\n"
    #         f"Esempi ID_key mancanti: {missing_keys[:20]}\n"
    #         f"File usato: {inner_file}"
    #     )

    # inner = inner_indexed.loc[meta["ID_key"]].reset_index()

    # if inner.shape[0] != meta.shape[0]:
    #     raise ValueError(
    #         f"Inner split alignment mismatch per {trait}: "
    #         f"meta={meta.shape[0]}, inner={inner.shape[0]}"
    #     )

    # return inner
def load_inner_splits_for_trait(trait: str, meta: pd.DataFrame, cv: pd.DataFrame, split_cols):
    """
    Load trait-specific inner validation split file.

    The file produced by Q0b has columns like:
        Acidity_CV1_Split1_Testing
        Acidity_CV1_Split1_Validation
        Acidity_CV1_Split1_Subtrain
        Acidity_CV1_Split1_role

    This function:
      1. filters rows for the requested trait
      2. keeps only that trait's columns
      3. renames them to the format expected by the rest of Q1:
            CV1_Split1_Testing
            CV1_Split1_Validation
            CV1_Split1_Subtrain
            CV1_Split1_role
    """

    inner_file = Path("Output") / "datasets" / "inner_validation_splits_newtraits.csv"

    if not inner_file.exists():
        raise FileNotFoundError(
            f"Inner validation split file non trovato:\n{inner_file}\n"
            "Devi prima eseguire Q0b_create_inner_validation_splits_newtraits.py"
        )

    print(f"[INFO] Uso inner split file: {inner_file}")

    inner_all = pd.read_csv(inner_file, low_memory=False)

    required_base = ["Trait", "Envir", "Genotype", "ID_key"]
    missing_base = [c for c in required_base if c not in inner_all.columns]

    if missing_base:
        raise ValueError(
            f"Nel file inner split mancano colonne base richieste: {missing_base}\n"
            f"File: {inner_file}\n"
            f"Colonne trovate: {inner_all.columns.tolist()[:80]}"
        )

    inner_all["Trait"] = inner_all["Trait"].astype(str).str.strip()
    inner_all["Envir"] = inner_all["Envir"].astype(str).str.strip()
    inner_all["Genotype"] = (
        inner_all["Genotype"]
        .astype(str)
        .str.replace("^G_", "", regex=True)
        .str.strip()
    )
    inner_all["ID_key"] = inner_all["ID_key"].astype(str).str.strip()

    inner_trait = inner_all[inner_all["Trait"] == trait].copy()

    if inner_trait.empty:
        raise ValueError(
            f"Nessuna riga trovata nel file inner split per trait={trait}.\n"
            f"Trait disponibili: {sorted(inner_all['Trait'].dropna().unique().tolist())}"
        )

    rename_map = {}

    missing_cols = []

    for split in split_cols:
        for suffix in ["Testing", "Validation", "Subtrain", "role"]:
            source_col = f"{trait}_{split}_{suffix}"
            target_col = f"{split}_{suffix}"

            if source_col not in inner_trait.columns:
                missing_cols.append(source_col)
            else:
                rename_map[source_col] = target_col

    if missing_cols:
        raise ValueError(
            f"Nel file inner split mancano colonne trait-specific per {trait}.\n"
            f"Esempi mancanti: {missing_cols[:30]}\n"
            f"File: {inner_file}"
        )

    keep_cols = required_base + list(rename_map.keys())

    inner_trait = inner_trait[keep_cols].rename(columns=rename_map).copy()

    # Converti le colonne 0/1 a numerico.
    for split in split_cols:
        for suffix in ["Testing", "Validation", "Subtrain"]:
            col = f"{split}_{suffix}"
            inner_trait[col] = pd.to_numeric(inner_trait[col], errors="coerce").fillna(0).astype(int)

        role_col = f"{split}_role"
        inner_trait[role_col] = inner_trait[role_col].astype(str).str.strip()

    # Controllo che le righe del meta siano presenti nel file inner.
    meta_keys = set(meta["ID_key"].astype(str))
    inner_keys = set(inner_trait["ID_key"].astype(str))

    missing_meta_keys = sorted(meta_keys - inner_keys)

    if len(missing_meta_keys) > 0:
        raise ValueError(
            f"Nel trait {trait}, alcune righe del metadata non sono presenti nel file inner split.\n"
            f"N missing: {len(missing_meta_keys)}\n"
            f"Esempi ID_key mancanti: {missing_meta_keys[:20]}"
        )

    # Riordina inner_trait nello stesso ordine di meta.
    inner_trait = (
        inner_trait
        .set_index("ID_key")
        .loc[meta["ID_key"].astype(str)]
        .reset_index()
    )

    print(f"[INFO] Inner split loaded for {trait}: {inner_trait.shape}")

    return inner_trait


def load_shared_arrays(trait: str):
    npy_dir = NPY_BASE_DIR / trait

    weather_file = npy_dir / "weather_period_features_v3.npy"
    pca_file = npy_dir / "pca.npy"

    check_file(weather_file)
    check_file(pca_file)

    weather_exp_all = np.load(weather_file).astype("float32")
    pca_all = np.load(pca_file).astype("float32")

    return weather_exp_all, pca_all


def load_split_objects(trait: str, split_name: str):
    split_dir = BIO_BASE_DIR / trait / "split_inputs" / split_name

    mapped_file = split_dir / "mapped_snps.csv"
    unmapped_file = split_dir / "unmapped_snps.csv"
    genes_file = split_dir / "all_genes.csv"
    edges_file = split_dir / "snp_to_gene_edges.csv"

    for f in [mapped_file, unmapped_file, genes_file, edges_file]:
        check_file(f)

    mapped_snps = pd.read_csv(mapped_file)["SNP"].astype(str).tolist()
    unmapped_snps = pd.read_csv(unmapped_file)["SNP"].astype(str).tolist()
    all_genes = pd.read_csv(genes_file)["Gene"].astype(str).tolist()
    snp_to_gene_edges = pd.read_csv(edges_file, dtype=str)

    return {
        "mapped_snps": mapped_snps,
        "unmapped_snps": unmapped_snps,
        "all_genes": all_genes,
        "snp_to_gene_edges": snp_to_gene_edges,
    }


def prepare_split_dataframe(
    trait: str,
    meta: pd.DataFrame,
    cv: pd.DataFrame,
    inner: pd.DataFrame,
    split_name: str,
):
    geno_file = GENO_BASE_DIR / trait / f"geno_{split_name}.csv"
    check_file(geno_file)

    Xgeno = pd.read_csv(geno_file)

    first_col = Xgeno.columns[0]
    Xgeno[first_col] = (
        Xgeno[first_col]
        .astype(str)
        .str.replace("^G_", "", regex=True)
        .str.strip()
    )

    Xgeno = Xgeno.rename(columns={first_col: "Genotype"})
    Xgeno["Genotype"] = Xgeno["Genotype"].astype(str).str.strip()

    df = meta.merge(Xgeno, on="Genotype", how="left", validate="many_to_one")

    if df.shape[0] != meta.shape[0]:
        raise ValueError(
            f"Merge geno ha cambiato righe per {trait}, {split_name}: "
            f"meta={meta.shape[0]}, df={df.shape[0]}"
        )

    df["Testing"] = cv[split_name].values

    inner_cols = [
        "ID_key",
        f"{split_name}_role",
        f"{split_name}_Testing",
        f"{split_name}_Validation",
        f"{split_name}_Subtrain",
    ]

    df = df.merge(inner[inner_cols], on="ID_key", how="left", validate="one_to_one")
    df["Split"] = split_name

    role_cols = [
        f"{split_name}_Testing",
        f"{split_name}_Validation",
        f"{split_name}_Subtrain",
    ]

    if df[role_cols].isna().any().any():
        raise ValueError(
            f"Inner split contiene NA dopo merge per {trait}, {split_name}."
        )

    return df


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
        self.mask_matrix = tf.constant(mask_matrix, dtype=tf.float32)
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

    # Weather branch
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

def run_one_split(
    trait: str,
    out_dir: Path,
    split_name: str,
    params: dict,
    save_outputs: bool = False,
):
    tf.keras.backend.clear_session()
    set_all_seeds(GLOBAL_SEED)

    meta, cv, split_cols = load_base_tables(trait)
    inner = load_inner_splits_for_trait(trait, meta, cv, split_cols)

    weather_exp_all, pca_all = load_shared_arrays(trait)

    if weather_exp_all.shape[0] != meta.shape[0]:
        raise ValueError(
            f"Weather rows mismatch per {trait}: "
            f"weather={weather_exp_all.shape[0]}, meta={meta.shape[0]}"
        )

    if pca_all.shape[0] != meta.shape[0]:
        raise ValueError(
            f"PCA rows mismatch per {trait}: "
            f"pca={pca_all.shape[0]}, meta={meta.shape[0]}"
        )

    split_obj = load_split_objects(trait, split_name)
    df = prepare_split_dataframe(trait, meta, cv, inner, split_name)

    mapped_snps = split_obj["mapped_snps"]
    unmapped_snps = split_obj["unmapped_snps"]
    all_genes = split_obj["all_genes"]
    snp_to_gene_edges = split_obj["snp_to_gene_edges"]

    if len(mapped_snps) == 0:
        raise ValueError(f"Nessuno SNP mapped per {trait}, {split_name}")

    if len(unmapped_snps) == 0:
        raise ValueError(f"Nessuno SNP unmapped per {trait}, {split_name}")

    X_mapped = df[mapped_snps].apply(pd.to_numeric, errors="coerce")
    X_unmapped = df[unmapped_snps].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[trait], errors="coerce")

    subtrain_mask = df[f"{split_name}_Subtrain"] == 1
    val_mask = df[f"{split_name}_Validation"] == 1
    test_mask = df["Testing"] == 1

    subtrain_idx = df.index[subtrain_mask]
    val_idx = df.index[val_mask]
    test_idx = df.index[test_mask]

    if len(subtrain_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Split vuoto per {trait}, {split_name}: "
            f"subtrain={len(subtrain_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

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
        "trait": trait,
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
    }

    return result


# =============================================================================
# TUNING
# =============================================================================

def all_param_configs(grid: dict):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    configs = []

    for combo in product(*values):
        configs.append(dict(zip(keys, combo)))

    return configs


def run_tuning_for_trait(trait: str):
    out_dir = make_trait_dirs(trait)

    tuning_summary_file = out_dir / "tuning" / f"tuning_results_summary_{trait}_no_soil.csv"
    tuning_partial_file = out_dir / "tuning" / f"tuning_results_summary_{trait}_no_soil_partial.csv"
    best_params_file = out_dir / "tuning" / f"best_params_{trait}_no_soil.json"

    configs = all_param_configs(GRID)

    print("\n" + "=" * 80)
    print(f"Running tuning for trait: {trait}")
    print("Model:", MODEL_NAME)
    print("Soil branch: REMOVED")
    print(f"Total tuning configs: {len(configs)}")
    print(f"Tuning splits: {TUNING_SPLITS}")
    print("=" * 80)

    existing_done = set()
    partial_rows = []

    if tuning_partial_file.exists():
        old = pd.read_csv(tuning_partial_file)
        partial_rows = old.to_dict(orient="records")

        for _, row in old.iterrows():
            key = (
                float(row["learning_rate"]),
                float(row["l2_lambda"]),
                int(row["fusion_hidden_units"]),
                float(row["dropout_rate"]),
            )
            existing_done.add(key)

        print(f"[RESUME] Found partial tuning file: {tuning_partial_file}")
        print(f"[RESUME] Completed configs: {len(existing_done)}")

    for i, params in enumerate(configs, start=1):
        key = (
            float(params["learning_rate"]),
            float(params["l2_lambda"]),
            int(params["fusion_hidden_units"]),
            float(params["dropout_rate"]),
        )

        if key in existing_done:
            print(f"\n[SKIP] Already completed config {i}/{len(configs)}: {params}")
            continue

        print("\n" + "-" * 80)
        print(f"[{trait}] Tuning config {i}/{len(configs)}")
        print(params)
        print("-" * 80)

        rows = []
        completed = 0

        for split_name in TUNING_SPLITS:
            try:
                print(f"  Running split: {split_name}")

                res = run_one_split(
                    trait=trait,
                    out_dir=out_dir,
                    split_name=split_name,
                    params=params,
                    save_outputs=False,
                )

                rows.append({
                    "Trait": trait,
                    "Split": split_name,
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
                })

                completed += 1

                print(
                    f"    RMSE={res['rmse']:.4f}, "
                    f"MAE={res['mae']:.4f}, "
                    f"R2={res['r2']:.4f}, "
                    f"r={res['r']:.4f}, "
                    f"best_epoch={res['best_epoch']}"
                )

            except Exception as e:
                print(f"  [FAILED] {trait} {split_name}: {e}")

        if completed == 0:
            summary_row = {
                "Trait": trait,
                "learning_rate": params["learning_rate"],
                "l2_lambda": params["l2_lambda"],
                "fusion_hidden_units": params["fusion_hidden_units"],
                "dropout_rate": params["dropout_rate"],
                "n_completed_splits": 0,
                "mean_best_val_loss": np.nan,
                "sd_best_val_loss": np.nan,
                "mean_best_val_mae": np.nan,
                "mean_RMSE_test": np.nan,
                "sd_RMSE_test": np.nan,
                "mean_MAE_test": np.nan,
                "mean_r2_test": np.nan,
                "mean_r_test": np.nan,
                "mean_best_epoch": np.nan,
                "mean_n_mapped_snps": np.nan,
                "mean_n_unmapped_snps": np.nan,
                "mean_n_genes": np.nan,
            }
        else:
            df = pd.DataFrame(rows)

            summary_row = {
                "Trait": trait,
                "learning_rate": params["learning_rate"],
                "l2_lambda": params["l2_lambda"],
                "fusion_hidden_units": params["fusion_hidden_units"],
                "dropout_rate": params["dropout_rate"],
                "n_completed_splits": completed,
                "mean_best_val_loss": df["best_val_loss"].mean(),
                "sd_best_val_loss": df["best_val_loss"].std(ddof=1) if completed > 1 else 0.0,
                "mean_best_val_mae": df["best_val_mae"].mean(),
                "mean_RMSE_test": df["RMSE"].mean(),
                "sd_RMSE_test": df["RMSE"].std(ddof=1) if completed > 1 else 0.0,
                "mean_MAE_test": df["MAE"].mean(),
                "mean_r2_test": df["r2"].mean(),
                "mean_r_test": df["r"].mean(),
                "mean_best_epoch": df["best_epoch"].mean(),
                "mean_n_mapped_snps": df["n_mapped_snps"].mean(),
                "mean_n_unmapped_snps": df["n_unmapped_snps"].mean(),
                "mean_n_genes": df["n_genes"].mean(),
            }

        partial_rows.append(summary_row)

        pd.DataFrame(partial_rows).to_csv(
            tuning_partial_file,
            index=False
        )

        print(
            f"[CONFIG SUMMARY] {trait} | "
            f"mean_best_val_loss={summary_row['mean_best_val_loss']:.6f}, "
            f"mean_RMSE_test={summary_row['mean_RMSE_test']:.4f}, "
            f"mean_MAE_test={summary_row['mean_MAE_test']:.4f}"
        )

    tuning_df = pd.DataFrame(partial_rows)

    tuning_df = tuning_df.sort_values(
        by=["mean_best_val_loss", "mean_RMSE_test", "mean_MAE_test"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    tuning_df.to_csv(tuning_summary_file, index=False)

    valid = tuning_df[tuning_df["n_completed_splits"] > 0].copy()

    if len(valid) == 0:
        best_params = FALLBACK_PARAMS
    else:
        top = valid.iloc[0]

        best_params = {
            "learning_rate": float(top["learning_rate"]),
            "l2_lambda": float(top["l2_lambda"]),
            "fusion_hidden_units": int(top["fusion_hidden_units"]),
            "dropout_rate": float(top["dropout_rate"]),
        }

    with open(best_params_file, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("\n" + "=" * 80)
    print(f"TUNING FINISHED FOR TRAIT: {trait}")
    print("=" * 80)
    print("Best params selected:")
    print(best_params)
    print(f"\nSaved tuning summary:")
    print(tuning_summary_file)
    print(f"\nSaved best params:")
    print(best_params_file)

    return {
        "Trait": trait,
        **best_params,
        "tuning_summary_file": str(tuning_summary_file),
        "best_params_file": str(best_params_file),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Q1 - TUNE NO-SOIL V3 FOR NEW TRAITS")
    print("=" * 80)

    set_all_seeds(GLOBAL_SEED)

    all_best = []

    for trait in TRAITS:
        best = run_tuning_for_trait(trait)
        all_best.append(best)

    best_summary = pd.DataFrame(all_best)
    best_summary_file = BASE_OUT_DIR / "Q1_best_params_all_traits.csv"
    best_summary.to_csv(best_summary_file, index=False)

    print("\n" + "=" * 80)
    print("Q1 COMPLETED")
    print("=" * 80)
    print(best_summary.to_string(index=False))
    print("\nSaved:")
    print(best_summary_file)


if __name__ == "__main__":
    main()