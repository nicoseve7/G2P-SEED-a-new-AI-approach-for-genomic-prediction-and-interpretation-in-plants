# -*- coding: utf-8 -*-

################################################################################
### Q4_compute_shap_no_soil_newtraits.py
###
### SHAP computation for V3 no-soil multi-input model
### for new traits:
###   - Acidity
###   - Color_over
###
### Uses already trained/saved .keras models from Q2.
###
### Da eseguire da:
###   dalpaper/nuovitrattinosoil/
################################################################################

import os
import sys
import gc
from pathlib import Path

print("PYTHON:", sys.executable)
print("Starting SHAP computation for V3 no-soil new traits...")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import shap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

MODEL_NAME = "paper4branches_bio_geni_relu_concathidden_dropout_meteoexp_v3_no_soil"

BASE_MODEL_DIR = Path("Output/02_no_soil_model")

NPY_BASE_DIR = Path("Output/Intermediate/numpy_arrays_newtraits")
GENO_BASE_DIR = Path("Output/Intermediate/geno_files")
BIO_BASE_DIR = Path("Output/biologic_objects")

INNER_SPLITS_FILE = Path("Output/datasets/inner_validation_splits_newtraits.csv")

GLOBAL_SEED = 42

# SHAP settings
# Background troppo grande = SHAP molto lento.
# 200 è un compromesso ragionevole. Puoi aumentare se hai tempo/GPU.
MAX_BACKGROUND_SAMPLES = 200

# None = spiega tutto il test set.
# Se diventa troppo lento, puoi mettere ad esempio 500.
MAX_EXPLAIN_SAMPLES = None

RANDOM_SEED_FOR_SHAP_SAMPLING = 42


# =============================================================================
# GPU INFO
# =============================================================================

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
# UTILS
# =============================================================================

def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def clean_genotype(x):
    return str(x).replace("G_", "").strip()


def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File non trovato:\n{path}")


def atomic_save_npy(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(path) + ".tmp.npy"
    np.save(tmp_path, array)
    os.replace(tmp_path, path)


def natural_split_sort_key(split_name: str):
    try:
        cv_part, split_part = str(split_name).split("_")
        cv_num = int(cv_part.replace("CV", ""))
        split_num = int(split_part.replace("Split", ""))
        return cv_num, split_num
    except Exception:
        return 999, 999


def sample_rows(X, max_n, seed=42):
    """
    X can be numpy array.
    Returns sampled rows.
    """
    if max_n is None:
        return X

    n = X.shape[0]

    if n <= max_n:
        return X

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx = np.sort(idx)

    return X[idx]


def sample_multiinput(inputs, max_n, seed=42):
    """
    Applies same row sampling to all input arrays.
    """
    if max_n is None:
        return inputs

    n = inputs[0].shape[0]

    if n <= max_n:
        return inputs

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx = np.sort(idx)

    return [x[idx] for x in inputs]


# =============================================================================
# CUSTOM LAYER
# =============================================================================

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
# SHAP NORMALIZATION
# =============================================================================

def normalize_multiinput_shap_output(shap_values):
    """
    Expected for multi-input single-output model:
        list with one item per input branch

    For no-soil model:
        [weather, pca, mapped, unmapped]

    Some SHAP/TensorFlow versions return arrays with final singleton output dim:
        (n_samples, n_features, 1)

    This function converts each branch to:
        (n_samples, n_features)
    """
    if not isinstance(shap_values, list):
        raise ValueError(
            f"Expected SHAP output as list for multi-input model, got {type(shap_values)}"
        )

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
# PATHS PER TRAIT
# =============================================================================

def get_trait_paths(trait: str):
    model_out_dir = BASE_MODEL_DIR / trait

    paths = {
        "model_out_dir": model_out_dir,
        "model_dir": model_out_dir / "models",
        "save_arrays_dir": model_out_dir / "Interpretation" / "SHAP_arrays",
        "save_tables_dir": model_out_dir / "Interpretation" / "Splitwise_tables",
        "save_reports_dir": model_out_dir / "Interpretation" / "Reports",
        "meta_file": NPY_BASE_DIR / trait / f"sample_metadata_{trait}.csv",
        "pca_file": NPY_BASE_DIR / trait / "pca.npy",
        "weather_file": NPY_BASE_DIR / trait / "weather_period_features_v3.npy",
        "weather_cols_file": NPY_BASE_DIR / trait / "weather_period_features_v3_columns.csv",
        "geno_dir": GENO_BASE_DIR / trait,
        "split_inputs_dir": BIO_BASE_DIR / trait / "split_inputs",
    }

    for key in ["save_arrays_dir", "save_tables_dir", "save_reports_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths


def find_model_path(model_dir: Path, trait: str, split: str):
    """
    Robust perché in alcuni script potremmo avere:
        model_Acidity_CV1_Split1.keras
    oppure:
        model_CV1_Split1.keras
    """
    candidates = [
        model_dir / f"model_{trait}_{split}.keras",
        model_dir / f"model_{split}.keras",
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


# =============================================================================
# LOAD INNER SPLITS
# =============================================================================

def load_inner_all():
    check_file(INNER_SPLITS_FILE)

    inner = pd.read_csv(INNER_SPLITS_FILE, low_memory=False)

    required = {"Trait", "Envir", "Genotype", "ID_key"}
    missing = required - set(inner.columns)

    if missing:
        raise ValueError(
            f"Nel file inner split mancano colonne base: {missing}\n"
            f"File: {INNER_SPLITS_FILE}\n"
            f"Colonne trovate: {inner.columns.tolist()[:30]}"
        )

    inner["Trait"] = inner["Trait"].astype(str).str.strip()
    inner["Envir"] = inner["Envir"].astype(str).str.strip()
    inner["Genotype"] = inner["Genotype"].apply(clean_genotype)
    inner["ID_key"] = inner["Envir"] + "-" + inner["Genotype"]

    return inner


def get_split_cols_for_trait(inner_trait: pd.DataFrame, trait: str):
    prefix = f"{trait}_"
    suffix = "_Testing"

    split_cols = []

    for c in inner_trait.columns:
        if c.startswith(prefix) and c.endswith(suffix):
            split = c.replace(prefix, "").replace(suffix, "")
            split_cols.append(split)

    split_cols = sorted(set(split_cols), key=natural_split_sort_key)

    if len(split_cols) == 0:
        raise ValueError(
            f"Nessuno split trovato per il trait {trait} nel file inner.\n"
            f"Mi aspettavo colonne tipo {trait}_CV1_Split1_Testing"
        )

    return split_cols


def load_trait_base_tables(trait: str, paths: dict, inner_all: pd.DataFrame):
    check_file(paths["meta_file"])

    meta = pd.read_csv(paths["meta_file"])

    needed = {"Genotype", "Envir", trait}
    missing = needed - set(meta.columns)

    if missing:
        raise ValueError(
            f"Nel metadata del trait {trait} mancano colonne: {missing}\n"
            f"File: {paths['meta_file']}\n"
            f"Colonne trovate: {meta.columns.tolist()}"
        )

    meta = meta[["Genotype", "Envir", trait]].copy()
    meta["Genotype"] = meta["Genotype"].apply(clean_genotype)
    meta["Envir"] = meta["Envir"].astype(str).str.strip()
    meta[trait] = pd.to_numeric(meta[trait], errors="coerce")
    meta["ID_key"] = meta["Envir"] + "-" + meta["Genotype"]

    inner_trait = inner_all[inner_all["Trait"] == trait].copy()

    if inner_trait.empty:
        raise ValueError(f"Nessuna riga inner trovata per trait {trait}.")

    split_cols = get_split_cols_for_trait(inner_trait, trait)

    # Keep only useful inner columns.
    keep_cols = ["Trait", "Envir", "Genotype", "ID_key"]

    for split in split_cols:
        keep_cols.extend([
            f"{trait}_{split}_Testing",
            f"{trait}_{split}_Validation",
            f"{trait}_{split}_Subtrain",
            f"{trait}_{split}_role",
        ])

    missing_cols = [c for c in keep_cols if c not in inner_trait.columns]

    if missing_cols:
        raise ValueError(
            f"Nel file inner mancano colonne richieste per {trait}.\n"
            f"Esempi mancanti: {missing_cols[:20]}"
        )

    inner_trait = inner_trait[keep_cols].copy()

    # Align inner to meta.
    merged = meta.merge(
        inner_trait,
        on=["Trait", "Envir", "Genotype", "ID_key"] if "Trait" in meta.columns else ["Envir", "Genotype", "ID_key"],
        how="left"
    )

    # La merge sopra con "Trait" non funziona perché meta non ha colonna Trait.
    # Quindi rifacciamo in modo esplicito e sicuro.
    inner_trait_for_merge = inner_trait.drop(columns=["Trait"]).copy()

    merged = meta.merge(
        inner_trait_for_merge,
        on=["Envir", "Genotype", "ID_key"],
        how="left",
        validate="one_to_one"
    )

    if merged.shape[0] != meta.shape[0]:
        raise ValueError(
            f"Merge meta-inner ha cambiato numero righe per {trait}: "
            f"meta={meta.shape[0]}, merged={merged.shape[0]}"
        )

    # Check at least testing column not missing.
    first_test_col = f"{trait}_{split_cols[0]}_Testing"
    missing_inner_rows = int(merged[first_test_col].isna().sum())

    if missing_inner_rows > 0:
        examples = (
            merged.loc[merged[first_test_col].isna(), "ID_key"]
            .head(20)
            .tolist()
        )

        raise ValueError(
            f"Per il trait {trait}, alcune righe metadata non sono presenti nell'inner split.\n"
            f"Righe mancanti: {missing_inner_rows}\n"
            f"Esempi ID_key mancanti: {examples}"
        )

    return merged, split_cols


# =============================================================================
# LOAD ARRAYS / FEATURE NAMES
# =============================================================================

def load_shared_arrays_and_names(trait: str, paths: dict):
    check_file(paths["pca_file"])
    check_file(paths["weather_file"])
    check_file(paths["weather_cols_file"])

    pca_all = np.load(paths["pca_file"]).astype("float32")
    weather_all = np.load(paths["weather_file"]).astype("float32")

    weather_cols = (
        pd.read_csv(paths["weather_cols_file"])["feature_name"]
        .astype(str)
        .tolist()
    )

    if weather_all.shape[1] != len(weather_cols):
        raise ValueError(
            f"Weather columns mismatch for {trait}: "
            f"weather array has {weather_all.shape[1]} columns, "
            f"columns file has {len(weather_cols)} names."
        )

    pca_cols = [f"PC{i+1}" for i in range(pca_all.shape[1])]

    return weather_all, pca_all, weather_cols, pca_cols


def load_split_objects(paths: dict, split_name: str):
    split_dir = paths["split_inputs_dir"] / split_name

    needed = [
        split_dir / "mapped_snps.csv",
        split_dir / "unmapped_snps.csv",
        split_dir / "all_genes.csv",
        split_dir / "snp_to_gene_edges.csv",
    ]

    for p in needed:
        check_file(p)

    mapped_snps = pd.read_csv(split_dir / "mapped_snps.csv")["SNP"].astype(str).tolist()
    unmapped_snps = pd.read_csv(split_dir / "unmapped_snps.csv")["SNP"].astype(str).tolist()
    all_genes = pd.read_csv(split_dir / "all_genes.csv")["Gene"].astype(str).tolist()
    edges = pd.read_csv(split_dir / "snp_to_gene_edges.csv", dtype=str)

    return mapped_snps, unmapped_snps, all_genes, edges


def prepare_split_dataframe(
    trait: str,
    meta_inner: pd.DataFrame,
    paths: dict,
    split_name: str
):
    geno_file = paths["geno_dir"] / f"geno_{split_name}.csv"
    check_file(geno_file)

    Xgeno = pd.read_csv(geno_file)

    first_col = Xgeno.columns[0]
    Xgeno[first_col] = Xgeno[first_col].apply(clean_genotype)
    Xgeno = Xgeno.rename(columns={first_col: "Genotype"})
    Xgeno["Genotype"] = Xgeno["Genotype"].astype(str).str.strip()

    df = meta_inner.merge(
        Xgeno,
        on="Genotype",
        how="left",
        validate="many_to_one"
    )

    if df.shape[0] != meta_inner.shape[0]:
        raise ValueError(
            f"Merge geno ha cambiato righe per {trait} {split_name}: "
            f"meta_inner={meta_inner.shape[0]}, df={df.shape[0]}"
        )

    df["Split"] = split_name

    return df


# =============================================================================
# SPLIT DATA
# =============================================================================

def get_split_data(
    trait: str,
    paths: dict,
    meta_inner: pd.DataFrame,
    weather_all: np.ndarray,
    pca_all: np.ndarray,
    weather_cols,
    pca_cols,
    split_name: str,
):
    df = prepare_split_dataframe(
        trait=trait,
        meta_inner=meta_inner,
        paths=paths,
        split_name=split_name
    )

    mapped_snps, unmapped_snps, all_genes, edges = load_split_objects(paths, split_name)

    X_mapped = df[mapped_snps].apply(pd.to_numeric, errors="coerce")
    X_unmapped = df[unmapped_snps].apply(pd.to_numeric, errors="coerce")

    subtrain_col = f"{trait}_{split_name}_Subtrain"
    validation_col = f"{trait}_{split_name}_Validation"
    testing_col = f"{trait}_{split_name}_Testing"
    role_col = f"{trait}_{split_name}_role"

    for c in [subtrain_col, validation_col, testing_col, role_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} for {trait} {split_name}")

    subtrain_mask = pd.to_numeric(df[subtrain_col], errors="coerce").fillna(0).astype(int) == 1
    test_mask = pd.to_numeric(df[testing_col], errors="coerce").fillna(0).astype(int) == 1

    sub_idx = df.index[subtrain_mask]
    test_idx = df.index[test_mask]

    if len(sub_idx) == 0:
        raise ValueError(f"No subtrain samples for {trait} {split_name}")

    if len(test_idx) == 0:
        raise ValueError(f"No test samples for {trait} {split_name}")

    # SNP missing imputation: subtrain means only.
    mapped_means = X_mapped.loc[sub_idx].mean()
    unmapped_means = X_unmapped.loc[sub_idx].mean()

    X_mapped_sub = X_mapped.loc[sub_idx].fillna(mapped_means)
    X_mapped_test = X_mapped.loc[test_idx].fillna(mapped_means)

    X_unmapped_sub = X_unmapped.loc[sub_idx].fillna(unmapped_means)
    X_unmapped_test = X_unmapped.loc[test_idx].fillna(unmapped_means)

    # Scaling: subtrain only, same logic as Q2 training.
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

    background = [
        weather_sub,
        pca_sub,
        mapped_sub,
        unmapped_sub,
    ]

    test_inputs = [
        weather_test,
        pca_test,
        mapped_test,
        unmapped_test,
    ]

    background = sample_multiinput(
        background,
        max_n=MAX_BACKGROUND_SAMPLES,
        seed=RANDOM_SEED_FOR_SHAP_SAMPLING
    )

    test_inputs = sample_multiinput(
        test_inputs,
        max_n=MAX_EXPLAIN_SAMPLES,
        seed=RANDOM_SEED_FOR_SHAP_SAMPLING
    )

    # Need matching metadata if test was subsampled.
    test_meta = df.loc[test_idx, ["Envir", "Genotype", trait, role_col, testing_col, "Split"]].copy()

    if MAX_EXPLAIN_SAMPLES is not None and test_meta.shape[0] > MAX_EXPLAIN_SAMPLES:
        rng = np.random.default_rng(RANDOM_SEED_FOR_SHAP_SAMPLING)
        sampled_pos = rng.choice(test_meta.shape[0], size=MAX_EXPLAIN_SAMPLES, replace=False)
        sampled_pos = np.sort(sampled_pos)
        test_meta = test_meta.iloc[sampled_pos].copy()

    test_meta = test_meta.rename(columns={
        trait: "Observed",
        role_col: "Role",
        testing_col: "Testing"
    })

    return {
        "background": background,
        "test_inputs": test_inputs,
        "test_metadata": test_meta,
        "weather_cols": weather_cols,
        "pca_cols": pca_cols,
        "mapped_cols": mapped_snps,
        "unmapped_cols": unmapped_snps,
        "all_genes": all_genes,
        "edges": edges,
        "n_subtrain": len(sub_idx),
        "n_test_total": len(test_idx),
        "n_test_explained": test_inputs[0].shape[0],
    }


# =============================================================================
# SAVE SPLITWISE TABLES
# =============================================================================

def save_mean_abs(arr, names, filename: Path, id_col: str, trait: str, split: str):
    mean_abs = np.mean(np.abs(arr), axis=0)

    df = pd.DataFrame({
        "Trait": trait,
        "Split": split,
        id_col: names,
        "mean_abs_SHAP": mean_abs
    }).sort_values(
        "mean_abs_SHAP",
        ascending=False
    ).reset_index(drop=True)

    df["Rank"] = np.arange(1, len(df) + 1)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)

    return df


# =============================================================================
# MAIN PER TRAIT
# =============================================================================

def process_one_trait(trait: str, inner_all: pd.DataFrame):
    print("\n" + "#" * 100)
    print(f"Q4 - SHAP for trait: {trait}")
    print("#" * 100)

    paths = get_trait_paths(trait)

    meta_inner, split_cols = load_trait_base_tables(trait, paths, inner_all)
    weather_all, pca_all, weather_cols, pca_cols = load_shared_arrays_and_names(trait, paths)

    if weather_all.shape[0] != meta_inner.shape[0]:
        raise ValueError(
            f"Weather rows mismatch for {trait}: "
            f"weather={weather_all.shape[0]}, metadata={meta_inner.shape[0]}"
        )

    if pca_all.shape[0] != meta_inner.shape[0]:
        raise ValueError(
            f"PCA rows mismatch for {trait}: "
            f"pca={pca_all.shape[0]}, metadata={meta_inner.shape[0]}"
        )

    print(f"Metadata rows: {meta_inner.shape[0]}")
    print(f"Weather shape: {weather_all.shape}")
    print(f"PCA shape: {pca_all.shape}")
    print(f"Splits: {len(split_cols)}")

    custom_objects = {"MaskedDense": MaskedDense}

    summary_rows = []

    for split in split_cols:
        print("\n" + "=" * 80)
        print(f"[{trait}] Processing SHAP for {split}")
        print("=" * 80)

        save_arrays_dir = paths["save_arrays_dir"]
        save_tables_dir = paths["save_tables_dir"]

        out_weather = save_arrays_dir / f"{trait}_{split}_SHAP_weather.npy"
        out_pca = save_arrays_dir / f"{trait}_{split}_SHAP_pca.npy"
        out_mapped = save_arrays_dir / f"{trait}_{split}_SHAP_mapped.npy"
        out_unmapped = save_arrays_dir / f"{trait}_{split}_SHAP_unmapped.npy"

        out_branch_csv = save_tables_dir / f"{trait}_{split}_branch_importance.csv"
        out_allsnp_csv = save_tables_dir / f"{trait}_{split}_all_snp_mean_abs_SHAP.csv"
        out_test_meta_csv = save_tables_dir / f"{trait}_{split}_test_metadata_SHAP_samples.csv"

        expected_outputs = [
            out_weather,
            out_pca,
            out_mapped,
            out_unmapped,
            out_branch_csv,
            out_allsnp_csv,
            out_test_meta_csv,
        ]

        if all(p.exists() for p in expected_outputs):
            print(f"Outputs already exist for {trait} {split}, skipping.")

            summary_rows.append({
                "Trait": trait,
                "Split": split,
                "status": "skipped_existing",
                "model_path": str(find_model_path(paths["model_dir"], trait, split)),
                "n_subtrain": np.nan,
                "n_test_total": np.nan,
                "n_test_explained": np.nan,
            })

            continue

        model_path = find_model_path(paths["model_dir"], trait, split)

        if not model_path.exists():
            print(f"[WARNING] Model not found for {trait} {split}, skipping.")
            print(model_path)

            summary_rows.append({
                "Trait": trait,
                "Split": split,
                "status": "model_missing",
                "model_path": str(model_path),
                "n_subtrain": np.nan,
                "n_test_total": np.nan,
                "n_test_explained": np.nan,
            })

            continue

        split_data = get_split_data(
            trait=trait,
            paths=paths,
            meta_inner=meta_inner,
            weather_all=weather_all,
            pca_all=pca_all,
            weather_cols=weather_cols,
            pca_cols=pca_cols,
            split_name=split,
        )

        background = split_data["background"]
        test_inputs = split_data["test_inputs"]

        tf.keras.backend.clear_session()
        gc.collect()

        print("Loading model:", model_path)
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )

        print("Model input names/shapes:")
        for inp in model.inputs:
            print(" -", inp.name, inp.shape)

        print("Background shapes:")
        for x in background:
            print(" ", x.shape)

        print("Test input shapes:")
        for x in test_inputs:
            print(" ", x.shape)

        print("Building SHAP GradientExplainer...")
        explainer = shap.GradientExplainer(model, background)

        print("Computing SHAP values...")
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

        # Save test metadata corresponding to explained samples
        split_data["test_metadata"].to_csv(out_test_meta_csv, index=False)

        # Splitwise feature tables
        weather_df = save_mean_abs(
            shap_weather,
            split_data["weather_cols"],
            save_tables_dir / f"{trait}_{split}_weather_mean_abs_SHAP.csv",
            "Feature",
            trait,
            split
        )

        pca_df = save_mean_abs(
            shap_pca,
            split_data["pca_cols"],
            save_tables_dir / f"{trait}_{split}_pca_mean_abs_SHAP.csv",
            "Feature",
            trait,
            split
        )

        mapped_df = save_mean_abs(
            shap_mapped,
            split_data["mapped_cols"],
            save_tables_dir / f"{trait}_{split}_mapped_snp_mean_abs_SHAP.csv",
            "SNP",
            trait,
            split
        )
        mapped_df["MappedStatus"] = "mapped"

        unmapped_df = save_mean_abs(
            shap_unmapped,
            split_data["unmapped_cols"],
            save_tables_dir / f"{trait}_{split}_unmapped_snp_mean_abs_SHAP.csv",
            "SNP",
            trait,
            split
        )
        unmapped_df["MappedStatus"] = "unmapped"

        # All SNP table
        all_snp_df = (
            pd.concat([mapped_df, unmapped_df], ignore_index=True)
            .sort_values("mean_abs_SHAP", ascending=False)
            .reset_index(drop=True)
        )

        all_snp_df["Rank"] = np.arange(1, len(all_snp_df) + 1)
        all_snp_df.to_csv(out_allsnp_csv, index=False)

        # Branch importance WITHOUT soil
        branch_df = pd.DataFrame({
            "Trait": [trait] * 4,
            "Split": [split] * 4,
            "Branch": [
                "Weather",
                "PCA",
                "Mapped_SNP",
                "Unmapped_SNP",
            ],
            "mean_abs_SHAP": [
                float(np.mean(np.abs(shap_weather))),
                float(np.mean(np.abs(shap_pca))),
                float(np.mean(np.abs(shap_mapped))),
                float(np.mean(np.abs(shap_unmapped))),
            ],
            "n_features": [
                shap_weather.shape[1],
                shap_pca.shape[1],
                shap_mapped.shape[1],
                shap_unmapped.shape[1],
            ],
        })

        branch_df.to_csv(out_branch_csv, index=False)

        summary_rows.append({
            "Trait": trait,
            "Split": split,
            "status": "ok",
            "model_path": str(model_path),
            "n_subtrain": split_data["n_subtrain"],
            "n_test_total": split_data["n_test_total"],
            "n_test_explained": split_data["n_test_explained"],
            "n_weather_features": shap_weather.shape[1],
            "n_pca_features": shap_pca.shape[1],
            "n_mapped_snps": shap_mapped.shape[1],
            "n_unmapped_snps": shap_unmapped.shape[1],
            "n_total_snps": shap_mapped.shape[1] + shap_unmapped.shape[1],
        })

        print(f"Saved SHAP outputs for {trait} {split}")

        del model, explainer, shap_values
        del shap_weather, shap_pca, shap_mapped, shap_unmapped
        tf.keras.backend.clear_session()
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    summary_file = paths["save_reports_dir"] / f"Q4_SHAP_compute_summary_{trait}.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\nSaved trait SHAP summary:")
    print(summary_file)

    return summary_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_all_seeds(GLOBAL_SEED)

    print("\n" + "=" * 100)
    print("Q4 - COMPUTE SHAP NO-SOIL V3 FOR NEW TRAITS")
    print("=" * 100)
    print(f"Traits: {TRAITS}")
    print(f"MAX_BACKGROUND_SAMPLES: {MAX_BACKGROUND_SAMPLES}")
    print(f"MAX_EXPLAIN_SAMPLES: {MAX_EXPLAIN_SAMPLES}")
    print("=" * 100)

    inner_all = load_inner_all()

    all_summaries = []

    for trait in TRAITS:
        summary_df = process_one_trait(trait, inner_all)
        all_summaries.append(summary_df)

    final_summary = pd.concat(all_summaries, ignore_index=True)

    final_summary_file = BASE_MODEL_DIR / "Q4_SHAP_compute_summary_all_traits.csv"
    final_summary_file.parent.mkdir(parents=True, exist_ok=True)
    final_summary.to_csv(final_summary_file, index=False)

    print("\n" + "=" * 100)
    print("Q4 SHAP computation completed.")
    print("=" * 100)
    print(final_summary.to_string(index=False))
    print("\nSaved:")
    print(final_summary_file)


if __name__ == "__main__":
    main()