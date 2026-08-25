# Questo:

# legge sample_metadata_harvest.csv
# legge una versione derivata meteo (v2 o v3)
# allinea le feature meteo a ogni campione
# crea un file numpy 2D:
# weather_period_features_v2.npy
# weather_period_features_v3_splitP2.npy

# -*- coding: utf-8 -*-

################################################################################
### M3_build_numpy_weather_from_period_features.py
### Build sample-aligned numpy weather arrays from derived period features
################################################################################

from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(".")
INPUT_BASE = BASE_DIR / "Input" / "base_files"
INPUT_DERIVED = BASE_DIR / "Input" / "derived"
OUT_DIR = BASE_DIR / "Output" / "numpy_arrays_weather_exp"

OUT_DIR.mkdir(parents=True, exist_ok=True)

META_FILE = INPUT_BASE / "sample_metadata_harvest.csv"

DERIVED_FILES = {
    "v2": INPUT_DERIVED / "weather_period_features_v2.csv",
    "v3": INPUT_DERIVED / "weather_period_features_v3_splitP2.csv",
}


def process_one_version(version_name: str, feature_file: Path):
    print(f"\n=== Processing weather version: {version_name} ===")

    if not feature_file.exists():
        raise FileNotFoundError(f"Missing derived weather feature file: {feature_file}")

    meta = pd.read_csv(META_FILE)
    feat = pd.read_csv(feature_file)

    meta["Envir"] = meta["Envir"].astype(str).str.strip()
    feat["Envir"] = feat["Envir"].astype(str).str.strip()

    merged = meta.merge(feat, on="Envir", how="left")

    feature_cols = [c for c in merged.columns if c not in ["Genotype", "Envir", "Harvest_date"]]

    # Keep only numeric features
    keep_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(merged[c]):
            keep_cols.append(c)

    feature_cols = keep_cols

    missing_by_col = merged[feature_cols].isna().sum()
    if missing_by_col.sum() > 0:
        print("Missing values detected in derived weather features after merge.")
        print(missing_by_col[missing_by_col > 0])

    # Fill missing using column means
    X = merged[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    X_np = X.to_numpy(dtype=np.float32)

    npy_file = OUT_DIR / f"weather_period_features_{version_name}.npy"
    cols_file = OUT_DIR / f"weather_period_features_{version_name}_columns.csv"
    merged_file = OUT_DIR / f"weather_period_features_{version_name}_aligned.csv"

    np.save(npy_file, X_np)
    pd.DataFrame({"feature_name": feature_cols}).to_csv(cols_file, index=False)
    merged.to_csv(merged_file, index=False)

    print(f"Saved numpy array: {npy_file}")
    print(f"Shape: {X_np.shape}")
    print(f"Saved columns: {cols_file}")
    print(f"Saved aligned table: {merged_file}")


def main():
    for version_name, feature_file in DERIVED_FILES.items():
        process_one_version(version_name, feature_file)


if __name__ == "__main__":
    main()