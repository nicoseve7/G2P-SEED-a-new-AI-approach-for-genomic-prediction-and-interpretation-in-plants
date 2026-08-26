# -*- coding: utf-8 -*-

################################################################################
### H1b_build_weather_feature_dictionary.py
### Build dictionary for WeatherV3 feature names
################################################################################

from pathlib import Path
import pandas as pd

TRAIT = "Harvest_date"

OUT_DIR = Path("Output")
DICT_DIR = OUT_DIR / "Interpretation" / TRAIT / "Feature_dictionaries"
DICT_DIR.mkdir(parents=True, exist_ok=True)

# FILE REALI CHE HAI GIÀ
WEATHER_COLUMNS_FILE = Path(
    "../esperimento_meteo/Output/numpy_arrays_weather_exp/weather_period_features_v3_columns.csv"
)

# opzionale, solo per controllo
WEATHER_ALIGNED_FILE = Path(
    "../esperimento_meteo/Output/numpy_arrays_weather_exp/weather_period_features_v3_aligned.csv"
)

SAVE_FILE = DICT_DIR / "weather_v3_feature_dictionary.csv"


def infer_period(name: str):
    low = name.lower()
    if "p1" in low:
        return "P1"
    if "p2a" in low:
        return "P2a"
    if "p2b" in low:
        return "P2b"
    if "p2" in low:
        return "P2"
    return "Unknown"


def infer_variable(name: str):
    low = name.lower()

    if low.startswith("n_days"):
        return "n_days"

    if "temperature" in low or "temp" in low:
        return "Temperature"
    if "humidity" in low or "humid" in low:
        return "Humidity"
    if "radiation" in low or "rad_" in low or "radsum" in low:
        return "Radiation"

    return "Unknown"


def infer_stat(name: str):
    low = name.lower()

    if low.startswith("n_days"):
        return "count_days"

    if "_sum" in low:
        return "sum"
    if "_mean" in low:
        return "mean"
    if "_sd" in low or "_std" in low:
        return "sd"
    if "_min" in low:
        return "min"
    if "_max" in low:
        return "max"

    return "Unknown"


def main():
    if not WEATHER_COLUMNS_FILE.exists():
        raise FileNotFoundError(
            f"File non trovato:\n{WEATHER_COLUMNS_FILE}"
        )

    cols_df = pd.read_csv(WEATHER_COLUMNS_FILE)

    if "feature_name" not in cols_df.columns:
        raise ValueError(
            f"Mi aspettavo una colonna 'feature_name' in {WEATHER_COLUMNS_FILE}, "
            f"ma ho trovato: {cols_df.columns.tolist()}"
        )

    feature_cols = cols_df["feature_name"].astype(str).tolist()

    # controllo opzionale con aligned.csv
    if WEATHER_ALIGNED_FILE.exists():
        aligned_df = pd.read_csv(WEATHER_ALIGNED_FILE, nrows=2)
        meta_cols = {"Genotype", "Envir", "Harvest_date"}
        aligned_feature_cols = [c for c in aligned_df.columns if c not in meta_cols]

        if aligned_feature_cols != feature_cols:
            print("ATTENZIONE: le colonne di aligned.csv e columns.csv non coincidono perfettamente.")
            print("Uso comunque l'ordine di weather_period_features_v3_columns.csv")

    out = pd.DataFrame({
        "Feature_Index_1based": range(1, len(feature_cols) + 1),
        "Feature_Code": [f"WeatherV3_{i+1}" for i in range(len(feature_cols))],
        "Original_Feature_Name": feature_cols,
    })

    out["Period"] = out["Original_Feature_Name"].apply(infer_period)
    out["Variable"] = out["Original_Feature_Name"].apply(infer_variable)
    out["Statistic"] = out["Original_Feature_Name"].apply(infer_stat)

    out.to_csv(SAVE_FILE, index=False)

    print("Salvato:")
    print(SAVE_FILE)
    print("\nPrime 20 righe:")
    print(out.head(20).to_string(index=False))
    print("\nConteggi per periodo:")
    print(out["Period"].value_counts(dropna=False).to_string())
    print("\nConteggi per variabile:")
    print(out["Variable"].value_counts(dropna=False).to_string())
    print("\nConteggi per statistica:")
    print(out["Statistic"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()