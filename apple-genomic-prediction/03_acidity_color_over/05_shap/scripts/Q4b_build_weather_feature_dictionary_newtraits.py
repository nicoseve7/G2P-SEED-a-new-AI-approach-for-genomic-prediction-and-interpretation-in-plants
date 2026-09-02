# -*- coding: utf-8 -*-

################################################################################
### Q4b_build_weather_feature_dictionary_newtraits.py
###
### Build dictionary for Weather V3 feature names for new no-soil traits.
###
### Da eseguire da:
###   dalpaper/nuovitrattinosoil/
################################################################################

from pathlib import Path
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

BASE_MODEL_DIR = Path("Output/02_no_soil_model")
NPY_BASE_DIR = Path("Output/Intermediate/numpy_arrays_newtraits")

COMMON_DICT_DIR = BASE_MODEL_DIR / "Feature_dictionaries"
COMMON_DICT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_SAVE_FILE = COMMON_DICT_DIR / "weather_v3_feature_dictionary.csv"


# =============================================================================
# HELPERS
# =============================================================================

def infer_period(name: str):
    low = name.lower()

    if "p2a" in low:
        return "P2a"
    if "p2b" in low:
        return "P2b"
    if "p1" in low:
        return "P1"
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

    if "radiation" in low or "rad" in low:
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


def load_weather_columns_for_trait(trait: str):
    weather_cols_file = (
        NPY_BASE_DIR / trait / "weather_period_features_v3_columns.csv"
    )

    if not weather_cols_file.exists():
        raise FileNotFoundError(f"File non trovato:\n{weather_cols_file}")

    cols_df = pd.read_csv(weather_cols_file)

    if "feature_name" not in cols_df.columns:
        raise ValueError(
            f"Mi aspettavo una colonna 'feature_name' in {weather_cols_file}, "
            f"ma ho trovato: {cols_df.columns.tolist()}"
        )

    return cols_df["feature_name"].astype(str).tolist()


def build_dictionary(feature_cols):
    out = pd.DataFrame({
        "Feature_Index_1based": range(1, len(feature_cols) + 1),
        "Feature_Code": [f"WeatherV3_{i+1}" for i in range(len(feature_cols))],
        "Original_Feature_Name": feature_cols,
    })

    out["Period"] = out["Original_Feature_Name"].apply(infer_period)
    out["Variable"] = out["Original_Feature_Name"].apply(infer_variable)
    out["Statistic"] = out["Original_Feature_Name"].apply(infer_stat)

    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Q4b - BUILD WEATHER V3 FEATURE DICTIONARY")
    print("=" * 80)

    # Usiamo il primo trait come riferimento.
    ref_trait = TRAITS[0]
    ref_cols = load_weather_columns_for_trait(ref_trait)

    # Controlliamo che gli altri trait abbiano le stesse feature nello stesso ordine.
    for trait in TRAITS[1:]:
        cols = load_weather_columns_for_trait(trait)

        if cols != ref_cols:
            raise ValueError(
                f"Le feature weather di {trait} non coincidono con quelle di {ref_trait}.\n"
                f"Controlla weather_period_features_v3_columns.csv"
            )

    dictionary = build_dictionary(ref_cols)

    dictionary.to_csv(COMMON_SAVE_FILE, index=False)

    print("Dizionario comune salvato:")
    print(COMMON_SAVE_FILE)

    # Copia anche dentro ogni trait, così H2/H3 possono trovarlo facilmente.
    for trait in TRAITS:
        trait_dict_dir = BASE_MODEL_DIR / trait / "Interpretation" / "Feature_dictionaries"
        trait_dict_dir.mkdir(parents=True, exist_ok=True)

        trait_save_file = trait_dict_dir / "weather_v3_feature_dictionary.csv"
        dictionary.to_csv(trait_save_file, index=False)

        print(f"Dizionario salvato anche per {trait}:")
        print(trait_save_file)

    print("\nPrime 20 righe:")
    print(dictionary.head(20).to_string(index=False))

    print("\nConteggi per periodo:")
    print(dictionary["Period"].value_counts(dropna=False).to_string())

    print("\nConteggi per variabile:")
    print(dictionary["Variable"].value_counts(dropna=False).to_string())

    print("\nConteggi per statistica:")
    print(dictionary["Statistic"].value_counts(dropna=False).to_string())

    print("\nQ4b completed.")


if __name__ == "__main__":
    main()