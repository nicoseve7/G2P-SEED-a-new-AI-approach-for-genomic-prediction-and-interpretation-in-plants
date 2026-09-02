# -*- coding: utf-8 -*-

################################################################################
### P6_build_numpy_inputs_no_soil_traits.py
###
### Build sample-aligned inputs for the no-soil V3 network
### for:
###   - Acidity
###   - Color_over
###
### This script builds:
###   - sample metadata, filtered to samples present in the official CV file
###   - PCA array aligned to each trait sample
###   - Weather V3 period feature array aligned to each trait sample
###
### It does NOT build:
###   - daily weather 300 x 3 arrays
###   - soil arrays
###   - SNP .npy arrays
###
### Because the no-soil V3 model uses:
###   - sample metadata
###   - PCA array
###   - Weather V3 period features: P1 / P2a / P2b
###   - split-specific geno_CV*_Split*.csv files read directly by the model
################################################################################

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

OUT_DIR = Path("Output/Intermediate/numpy_arrays_newtraits")

PHENO_BASE_DIR = Path("Output/01_pheno_processed")

PCA_FILE = Path("Output/genomic_PCs_20_paper_style.csv")

CV_FILE = Path("Input/CV1_Strategy/Harvest_date_CV.csv")

# File weather V3 prodotti nella pipeline esperimento_meteo.
# weather_period_features_v3_aligned.csv contiene molte righe per Envir
# perché era già allineato ai campioni Harvest_date.
# Per questo script lo riduciamo internamente a una riga per Envir.
WEATHER_FEATURES_FILE = Path(
    "../esperimento_meteo/Output/numpy_arrays_weather_exp/weather_period_features_v3_aligned.csv"
)

WEATHER_COLUMNS_FILE = Path(
    "../esperimento_meteo/Output/numpy_arrays_weather_exp/weather_period_features_v3_columns.csv"
)


# =============================================================================
# HELPERS
# =============================================================================

def clean_genotype(x):
    return str(x).replace("G_", "").strip()


def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File non trovato:\n{path}")


def make_id_key(df: pd.DataFrame) -> pd.Series:
    return df["Envir"].astype(str).str.strip() + "-" + df["Genotype"].astype(str).str.strip()


def load_cv_keys() -> set:
    """
    Carica la CV ufficiale e restituisce gli ID_key Envir-Genotype presenti.

    Serve perché la rete può allenarsi/testare solo campioni che hanno
    assegnazione nei 25 split ufficiali.
    """
    check_file(CV_FILE)

    cv = pd.read_csv(CV_FILE)

    needed = {"Genotype", "Envir"}
    missing = needed - set(cv.columns)

    if missing:
        raise ValueError(
            f"Nel file CV mancano colonne: {missing}\n"
            f"Colonne trovate: {cv.columns.tolist()}"
        )

    cv = cv[["Genotype", "Envir"]].copy()
    cv["Genotype"] = cv["Genotype"].apply(clean_genotype)
    cv["Envir"] = cv["Envir"].astype(str).str.strip()
    cv["ID_key"] = make_id_key(cv)

    duplicated = int(cv["ID_key"].duplicated().sum())
    if duplicated > 0:
        dup_examples = cv.loc[cv["ID_key"].duplicated(), "ID_key"].head(20).tolist()
        raise ValueError(
            f"Nel file CV ci sono {duplicated} ID_key duplicati.\n"
            f"Esempi: {dup_examples}"
        )

    return set(cv["ID_key"].astype(str))


def load_trait_pheno(trait: str) -> pd.DataFrame:
    pheno_file = PHENO_BASE_DIR / trait / f"{trait}_processed_final.csv"
    check_file(pheno_file)

    pheno = pd.read_csv(pheno_file)

    needed = {"Genotype", "Envir", trait}
    missing = needed - set(pheno.columns)

    if missing:
        raise ValueError(
            f"Nel file {pheno_file} mancano colonne: {missing}\n"
            f"Colonne trovate: {pheno.columns.tolist()}"
        )

    pheno = pheno[["Genotype", "Envir", trait]].copy()
    pheno["Genotype"] = pheno["Genotype"].apply(clean_genotype)
    pheno["Envir"] = pheno["Envir"].astype(str).str.strip()
    pheno[trait] = pd.to_numeric(pheno[trait], errors="coerce")

    pheno = pheno.dropna(subset=[trait]).copy()
    pheno = pheno.reset_index(drop=True)

    if pheno.empty:
        raise ValueError(f"Nessuna riga valida per il trait {trait} dopo dropna.")

    # Controllo importante: non vogliamo righe duplicate Envir-Genotype.
    pheno["ID_key"] = make_id_key(pheno)
    duplicated = int(pheno["ID_key"].duplicated().sum())

    if duplicated > 0:
        dup_examples = pheno.loc[pheno["ID_key"].duplicated(), "ID_key"].head(20).tolist()
        raise ValueError(
            f"Nel fenotipo finale di {trait} ci sono {duplicated} ID_key duplicati.\n"
            f"Esempi: {dup_examples}\n"
            f"La rete si aspetta una sola riga per combinazione Envir-Genotype."
        )

    return pheno


def filter_pheno_to_cv(pheno: pd.DataFrame, trait: str, cv_keys: set):
    """
    Mantiene solo i campioni del trait presenti nella CV ufficiale.

    Questo è necessario perché Q1/Q2 devono sapere, per ogni campione,
    se appartiene a test/training in ciascuno dei 25 split.
    """
    out = pheno.copy()

    if "ID_key" not in out.columns:
        out["ID_key"] = make_id_key(out)

    n_before = out.shape[0]
    n_geno_before = out["Genotype"].nunique()
    n_env_before = out["Envir"].nunique()

    missing_mask = ~out["ID_key"].isin(cv_keys)
    missing_keys = out.loc[missing_mask, "ID_key"].tolist()

    out = out.loc[~missing_mask].copy().reset_index(drop=True)

    n_after = out.shape[0]
    n_geno_after = out["Genotype"].nunique()
    n_env_after = out["Envir"].nunique()

    print("\nCV filtering:")
    print(f"  Rows before CV filtering: {n_before}")
    print(f"  Rows after CV filtering:  {n_after}")
    print(f"  Dropped because not in CV: {n_before - n_after}")
    print(f"  Genotypes before/after: {n_geno_before} -> {n_geno_after}")
    print(f"  Environments before/after: {n_env_before} -> {n_env_after}")

    if len(missing_keys) > 0:
        print("  Example dropped ID_key:")
        print(f"  {missing_keys[:20]}")

    if out.empty:
        raise ValueError(
            f"Dopo il filtro sulla CV ufficiale, il trait {trait} non ha più righe."
        )

    dropped_df = pd.DataFrame({"dropped_ID_key_not_in_CV": missing_keys})

    return out, dropped_df


def load_pca() -> pd.DataFrame:
    check_file(PCA_FILE)

    pca = pd.read_csv(PCA_FILE)

    if "Genotype" not in pca.columns:
        raise ValueError(
            f"Nel file PCA manca la colonna Genotype.\n"
            f"Colonne trovate: {pca.columns.tolist()}"
        )

    pca["Genotype"] = pca["Genotype"].apply(clean_genotype)

    pc_cols = [c for c in pca.columns if c.startswith("PC")]

    if len(pc_cols) == 0:
        raise ValueError(
            f"Nessuna colonna PC trovata nel file PCA.\n"
            f"Colonne trovate: {pca.columns.tolist()}"
        )

    duplicated = int(pca["Genotype"].duplicated().sum())
    if duplicated > 0:
        dup_examples = pca.loc[pca["Genotype"].duplicated(), "Genotype"].head(20).tolist()
        raise ValueError(
            f"Nel file PCA ci sono {duplicated} Genotype duplicati.\n"
            f"Esempi: {dup_examples}"
        )

    for c in pc_cols:
        pca[c] = pd.to_numeric(pca[c], errors="coerce")

    return pca[["Genotype"] + pc_cols].copy()


def load_weather_v3():
    """
    Carica le feature meteo V3.

    Il file weather_period_features_v3_aligned.csv è sample-aligned alla vecchia
    pipeline Harvest_date, quindi contiene più righe per lo stesso Envir.

    Siccome le feature meteo V3 dipendono solo dall'ambiente/anno, qui costruiamo
    una tabella weather_by_env con una sola riga per Envir.

    Poi, per ogni nuovo trait, questa tabella viene riallineata ai campioni:
        sample trait rows x 48 weather features
    """
    check_file(WEATHER_FEATURES_FILE)
    check_file(WEATHER_COLUMNS_FILE)

    weather_raw = pd.read_csv(WEATHER_FEATURES_FILE)
    cols_df = pd.read_csv(WEATHER_COLUMNS_FILE)

    if "Envir" not in weather_raw.columns:
        raise ValueError(
            f"Nel file weather manca Envir.\n"
            f"Colonne trovate: {weather_raw.columns.tolist()}"
        )

    if "feature_name" not in cols_df.columns:
        raise ValueError(
            f"Nel file colonne weather manca feature_name.\n"
            f"Colonne trovate: {cols_df.columns.tolist()}"
        )

    weather_raw["Envir"] = weather_raw["Envir"].astype(str).str.strip()
    feature_cols = cols_df["feature_name"].astype(str).tolist()

    missing_features = [c for c in feature_cols if c not in weather_raw.columns]

    if missing_features:
        raise ValueError(
            f"Alcune feature meteo dichiarate in columns.csv non sono presenti "
            f"in weather_period_features_v3_aligned.csv.\n"
            f"Esempi missing: {missing_features[:20]}"
        )

    weather_raw = weather_raw[["Envir"] + feature_cols].copy()

    for c in feature_cols:
        weather_raw[c] = pd.to_numeric(weather_raw[c], errors="coerce")

    n_rows_raw = weather_raw.shape[0]
    n_env_raw = weather_raw["Envir"].nunique()

    print("\nWeather raw loaded:")
    print(f"  rows: {n_rows_raw}")
    print(f"  unique Envir: {n_env_raw}")

    # Verifica consistenza: ogni Envir deve avere un unico profilo meteo.
    inconsistent_envs = []

    for env, sub in weather_raw.groupby("Envir", sort=False):
        unique_profiles = sub[feature_cols].drop_duplicates()
        if unique_profiles.shape[0] > 1:
            inconsistent_envs.append(env)

    if len(inconsistent_envs) > 0:
        raise ValueError(
            "Alcuni Envir hanno più di un profilo meteo diverso. "
            "Questo non dovrebbe succedere per le feature V3 environment-level.\n"
            f"Esempi Envir inconsistenti: {inconsistent_envs[:20]}"
        )

    # Una sola riga per Envir.
    weather_by_env = (
        weather_raw
        .drop_duplicates(subset=["Envir"])
        .reset_index(drop=True)
        .copy()
    )

    if weather_by_env["Envir"].duplicated().any():
        raise ValueError("Errore interno: weather_by_env contiene Envir duplicati.")

    missing_total = int(weather_by_env[feature_cols].isna().sum().sum())
    if missing_total > 0:
        missing_by_col = weather_by_env[feature_cols].isna().sum()
        missing_by_col = missing_by_col[missing_by_col > 0]
        raise ValueError(
            f"Weather V3 contiene {missing_total} valori mancanti dopo riduzione per Envir.\n"
            f"Missing per colonna:\n{missing_by_col.to_string()}"
        )

    print("Weather reduced to one row per Envir:")
    print(f"  shape: {weather_by_env.shape}")
    print(f"  features: {len(feature_cols)}")

    return weather_by_env, feature_cols, n_rows_raw, n_env_raw


def process_one_trait(
    trait: str,
    pca: pd.DataFrame,
    weather_by_env: pd.DataFrame,
    weather_cols,
    weather_raw_n_rows: int,
    weather_raw_n_env: int,
    cv_keys: set,
):
    print("\n" + "=" * 80)
    print(f"Processing trait: {trait}")
    print("=" * 80)

    trait_out_dir = OUT_DIR / trait
    trait_out_dir.mkdir(parents=True, exist_ok=True)

    pheno_raw = load_trait_pheno(trait)

    print(f"Phenotype rows before CV filtering: {pheno_raw.shape[0]}")
    print(f"Unique genotypes before CV filtering: {pheno_raw['Genotype'].nunique()}")
    print(f"Unique environments before CV filtering: {pheno_raw['Envir'].nunique()}")

    pheno, dropped_cv_df = filter_pheno_to_cv(
        pheno=pheno_raw,
        trait=trait,
        cv_keys=cv_keys
    )

    dropped_cv_file = trait_out_dir / f"P6_dropped_samples_not_in_CV_{trait}.csv"
    dropped_cv_df.to_csv(dropped_cv_file, index=False)

    print(f"\nPhenotype rows used by network: {pheno.shape[0]}")
    print(f"Unique genotypes used by network: {pheno['Genotype'].nunique()}")
    print(f"Unique environments used by network: {pheno['Envir'].nunique()}")

    # Rimuoviamo ID_key dal metadata finale perché Q1 lo ricostruisce.
    # Però lo teniamo internamente finché serve per il report.
    pheno_for_output = pheno[["Genotype", "Envir", trait]].copy()

    # -------------------------------------------------------------------------
    # PCA alignment
    # PCA dipende dal Genotype, quindi tutti i campioni dello stesso genotipo
    # ricevono lo stesso vettore PC.
    # -------------------------------------------------------------------------
    merged_pca = pheno_for_output[["Genotype", "Envir", trait]].merge(
        pca,
        on="Genotype",
        how="left",
        validate="many_to_one"
    )

    if merged_pca.shape[0] != pheno_for_output.shape[0]:
        raise ValueError(
            f"Il merge PCA ha cambiato il numero di righe per {trait}: "
            f"pheno={pheno_for_output.shape[0]}, merged_pca={merged_pca.shape[0]}"
        )

    pc_cols = [c for c in pca.columns if c != "Genotype"]

    missing_pca_rows = int(merged_pca[pc_cols].isna().any(axis=1).sum())

    if missing_pca_rows > 0:
        missing_genotypes = (
            merged_pca.loc[merged_pca[pc_cols].isna().any(axis=1), "Genotype"]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Per il trait {trait}, {missing_pca_rows} righe non hanno PCA.\n"
            f"Esempi genotipi mancanti: {missing_genotypes[:20]}"
        )

    X_pca = merged_pca[pc_cols].to_numpy(dtype=np.float32)

    # -------------------------------------------------------------------------
    # Weather V3 alignment
    # Weather V3 dipende da Envir, quindi tutti i campioni dello stesso Envir
    # ricevono lo stesso vettore weather.
    #
    # Output finale: n_sample_trait x 48
    # -------------------------------------------------------------------------
    merged_weather = pheno_for_output[["Genotype", "Envir", trait]].merge(
        weather_by_env,
        on="Envir",
        how="left",
        validate="many_to_one"
    )

    if merged_weather.shape[0] != pheno_for_output.shape[0]:
        raise ValueError(
            f"Il merge weather ha cambiato il numero di righe per {trait}: "
            f"pheno={pheno_for_output.shape[0]}, merged_weather={merged_weather.shape[0]}"
        )

    missing_weather_rows = int(merged_weather[weather_cols].isna().any(axis=1).sum())

    if missing_weather_rows > 0:
        missing_envs = (
            merged_weather.loc[merged_weather[weather_cols].isna().any(axis=1), "Envir"]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Per il trait {trait}, {missing_weather_rows} righe non hanno weather V3.\n"
            f"Environment mancanti: {missing_envs}"
        )

    X_weather = merged_weather[weather_cols].to_numpy(dtype=np.float32)

    # -------------------------------------------------------------------------
    # Final sanity checks
    # -------------------------------------------------------------------------
    if X_pca.shape[0] != pheno_for_output.shape[0]:
        raise ValueError(
            f"PCA rows mismatch per {trait}: "
            f"X_pca={X_pca.shape[0]}, pheno={pheno_for_output.shape[0]}"
        )

    if X_weather.shape[0] != pheno_for_output.shape[0]:
        raise ValueError(
            f"Weather rows mismatch per {trait}: "
            f"X_weather={X_weather.shape[0]}, pheno={pheno_for_output.shape[0]}"
        )

    if X_weather.shape[1] != len(weather_cols):
        raise ValueError(
            f"Weather columns mismatch per {trait}: "
            f"X_weather={X_weather.shape[1]}, weather_cols={len(weather_cols)}"
        )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    sample_meta = pheno_for_output.copy()

    sample_meta_file = trait_out_dir / f"sample_metadata_{trait}.csv"
    pca_file = trait_out_dir / "pca.npy"
    weather_file = trait_out_dir / "weather_period_features_v3.npy"
    weather_cols_file = trait_out_dir / "weather_period_features_v3_columns.csv"
    report_file = trait_out_dir / f"P6_numpy_inputs_report_{trait}.txt"

    sample_meta.to_csv(sample_meta_file, index=False)
    np.save(pca_file, X_pca)
    np.save(weather_file, X_weather)
    pd.DataFrame({"feature_name": weather_cols}).to_csv(weather_cols_file, index=False)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"=== P6 NUMPY INPUTS REPORT: {trait} ===\n\n")
        f.write(f"Trait: {trait}\n\n")

        f.write("CV filtering:\n")
        f.write(f"Rows before CV filtering: {pheno_raw.shape[0]}\n")
        f.write(f"Rows after CV filtering: {sample_meta.shape[0]}\n")
        f.write(f"Dropped because not in CV: {pheno_raw.shape[0] - sample_meta.shape[0]}\n")
        f.write(f"Dropped samples file: {dropped_cv_file}\n\n")

        f.write("Sample metadata used by network:\n")
        f.write(f"Rows: {sample_meta.shape[0]}\n")
        f.write(f"Unique genotypes: {sample_meta['Genotype'].nunique()}\n")
        f.write(f"Unique environments: {sample_meta['Envir'].nunique()}\n\n")

        f.write("PCA:\n")
        f.write(f"Shape: {X_pca.shape}\n")
        f.write(f"Columns: {pc_cols}\n\n")

        f.write("Weather V3:\n")
        f.write(f"Raw weather aligned rows: {weather_raw_n_rows}\n")
        f.write(f"Raw weather aligned unique Envir: {weather_raw_n_env}\n")
        f.write(f"Weather by Envir shape used for merge: {weather_by_env.shape}\n")
        f.write(f"Final sample-aligned weather shape: {X_weather.shape}\n")
        f.write(f"Number of features: {len(weather_cols)}\n")
        f.write("Weather feature names:\n")
        for c in weather_cols:
            f.write(f"- {c}\n")

        f.write("\nOutput files:\n")
        f.write(f"- {sample_meta_file}\n")
        f.write(f"- {pca_file}\n")
        f.write(f"- {weather_file}\n")
        f.write(f"- {weather_cols_file}\n")
        f.write(f"- {dropped_cv_file}\n")

    print("Saved:")
    print(sample_meta_file)
    print(pca_file)
    print(weather_file)
    print(weather_cols_file)
    print(dropped_cv_file)
    print(report_file)

    return {
        "Trait": trait,
        "n_rows_before_CV_filter": pheno_raw.shape[0],
        "n_rows": sample_meta.shape[0],
        "n_dropped_not_in_CV": pheno_raw.shape[0] - sample_meta.shape[0],
        "n_genotypes": sample_meta["Genotype"].nunique(),
        "n_environments": sample_meta["Envir"].nunique(),
        "pca_shape": str(X_pca.shape),
        "weather_shape": str(X_weather.shape),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("P6 - BUILD NUMPY INPUTS FOR NO-SOIL NEW TRAITS")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cv_keys = load_cv_keys()

    print("\nLoaded official CV keys:")
    print(f"  n ID_key: {len(cv_keys)}")

    pca = load_pca()
    weather_by_env, weather_cols, weather_raw_n_rows, weather_raw_n_env = load_weather_v3()

    print("\nLoaded PCA:")
    print(pca.shape)

    print("\nLoaded Weather V3 by Envir:")
    print(weather_by_env.shape)
    print(f"Weather features: {len(weather_cols)}")

    rows = []

    for trait in TRAITS:
        rows.append(
            process_one_trait(
                trait=trait,
                pca=pca,
                weather_by_env=weather_by_env,
                weather_cols=weather_cols,
                weather_raw_n_rows=weather_raw_n_rows,
                weather_raw_n_env=weather_raw_n_env,
                cv_keys=cv_keys,
            )
        )

    summary = pd.DataFrame(rows)
    summary_file = OUT_DIR / "P6_numpy_inputs_newtraits_summary.csv"
    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 80)
    print("P6 completed.")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("\nSaved summary:")
    print(summary_file)


if __name__ == "__main__":
    main()