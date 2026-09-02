################################################################################
### Q0b_make_inner_validation_splits_newtraits.py
###
### Creates inner validation splits for the new traits:
###   - Acidity
###   - Color_over
###
### Uses:
###   - Output/Intermediate/numpy_arrays_newtraits/<Trait>/sample_metadata_<Trait>.csv
###   - Input/CV1_Strategy/Harvest_date_CV.csv
###
### Logic:
###   For each trait and for each CV split:
###       Testing = CV == 1
###       Outer training = CV == 0
###
###       Within the outer training set:
###           - unique genotypes are identified
###           - 15% of the genotypes are assigned to validation
###           - 85% of the genotypes are assigned to subtraining
###
###       All genotype-environment observations belonging to the same
###       genotype are retained in the same inner subset.
###
### Output:
###   Output/datasets/inner_validation_splits_newtraits.csv
###
### Note:
###   The file contains columns specific to each trait:
###       Acidity_CV1_Split1_Subtrain
###       Acidity_CV1_Split1_Validation
###       Acidity_CV1_Split1_Testing
###       Acidity_CV1_Split1_role
###       ...
################################################################################

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

NN_OUT_DIR = (
    Path("03_acidity_color_over")
    / "04_neural_network"
    / "output"
)

NPY_BASE_DIR = NN_OUT_DIR / "numpy_arrays_newtraits"

CV_FILE = (
    Path("data")
    / "raw"
    / "cv"
    / "Harvest_date_CV.csv"
)

OUT_DIR = NN_OUT_DIR / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "inner_validation_splits_newtraits.csv"
REPORT_FILE = OUT_DIR / "inner_validation_splits_newtraits_report.txt"

GLOBAL_SEED = 42
VALIDATION_FRACTION = 0.15


# =============================================================================
# HELPERS
# =============================================================================

def stable_seed(text: str, base_seed: int = 42) -> int:
    """
    Crea un seed stabile da una stringa.
    Non usa hash() di Python perché hash() può cambiare tra sessioni.
    """
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return base_seed + int(h[:8], 16) % 1000000


def clean_genotype(x):
    return str(x).replace("G_", "").strip()


def load_trait_metadata(trait: str) -> pd.DataFrame:
    meta_file = NPY_BASE_DIR / trait / f"sample_metadata_{trait}.csv"

    if not meta_file.exists():
        raise FileNotFoundError(f"File metadata non trovato:\n{meta_file}")

    meta = pd.read_csv(meta_file)

    needed = {"Genotype", "Envir", trait}
    missing = needed - set(meta.columns)

    if missing:
        raise ValueError(
            f"Nel file {meta_file} mancano colonne: {missing}\n"
            f"Colonne trovate: {meta.columns.tolist()}"
        )

    meta = meta[["Genotype", "Envir", trait]].copy()
    meta["Genotype"] = meta["Genotype"].apply(clean_genotype)
    meta["Envir"] = meta["Envir"].astype(str).str.strip()
    meta["ID_key"] = meta["Envir"] + "-" + meta["Genotype"]

    if meta["ID_key"].duplicated().any():
        dup = meta.loc[meta["ID_key"].duplicated(), "ID_key"].head(20).tolist()
        raise ValueError(
            f"ID_key duplicati nel metadata di {trait}. Esempi:\n{dup}"
        )

    return meta


def load_cv() -> pd.DataFrame:
    if not CV_FILE.exists():
        raise FileNotFoundError(f"File CV non trovato:\n{CV_FILE}")

    cv = pd.read_csv(CV_FILE)
    cv["Genotype"] = cv["Genotype"].apply(clean_genotype)
    cv["Envir"] = cv["Envir"].astype(str).str.strip()
    cv["ID_key"] = cv["Envir"] + "-" + cv["Genotype"]

    if cv["ID_key"].duplicated().any():
        dup = cv.loc[cv["ID_key"].duplicated(), "ID_key"].head(20).tolist()
        raise ValueError(
            f"ID_key duplicati nel file CV. Esempi:\n{dup}"
        )

    return cv

def make_inner_for_trait(
    trait: str,
    meta: pd.DataFrame,
    cv: pd.DataFrame,
    split_cols
):
    """
    For each outer split:
      - Testing observations are inherited from the official CV strategy.
      - Unique outer-training genotypes are identified.
      - 15% of outer-training genotypes are assigned to validation.
      - The remaining genotypes are assigned to subtraining.
      - All genotype-environment observations of the same genotype
        receive the same inner role.
    """

    cv_indexed = cv.set_index("ID_key")

    missing_keys = sorted(
        set(meta["ID_key"]) - set(cv_indexed.index)
    )

    if len(missing_keys) > 0:
        raise ValueError(
            f"For {trait}, some metadata rows are absent from the "
            f"official CV file.\n"
            f"Example missing ID_key values: {missing_keys[:20]}"
        )

    # Align official CV rows to trait metadata.
    cv_aligned = cv_indexed.loc[meta["ID_key"]].reset_index()

    out = meta[
        ["Trait", "Envir", "Genotype", "ID_key"]
    ].copy()

    summary_rows = []

    for split_name in split_cols:

        cv_values = (
            cv_aligned[split_name]
            .astype(int)
            .to_numpy()
        )

        test_mask = cv_values == 1
        outer_train_mask = cv_values == 0

        if test_mask.sum() == 0:
            raise ValueError(
                f"{trait} {split_name}: empty outer test set."
            )

        # Check that one genotype is not present in both outer train and test.
        train_genotypes = set(
            meta.loc[outer_train_mask, "Genotype"]
            .astype(str)
            .unique()
        )

        test_genotypes = set(
            meta.loc[test_mask, "Genotype"]
            .astype(str)
            .unique()
        )

        overlapping_genotypes = sorted(
            train_genotypes.intersection(test_genotypes)
        )

        if overlapping_genotypes:
            raise ValueError(
                f"{trait} {split_name}: some genotypes occur in both "
                f"outer training and outer test.\n"
                f"Examples: {overlapping_genotypes[:20]}"
            )

        train_genotypes = np.array(
            sorted(train_genotypes),
            dtype=object
        )

        if len(train_genotypes) < 5:
            raise ValueError(
                f"{trait} {split_name}: too few outer-training "
                f"genotypes: {len(train_genotypes)}"
            )

        # Stable trait- and split-specific random generator.
        rng = np.random.default_rng(
            stable_seed(
                f"{trait}_{split_name}",
                GLOBAL_SEED
            )
        )

        shuffled_genotypes = train_genotypes.copy()
        rng.shuffle(shuffled_genotypes)

        n_validation_genotypes = max(
            1,
            int(
                round(
                    VALIDATION_FRACTION
                    * len(shuffled_genotypes)
                )
            )
        )

        validation_genotypes = set(
            shuffled_genotypes[
                :n_validation_genotypes
            ].tolist()
        )

        subtrain_genotypes = set(
            shuffled_genotypes[
                n_validation_genotypes:
            ].tolist()
        )

        testing = test_mask.astype(int)

        validation = (
            outer_train_mask
            & meta["Genotype"]
                .astype(str)
                .isin(validation_genotypes)
                .to_numpy()
        ).astype(int)

        subtrain = (
            outer_train_mask
            & meta["Genotype"]
                .astype(str)
                .isin(subtrain_genotypes)
                .to_numpy()
        ).astype(int)

        # Every row must belong to exactly one role.
        role_sum = testing + validation + subtrain

        if not np.all(role_sum == 1):
            bad_indices = np.where(role_sum != 1)[0][:20]

            raise ValueError(
                f"{trait} {split_name}: invalid role assignment. "
                f"Some rows belong to zero or multiple subsets.\n"
                f"Example row indices: {bad_indices.tolist()}"
            )

        role = np.full(
            len(meta),
            "Unused",
            dtype=object
        )

        role[testing == 1] = "Testing"
        role[validation == 1] = "Validation"
        role[subtrain == 1] = "Subtrain"

        prefix = f"{trait}_{split_name}"

        out[f"{prefix}_Testing"] = testing
        out[f"{prefix}_Validation"] = validation
        out[f"{prefix}_Subtrain"] = subtrain
        out[f"{prefix}_role"] = role

        summary_rows.append({
            "Trait": trait,
            "Split": split_name,
            "n_total_observations": len(meta),
            "n_test_observations": int(testing.sum()),
            "n_subtrain_observations": int(subtrain.sum()),
            "n_validation_observations": int(validation.sum()),
            "n_outer_training_genotypes": len(train_genotypes),
            "n_subtrain_genotypes": len(subtrain_genotypes),
            "n_validation_genotypes": len(validation_genotypes),
            "validation_genotype_fraction": (
                len(validation_genotypes)
                / len(train_genotypes)
            ),
        })

    summary = pd.DataFrame(summary_rows)

    return out, summary

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Q0b - MAKE INNER VALIDATION SPLITS FOR NEW TRAITS")
    print("=" * 80)

    cv = load_cv()
    split_cols = [c for c in cv.columns if c.startswith("CV")]

    print(f"CV rows: {cv.shape[0]}")
    print(f"Number of split columns: {len(split_cols)}")

    all_inner = []
    all_summary = []

    for trait in TRAITS:
        print("\n" + "-" * 80)
        print(f"Processing trait: {trait}")
        print("-" * 80)

        meta = load_trait_metadata(trait)
        meta["Trait"] = trait

        print(f"Metadata rows: {meta.shape[0]}")
        print(f"Unique genotypes: {meta['Genotype'].nunique()}")
        print(f"Unique environments: {meta['Envir'].nunique()}")

        inner_trait, summary_trait = make_inner_for_trait(
            trait=trait,
            meta=meta,
            cv=cv,
            split_cols=split_cols
        )

        all_inner.append(inner_trait)
        all_summary.append(summary_trait)

        print(summary_trait[["Split","n_total_observations","n_subtrain_observations","n_validation_observations","n_test_observations",]].head())

    inner_all = pd.concat(all_inner, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)

    inner_all.to_csv(OUT_FILE, index=False)

    summary_file = OUT_DIR / "inner_validation_splits_newtraits_summary.csv"
    summary_all.to_csv(summary_file, index=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== INNER VALIDATION SPLITS NEW TRAITS REPORT ===\n\n")
        f.write(f"TRAITS: {TRAITS}\n")
        f.write(f"GLOBAL_SEED: {GLOBAL_SEED}\n")
        f.write(f"VALIDATION_FRACTION: {VALIDATION_FRACTION}\n")
        f.write(f"CV_FILE: {CV_FILE}\n\n")

        f.write("Summary by trait:\n")
        f.write(
            summary_all
            .groupby("Trait")
            .agg(
                n_splits=(
                    "Split",
                    "nunique"
                ),
                mean_outer_training_genotypes=(
                    "n_outer_training_genotypes",
                    "mean"
                ),
                mean_subtrain_genotypes=(
                    "n_subtrain_genotypes",
                    "mean"
                ),
                mean_validation_genotypes=(
                    "n_validation_genotypes",
                    "mean"
                ),
                mean_validation_genotype_fraction=(
                    "validation_genotype_fraction",
                    "mean"
                ),
                mean_subtrain_observations=(
                    "n_subtrain_observations",
                    "mean"
                ),
                mean_validation_observations=(
                    "n_validation_observations",
                    "mean"
                ),
                mean_test_observations=(
                    "n_test_observations",
                    "mean"
                ),
            )
            .reset_index()
            .to_string(index=False)
        )

        f.write("\n\nFull summary:\n")
        f.write(summary_all.to_string(index=False))

    print("\n" + "=" * 80)
    print("Q0b completed.")
    print("=" * 80)
    print("Saved:")
    print(OUT_FILE)
    print(summary_file)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()
