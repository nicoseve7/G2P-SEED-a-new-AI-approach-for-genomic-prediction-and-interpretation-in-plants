# -*- coding: utf-8 -*-

################################################################################
### G5_add_unique_snps_to_region_stability.py
###
### Adds a column with the list of unique SNPs selected in each 50kb region.
###
### Input:
###   Output/04_ga_runs/G2B_multiseed_variable_split_inner3cv/
###     G2B_region_stability_across_seeds.csv
###     G2B_selected_regions_all_seeds_long.csv
###
### Output:
###   Output/04_ga_runs/G2B_multiseed_variable_split_inner3cv/
###     G2B_region_stability_across_seeds_with_unique_snps.csv
################################################################################

from pathlib import Path
import pandas as pd

RUN_DIR = (
    Path("02_harvest_date")
    / "09_genetic_algorithm"
    / "output"
    / "04_ga_runs"
    / "G2B_no_soil_multiseed_variable_split_inner3cv"
)

REGION_STABILITY_FILE = RUN_DIR / "G2B_region_stability_across_seeds.csv"
SELECTED_REGIONS_LONG_FILE = RUN_DIR / "G2B_selected_regions_all_seeds_long.csv"

OUT_FILE = RUN_DIR / "G2B_region_stability_across_seeds_with_unique_snps.csv"


def main():
    print("=" * 80)
    print("G5 - ADD UNIQUE SNP LIST TO NO-SOIL REGION STABILITY TABLE")
    print("=" * 80)

    if not REGION_STABILITY_FILE.exists():
        raise FileNotFoundError(f"Missing file: {REGION_STABILITY_FILE}")

    if not SELECTED_REGIONS_LONG_FILE.exists():
        raise FileNotFoundError(f"Missing file: {SELECTED_REGIONS_LONG_FILE}")

    region_stability = pd.read_csv(REGION_STABILITY_FILE)
    selected_long = pd.read_csv(SELECTED_REGIONS_LONG_FILE)

    required_cols_stability = {"region"}
    required_cols_long = {"region", "snp", "seed"}

    missing_stability = required_cols_stability - set(region_stability.columns)
    missing_long = required_cols_long - set(selected_long.columns)

    if missing_stability:
        raise ValueError(f"Missing columns in region stability file: {missing_stability}")

    if missing_long:
        raise ValueError(f"Missing columns in selected regions long file: {missing_long}")

    selected_long["region"] = selected_long["region"].astype(str)
    selected_long["snp"] = selected_long["snp"].astype(str)
    selected_long["seed"] = selected_long["seed"].astype(str)

    # For each region:
    # - list unique SNPs selected at least once
    # - count unique SNPs, as a check
    # - optionally list SNPs per seed in compact form
    snp_summary = (
        selected_long
        .groupby("region")
        .agg(
            unique_snps_selected=(
                "snp",
                lambda x: ";".join(sorted(pd.Series(x).dropna().astype(str).unique()))
            ),
            n_unique_snps_selected_recomputed=(
                "snp",
                lambda x: pd.Series(x).dropna().astype(str).nunique()
            ),
            seeds_recomputed=(
                "seed",
                lambda x: ",".join(sorted(pd.Series(x).dropna().astype(str).unique(), key=lambda z: int(float(z))))
            ),
        )
        .reset_index()
    )

    out = region_stability.merge(
        snp_summary,
        on="region",
        how="left"
    )

    # Put unique SNPs near n_unique_snps_selected
    preferred_order = [
        "region",
        "n_seeds_selected",
        "n_selected_snp_events",
        "n_unique_snps_selected",
        "n_unique_snps_selected_recomputed",
        "unique_snps_selected",
        "seeds",
        "seeds_recomputed",
        "selection_frequency",
    ]

    remaining_cols = [c for c in out.columns if c not in preferred_order]
    ordered_cols = [c for c in preferred_order if c in out.columns] + remaining_cols
    out = out[ordered_cols]

    out.to_csv(OUT_FILE, index=False)

    print(f"Input region stability rows: {region_stability.shape[0]}")
    print(f"Input long selected rows: {selected_long.shape[0]}")
    print(f"Output rows: {out.shape[0]}")
    print()
    print(f"Saved:")
    print(OUT_FILE)

    print()
    print("Preview:")
    preview_cols = [
        "region",
        "n_seeds_selected",
        "n_selected_snp_events",
        "n_unique_snps_selected",
        "unique_snps_selected",
        "seeds",
        "selection_frequency",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]
    print(out[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
