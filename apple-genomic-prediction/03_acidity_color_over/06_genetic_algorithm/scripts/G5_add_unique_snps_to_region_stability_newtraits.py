# -*- coding: utf-8 -*-

################################################################################
### G5_add_unique_snps_to_region_stability_newtraits.py
###
### Adds a column with the list of unique SNPs selected in each 50kb region
### for:
###   - Acidity
###   - Color_over
################################################################################

from pathlib import Path
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

TRAITS = ["Acidity", "Color_over"]

BASE_RUN_DIR = (
    Path("03_acidity_color_over")
    / "06_genetic_algorithm"
    / "output"
    / "04_ga_runs"
    / "G2B_multiseed_ga_newtraits"
)


# =============================================================================
# HELPERS
# =============================================================================

def sort_seed_strings(values):
    """
    Sort seed values even if they are read as strings/floats.
    Example: ['42.0', '43', '44'] -> ['42', '43', '44']
    """
    cleaned = []

    for v in values:
        if pd.isna(v):
            continue

        s = str(v).strip()

        if s == "" or s.lower() == "nan":
            continue

        try:
            s_clean = str(int(float(s)))
        except Exception:
            s_clean = s

        cleaned.append(s_clean)

    try:
        return sorted(set(cleaned), key=lambda z: int(float(z)))
    except Exception:
        return sorted(set(cleaned))


def build_paths(trait: str):
    run_dir = BASE_RUN_DIR / trait

    region_stability_file = run_dir / f"G2B_region_stability_across_seeds_{trait}.csv"
    selected_regions_long_file = run_dir / f"G2B_selected_regions_all_seeds_long_{trait}.csv"

    out_file = run_dir / f"G2B_region_stability_across_seeds_with_unique_snps_{trait}.csv"

    return run_dir, region_stability_file, selected_regions_long_file, out_file


def process_one_trait(trait: str):
    print("\n" + "=" * 80)
    print(f"G5 - ADD UNIQUE SNP LIST TO REGION STABILITY TABLE: {trait}")
    print("=" * 80)

    run_dir, region_stability_file, selected_regions_long_file, out_file = build_paths(trait)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found for {trait}:\n{run_dir}")

    if not region_stability_file.exists():
        raise FileNotFoundError(f"Missing file:\n{region_stability_file}")

    if not selected_regions_long_file.exists():
        raise FileNotFoundError(f"Missing file:\n{selected_regions_long_file}")

    region_stability = pd.read_csv(region_stability_file)
    selected_long = pd.read_csv(selected_regions_long_file)

    required_cols_stability = {"region"}
    required_cols_long = {"region", "snp", "seed"}

    missing_stability = required_cols_stability - set(region_stability.columns)
    missing_long = required_cols_long - set(selected_long.columns)

    if missing_stability:
        raise ValueError(
            f"Missing columns in region stability file for {trait}: {missing_stability}\n"
            f"Columns found: {region_stability.columns.tolist()}"
        )

    if missing_long:
        raise ValueError(
            f"Missing columns in selected regions long file for {trait}: {missing_long}\n"
            f"Columns found: {selected_long.columns.tolist()}"
        )

    region_stability["region"] = region_stability["region"].astype(str).str.strip()

    selected_long["region"] = selected_long["region"].astype(str).str.strip()
    selected_long["snp"] = selected_long["snp"].astype(str).str.strip()
    selected_long["seed"] = selected_long["seed"].astype(str).str.strip()

    # -------------------------------------------------------------------------
    # For each region:
    # - list unique SNPs selected at least once
    # - recount unique SNPs as a check
    # - recount seeds as a check
    # - optionally list seed:SNPs compactly
    # -------------------------------------------------------------------------
    snp_summary = (
        selected_long
        .groupby("region")
        .agg(
            unique_snps_selected=(
                "snp",
                lambda x: ";".join(
                    sorted(pd.Series(x).dropna().astype(str).str.strip().unique())
                )
            ),
            n_unique_snps_selected_recomputed=(
                "snp",
                lambda x: pd.Series(x).dropna().astype(str).str.strip().nunique()
            ),
            seeds_recomputed=(
                "seed",
                lambda x: ",".join(sort_seed_strings(x))
            ),
        )
        .reset_index()
    )

    # SNP list by seed, useful for checking whether the same SNPs recur
    seed_snp_summary = (
        selected_long
        .groupby(["region", "seed"])
        .agg(
            snps_selected_in_seed=(
                "snp",
                lambda x: ";".join(
                    sorted(pd.Series(x).dropna().astype(str).str.strip().unique())
                )
            )
        )
        .reset_index()
    )

    seed_snp_summary["seed_snp_entry"] = (
        "seed_" + seed_snp_summary["seed"].astype(str)
        + ":"
        + seed_snp_summary["snps_selected_in_seed"].astype(str)
    )

    seed_snp_summary_region = (
        seed_snp_summary
        .groupby("region")
        .agg(
            unique_snps_selected_by_seed=(
                "seed_snp_entry",
                lambda x: " | ".join(pd.Series(x).dropna().astype(str).tolist())
            )
        )
        .reset_index()
    )

    snp_summary = snp_summary.merge(
        seed_snp_summary_region,
        on="region",
        how="left"
    )

    out = region_stability.merge(
        snp_summary,
        on="region",
        how="left"
    )

    # -------------------------------------------------------------------------
    # Consistency check
    # -------------------------------------------------------------------------
    if "n_unique_snps_selected" in out.columns:
        out["unique_snp_count_match"] = (
            pd.to_numeric(out["n_unique_snps_selected"], errors="coerce")
            ==
            pd.to_numeric(out["n_unique_snps_selected_recomputed"], errors="coerce")
        )

    # Put useful columns near each other
    preferred_order = [
        "region",
        "n_seeds_selected",
        "n_selected_snp_events",
        "n_unique_snps_selected",
        "n_unique_snps_selected_recomputed",
        "unique_snp_count_match",
        "unique_snps_selected",
        "unique_snps_selected_by_seed",
        "seeds",
        "seeds_recomputed",
        "selection_frequency",
    ]

    remaining_cols = [c for c in out.columns if c not in preferred_order]
    ordered_cols = [c for c in preferred_order if c in out.columns] + remaining_cols
    out = out[ordered_cols]

    out.to_csv(out_file, index=False)

    print(f"Input region stability rows: {region_stability.shape[0]}")
    print(f"Input long selected rows: {selected_long.shape[0]}")
    print(f"Output rows: {out.shape[0]}")
    print()
    print("Saved:")
    print(out_file)

    print()
    print("Preview:")
    preview_cols = [
        "region",
        "n_seeds_selected",
        "n_selected_snp_events",
        "n_unique_snps_selected",
        "n_unique_snps_selected_recomputed",
        "unique_snp_count_match",
        "unique_snps_selected",
        "seeds",
        "selection_frequency",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]
    print(out[preview_cols].head(10).to_string(index=False))

    return {
        "Trait": trait,
        "region_stability_rows": region_stability.shape[0],
        "selected_long_rows": selected_long.shape[0],
        "output_rows": out.shape[0],
        "out_file": str(out_file),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("G5 - ADD UNIQUE SNP LIST TO REGION STABILITY TABLES FOR NEW TRAITS")
    print("=" * 80)

    rows = []

    for trait in TRAITS:
        rows.append(process_one_trait(trait))

    summary = pd.DataFrame(rows)

    summary_file = BASE_RUN_DIR / "G5_unique_snps_region_stability_summary_all_traits.csv"
    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 80)
    print("G5 completed for all traits.")
    print("=" * 80)
    print(summary.to_string(index=False))
    print()
    print("Saved summary:")
    print(summary_file)


if __name__ == "__main__":
    main()
