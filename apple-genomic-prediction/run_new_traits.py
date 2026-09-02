#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys


# =============================================================================
# PROJECT ROOT
# =============================================================================

ROOT = Path(__file__).resolve().parent

NEW_TRAITS_DIR = ROOT / "03_acidity_color_over"


# =============================================================================
# SCRIPT DIRECTORIES
# =============================================================================

AUDIT_DIR = (
    NEW_TRAITS_DIR
    / "01_data_audit"
    / "scripts"
)

PHENO_DIR = (
    NEW_TRAITS_DIR
    / "02_phenotype_preprocessing"
    / "scripts"
)

GB_DIR = (
    NEW_TRAITS_DIR
    / "03_gradient_boosting"
    / "scripts"
)

NN_DIR = (
    NEW_TRAITS_DIR
    / "04_neural_network"
    / "scripts"
)

SHAP_DIR = (
    NEW_TRAITS_DIR
    / "05_shap"
    / "scripts"
)

GA_DIR = (
    NEW_TRAITS_DIR
    / "06_genetic_algorithm"
    / "scripts"
)


# =============================================================================
# SHARED PREREQUISITES
# =============================================================================

REQUIRED_SHARED_FILES = [
    # Common genomic preprocessing
    ROOT
    / "01_common_genomic_preprocessing"
    / "output"
    / "SNP_matrix_modeling_var_gt0.RData",

    ROOT
    / "01_common_genomic_preprocessing"
    / "output"
    / "genomic_PCs_20_paper_style.csv",

    ROOT
    / "01_common_genomic_preprocessing"
    / "output"
    / "SNPs_final_2022.gds",

    # Harvest weather V3 reused by P6
    ROOT
    / "02_harvest_date"
    / "07_neural_network"
    / "output"
    / "weather_features"
    / "weather_period_features_v3_aligned.csv",

    ROOT
    / "02_harvest_date"
    / "07_neural_network"
    / "output"
    / "weather_features"
    / "weather_period_features_v3_columns.csv",

    # Global SNP -> gene mapping reused by Q0
    ROOT
    / "02_harvest_date"
    / "07_neural_network"
    / "output"
    / "biologic_objects"
    / "snp_gene_mapping"
    / "global_snp_gene_edges.csv",
]


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def check_shared_prerequisites():
    """
    Check outputs that are intentionally reused from common preprocessing
    and from the Harvest_date pipeline.
    """

    missing = [
        path
        for path in REQUIRED_SHARED_FILES
        if not path.exists()
    ]

    if missing:
        print_header("MISSING SHARED PREREQUISITES")

        print(
            "The Acidity/Color_over pipeline reuses some outputs from "
            "common preprocessing and Harvest_date.\n"
        )

        print("Missing files:")

        for path in missing:
            try:
                rel = path.relative_to(ROOT)
            except ValueError:
                rel = path

            print(f"  - {rel}")

        print(
            "\nRun the required common/Harvest preprocessing steps "
            "before running run_new_traits.py."
        )

        raise SystemExit(1)

    print("[OK] Shared prerequisites found.")


def find_script(folder: Path, prefix: str) -> Path:
    """
    Find exactly one .py or .R script starting with the requested prefix.

    Example:
        find_script(NN_DIR, "Q0b_")
    """

    if not folder.exists():
        raise FileNotFoundError(
            f"Script directory does not exist:\n{folder}"
        )

    candidates = sorted(
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.name.startswith(prefix)
            and path.suffix.lower() in {".py", ".r"}
        )
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No script found with prefix '{prefix}' in:\n{folder}"
        )

    if len(candidates) > 1:
        names = "\n".join(
            f"  - {p.name}"
            for p in candidates
        )

        raise RuntimeError(
            f"More than one script starts with '{prefix}' in:\n"
            f"{folder}\n\n"
            f"Candidates:\n{names}"
        )

    return candidates[0]


def run_script(script: Path):
    """
    Execute Python scripts with the current Python interpreter
    and R scripts with Rscript.

    All scripts are executed with cwd=ROOT so that every path in the
    repository is interpreted relative to the repository root.
    """

    if script.suffix.lower() == ".py":
        cmd = [
            sys.executable,
            str(script)
        ]

    elif script.suffix.lower() == ".r":
        cmd = [
            "Rscript",
            str(script)
        ]

    else:
        raise ValueError(
            f"Unsupported script type: {script}"
        )

    try:
        rel_script = script.relative_to(ROOT)
    except ValueError:
        rel_script = script

    print("\n" + "-" * 100)
    print(f"RUNNING: {rel_script}")
    print("-" * 100)

    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True
    )

    print(f"[OK] Completed: {script.name}")


def run_prefix(folder: Path, prefix: str):
    script = find_script(
        folder=folder,
        prefix=prefix
    )

    run_script(script)


def run_block(title: str, folder: Path, prefixes):
    print_header(title)

    for prefix in prefixes:
        run_prefix(
            folder=folder,
            prefix=prefix
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "ACIDITY + COLOR_OVER COMPLETE PIPELINE"
    )

    print(f"Repository root:\n{ROOT}\n")

    # -------------------------------------------------------------------------
    # 0. Shared prerequisites
    # -------------------------------------------------------------------------

    check_shared_prerequisites()

    # -------------------------------------------------------------------------
    # 1. Initial data audit
    # -------------------------------------------------------------------------

    run_block(
        title="1/6 - DATA AUDIT",
        folder=AUDIT_DIR,
        prefixes=[
            "P0_",
        ],
    )

    # -------------------------------------------------------------------------
    # 2. Phenotype preprocessing
    # -------------------------------------------------------------------------

    run_block(
        title="2/6 - PHENOTYPE PREPROCESSING",
        folder=PHENO_DIR,
        prefixes=[
            "P1_",
        ],
    )

    # -------------------------------------------------------------------------
    # 3. Gradient Boosting + split-specific genotype files
    # -------------------------------------------------------------------------

    run_block(
        title="3/6 - GRADIENT BOOSTING",
        folder=GB_DIR,
        prefixes=[
            "P2_",
            "P3_",
            "P4_",
            "P5_",
        ],
    )

    # -------------------------------------------------------------------------
    # 4. No-soil neural network
    # -------------------------------------------------------------------------

    run_block(
        title="4/6 - NO-SOIL NEURAL NETWORK",
        folder=NN_DIR,
        prefixes=[
            "P6_",
            "Q0_",
            "Q0b_",
            "Q1_",
            "Q2_",
            "Q3_",
        ],
    )

    # -------------------------------------------------------------------------
    # 5. SHAP interpretation
    # -------------------------------------------------------------------------

    run_block(
        title="5/6 - SHAP INTERPRETATION",
        folder=SHAP_DIR,
        prefixes=[
            "Q4_",
            "Q4b_",
            "Q5_",
            "Q6_",
        ],
    )

    # -------------------------------------------------------------------------
    # 6. Region ranking + Genetic Algorithm
    # -------------------------------------------------------------------------

    run_block(
        title="6/6 - GENOMIC REGIONS + GENETIC ALGORITHM",
        folder=GA_DIR,
        prefixes=[
            "Q7_",
            "Q8_",
            "Q9_",
            "G2D_",
            "G2B_",
            "G3_",
            "G4_",
            "G5_",
        ],
    )

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------

    print_header(
        "ACIDITY + COLOR_OVER PIPELINE COMPLETED SUCCESSFULLY"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
