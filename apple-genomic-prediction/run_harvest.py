from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


# =============================================================================
# HELPERS
# =============================================================================

def find_script(folder: Path, prefix: str) -> Path:
    """
    Find exactly one Python or R script in 'folder'
    whose filename starts with 'prefix'.
    """

    if not folder.exists():
        raise FileNotFoundError(
            f"Scripts folder not found:\n{folder}"
        )

    candidates = sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.name.startswith(prefix)
        and p.suffix.lower() in {".py", ".r"}
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No script starting with '{prefix}' found in:\n{folder}"
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"More than one script starts with '{prefix}' in:\n{folder}\n\n"
            + "\n".join(str(p) for p in candidates)
        )

    return candidates[0]


def run_script(script: Path):
    """
    Run one Python or R script from the repository root.

    check=True means that the pipeline stops immediately
    if the script returns an error.
    """

    relative_script = script.relative_to(ROOT)

    print("\n" + "=" * 80)
    print(f"RUNNING: {relative_script}")
    print("=" * 80)

    suffix = script.suffix.lower()

    if suffix == ".py":
        cmd = [sys.executable, str(script)]

    elif suffix == ".r":
        cmd = ["Rscript", str(script)]

    else:
        raise ValueError(
            f"Unsupported script type: {script}"
        )

    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
    )

    print(f"\nCOMPLETED: {relative_script}")


def run_prefix(folder: Path, prefix: str):
    """
    Find a script by prefix and run it.
    """
    script = find_script(folder, prefix)
    run_script(script)


def run_block(title: str, folder: Path, prefixes):
    """
    Run a sequence of scripts belonging to the same pipeline block.
    """

    print("\n\n" + "#" * 80)
    print(title)
    print("#" * 80)

    for prefix in prefixes:
        run_prefix(folder, prefix)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    print("\n" + "#" * 80)
    print("HARVEST DATE FULL PIPELINE")
    print("#" * 80)

    harvest = ROOT / "02_harvest_date"

    # -------------------------------------------------------------------------
    # Script directories
    # -------------------------------------------------------------------------

    audit_dir = (
        harvest
        / "01_data_audit"
        / "scripts"
    )

    phenotype_dir = (
        harvest
        / "02_phenotype_preprocessing"
        / "scripts"
    )

    environment_dir = (
        harvest
        / "03_environment_preprocessing"
        / "scripts"
    )

    input_dir = (
        harvest
        / "04_input_preparation"
        / "scripts"
    )

    gb_dir = (
        harvest
        / "05_gradient_boosting"
        / "scripts"
    )

    baseline_dir = (
        harvest
        / "06_deep_learning_baseline"
        / "scripts"
    )

    nn_dir = (
        harvest
        / "07_neural_network"
        / "scripts"
    )

    weather_dir = (
        nn_dir
        / "weather_features"
    )

    shap_dir = (
        harvest
        / "08_shap"
        / "scripts"
    )

    ga_dir = (
        harvest
        / "09_genetic_algorithm"
        / "scripts"
    )

    # =========================================================================
    # 1. DATA AUDIT
    # =========================================================================

    run_block(
        title="1. DATA AUDIT",
        folder=audit_dir,
        prefixes=[
            "A1_",
            "A2_",
            "A3_",
        ],
    )

    # =========================================================================
    # 2. PHENOTYPE PREPROCESSING
    # =========================================================================

    run_block(
        title="2. PHENOTYPE PREPROCESSING",
        folder=phenotype_dir,
        prefixes=[
            "A4_",
            "A5_",
            "A6_",
            "A7_",
            "A8_",
        ],
    )

    # =========================================================================
    # 3. ENVIRONMENT PREPROCESSING
    # =========================================================================

    run_block(
        title="3. ENVIRONMENT PREPROCESSING",
        folder=environment_dir,
        prefixes=[
            "B1_",
            "B2_",
            "B3_",
            "B4_",
            "B5_",
            "B6_",
            "B7_",
        ],
    )

    # =========================================================================
    # 4. HARVEST-SPECIFIC INPUT PREPARATION
    # =========================================================================

    # C3 and C5 are NOT run here.
    # They belong to run_preprocessing.py and are shared across traits.

    run_block(
        title="4. HARVEST-SPECIFIC INPUT PREPARATION",
        folder=input_dir,
        prefixes=[
            "C1_",
            "C2_",
            "C4_",
            "C6_",
        ],
    )

    # =========================================================================
    # 5. GRADIENT BOOSTING FEATURE SELECTION
    # =========================================================================

    run_block(
        title="5. GRADIENT BOOSTING FEATURE SELECTION",
        folder=gb_dir,
        prefixes=[
            "D3_",
            "D4_",
            "D5_",
            "D7_",
        ],
    )

    # =========================================================================
    # 6. DEEP-LEARNING BASELINE
    # =========================================================================

    run_block(
        title="6. DEEP-LEARNING BASELINE",
        folder=baseline_dir,
        prefixes=[
            "D8_",
            "D10_",
            "D10b_",
            "D11_",
            "D11b_",
            "D11c_",
        ],
    )

    # =========================================================================
    # 7. FINAL NO-SOIL NEURAL NETWORK
    # =========================================================================

    print("\n\n" + "#" * 80)
    print("7. FINAL NO-SOIL NEURAL NETWORK")
    print("#" * 80)

    # -------------------------------------------------------------------------
    # 7A. Biological objects
    # -------------------------------------------------------------------------

    for prefix in [
        "F0_",
        "F1_",
        "V1_",
    ]:
        run_prefix(nn_dir, prefix)

    # -------------------------------------------------------------------------
    # 7B. Expanded weather features
    # -------------------------------------------------------------------------

    for prefix in [
        "M0_",
        "M1_",
        "M2_",
        "M3_",
        "M4_",
        "M5_",
    ]:
        run_prefix(weather_dir, prefix)

    # -------------------------------------------------------------------------
    # 7C. No-soil tuning, training and evaluation
    # -------------------------------------------------------------------------

    for prefix in [
        "W1_",
        "W2_",
        "W3_",
    ]:
        run_prefix(nn_dir, prefix)

    # =========================================================================
    # 8. SHAP INTERPRETATION - NO SOIL
    # =========================================================================

    run_block(
        title="8. SHAP INTERPRETATION - NO SOIL",
        folder=shap_dir,
        prefixes=[
            "H1_compute_",
            "H1b_",
            "H2_",
            "H3_",
        ],
    )

    # =========================================================================
    # 9. REGION CONSTRUCTION + GENETIC ALGORITHM
    # =========================================================================

    run_block(
        title="9. REGION CONSTRUCTION AND GENETIC ALGORITHM",
        folder=ga_dir,
        prefixes=[
            "R0_",
            "R1_",
            "R3_",
            "G0_",
            "G2D_",
            "G2B_",
            "G3_",
            "G4_",
            "G5_",
            "G6_",
        ],
    )

    # =========================================================================
    # DONE
    # =========================================================================

    print("\n\n" + "#" * 80)
    print("HARVEST DATE PIPELINE COMPLETED SUCCESSFULLY")
    print("#" * 80)


if __name__ == "__main__":
    main()
