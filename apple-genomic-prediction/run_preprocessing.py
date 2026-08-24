from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_r_script(relative_path):
    script = ROOT / relative_path

    print(f"\n{'=' * 70}")
    print(f"Running: {script.name}")
    print(f"{'=' * 70}\n")

    subprocess.run(
        ["Rscript", str(script)],
        cwd=ROOT,
        check=True
    )


def main():

    run_r_script(
        "01_common_genomic_preprocessing/scripts/"
        "C3_build_genomic_PCs_paper_style.R"
    )

    run_r_script(
        "01_common_genomic_preprocessing/scripts/"
        "C5_build_SNP_matrix_for_modeling.R"
    )

    print("\nCommon genomic preprocessing completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print("\nPreprocessing failed.")
        sys.exit(error.returncode)
