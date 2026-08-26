from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_r(script: Path):
    print("\n" + "=" * 80)
    print(f"RUNNING: {script}")
    print("=" * 80)

    subprocess.run(
        ["Rscript", str(script)],
        cwd=ROOT,
        check=True,
    )


def main():
    print("\n" + "#" * 80)
    print("COMMON GENOMIC PREPROCESSING")
    print("#" * 80)

    scripts_dir = (
        ROOT
        / "01_common_genomic_preprocessing"
        / "scripts"
    )

    steps = [
        scripts_dir / "C3_build_genomic_PCs_paper_style.R",
        scripts_dir / "C5_build_SNP_matrix_for_modeling.R",
    ]

    for script in steps:
        if not script.exists():
            raise FileNotFoundError(
                f"Required script not found:\n{script}"
            )

        run_r(script)

    print("\n" + "#" * 80)
    print("COMMON GENOMIC PREPROCESSING COMPLETED")
    print("#" * 80)


if __name__ == "__main__":
    main()
