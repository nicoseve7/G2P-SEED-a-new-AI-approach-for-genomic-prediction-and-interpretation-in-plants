# Questo fa controlli rapidi su:

# numero righe
# environment match
# shape dei numpy

# -*- coding: utf-8 -*-

################################################################################
### M4_check_alignment_weather_inputs.py
### Quick diagnostics for derived weather inputs
################################################################################

from pathlib import Path
import numpy as np
import pandas as pd

OUT_DIR = (
    Path("02_harvest_date")
    / "07_neural_network"
    / "output"
    / "weather_features"
)

REPORT_DIR = OUT_DIR / "diagnostics"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

META_FILE = (
    Path("02_harvest_date")
    / "06_deep_learning_baseline"
    / "output"
    / "numpy_arrays_harvest"
    / "sample_metadata_harvest.csv"
)

VERSIONS = ["v2", "v3"]


def main():
    meta = pd.read_csv(META_FILE)
    meta["Envir"] = meta["Envir"].astype(str).str.strip()

    report_lines = []
    report_lines.append("=== WEATHER INPUT ALIGNMENT REPORT ===\n")

    report_lines.append(f"Metadata rows: {len(meta)}\n")
    report_lines.append(f"Unique environments in metadata: {meta['Envir'].nunique()}\n\n")

    for version in VERSIONS:
        npy_file = OUT_DIR / f"weather_period_features_{version}.npy"
        cols_file = OUT_DIR / f"weather_period_features_{version}_columns.csv"
        aligned_file = OUT_DIR / f"weather_period_features_{version}_aligned.csv"

        if not npy_file.exists():
            report_lines.append(f"[{version}] Missing numpy file: {npy_file}\n")
            continue

        X = np.load(npy_file)
        cols = pd.read_csv(cols_file)
        aligned = pd.read_csv(aligned_file)

        report_lines.append(f"--- VERSION {version} ---\n")
        report_lines.append(f"Numpy shape: {X.shape}\n")
        report_lines.append(f"Number of columns file entries: {len(cols)}\n")
        report_lines.append(f"Aligned table rows: {len(aligned)}\n")
        report_lines.append(f"Aligned unique environments: {aligned['Envir'].nunique()}\n")

        if X.shape[0] != len(meta):
            report_lines.append("WARNING: number of rows in numpy does not match metadata\n")

        if X.shape[1] != len(cols):
            report_lines.append("WARNING: number of numpy columns does not match columns file\n")

        report_lines.append("\n")

    report_file = REPORT_DIR / "weather_input_alignment_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"Saved report: {report_file}")


if __name__ == "__main__":
    main()
