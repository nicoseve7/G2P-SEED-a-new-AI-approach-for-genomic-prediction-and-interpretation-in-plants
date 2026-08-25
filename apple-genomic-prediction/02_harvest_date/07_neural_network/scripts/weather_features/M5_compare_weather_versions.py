# Questo confronta V2 e V3 in modo semplice:

# numero feature
# distribuzione per colonna
# correlazioni medie assolute

# -*- coding: utf-8 -*-

################################################################################
### M5_compare_weather_versions.py
### Simple comparison between weather feature versions
################################################################################

from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(".")
OUT_DIR = BASE_DIR / "Output" / "numpy_arrays_weather_exp"
REPORT_DIR = BASE_DIR / "Output" / "diagnostics"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def describe_version(version: str):
    npy_file = OUT_DIR / f"weather_period_features_{version}.npy"
    cols_file = OUT_DIR / f"weather_period_features_{version}_columns.csv"

    X = np.load(npy_file)
    cols = pd.read_csv(cols_file)["feature_name"].tolist()

    df = pd.DataFrame(X, columns=cols)
    summary = df.describe().T
    summary["feature_name"] = summary.index
    summary = summary.reset_index(drop=True)

    return df, summary


def main():
    df_v2, summary_v2 = describe_version("v2")
    df_v3, summary_v3 = describe_version("v3")

    summary_v2.to_csv(REPORT_DIR / "weather_v2_summary.csv", index=False)
    summary_v3.to_csv(REPORT_DIR / "weather_v3_summary.csv", index=False)

    comparison_lines = []
    comparison_lines.append("=== WEATHER VERSION COMPARISON ===\n\n")
    comparison_lines.append(f"V2 shape: {df_v2.shape}\n")
    comparison_lines.append(f"V3 shape: {df_v3.shape}\n\n")

    comparison_lines.append(f"V2 number of features: {df_v2.shape[1]}\n")
    comparison_lines.append(f"V3 number of features: {df_v3.shape[1]}\n\n")

    if df_v2.shape[1] > 1:
        corr_v2 = df_v2.corr().abs()
        mean_abs_corr_v2 = (corr_v2.values.sum() - len(corr_v2)) / (corr_v2.size - len(corr_v2))
        comparison_lines.append(f"V2 mean absolute pairwise correlation: {mean_abs_corr_v2:.4f}\n")

    if df_v3.shape[1] > 1:
        corr_v3 = df_v3.corr().abs()
        mean_abs_corr_v3 = (corr_v3.values.sum() - len(corr_v3)) / (corr_v3.size - len(corr_v3))
        comparison_lines.append(f"V3 mean absolute pairwise correlation: {mean_abs_corr_v3:.4f}\n")

    report_file = REPORT_DIR / "weather_versions_comparison.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.writelines(comparison_lines)

    print(f"Saved summaries and comparison report in {REPORT_DIR}")


if __name__ == "__main__":
    main()