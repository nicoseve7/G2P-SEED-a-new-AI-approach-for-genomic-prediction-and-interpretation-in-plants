# -*- coding: utf-8 -*-

################################################################################
### D8_generate_numpy_arrays_harvest.py
### Adattamento di D_Generate_Numpy_arr.py al caso Harvest_date
################################################################################

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------------------------
data_dir = os.path.join(
    "02_harvest_date",
    "06_deep_learning_baseline",
    "output",
    "numpy_arrays_harvest"
)

snp_dir = os.path.join(data_dir, "snps")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(snp_dir, exist_ok=True)

pheno_file = os.path.join(
    "02_harvest_date",
    "04_input_preparation",
    "output",
    "master_alignment_table_with_PCs.csv"
)

weather_file = os.path.join(
    "02_harvest_date",
    "03_environment_preprocessing",
    "output",
    "Weather_daily.csv"
)

pca_file = os.path.join(
    "01_common_genomic_preprocessing",
    "output",
    "genomic_PCs_20_paper_style.csv"
)

soil_file = os.path.join(
    "02_harvest_date",
    "03_environment_preprocessing",
    "output",
    "soil_by_envir.csv"
)

cv_file = os.path.join(
    "data",
    "raw",
    "cv",
    "Harvest_date_CV.csv"
)

geno_split_dir = os.path.join(
    "02_harvest_date",
    "05_gradient_boosting",
    "output",
    "geno_files",
    "Harvest_date"
)

# ------------------------------------------------------------------------------
# 2. Load base data
# ------------------------------------------------------------------------------
print("Loading phenotype table...")
data = pd.read_csv(pheno_file)

# keep only columns useful for sample identity + target
data = data[["Genotype", "Envir", "Harvest_date"]].copy()

# add year/start/end as in repo logic
data["Year"] = data["Envir"].str.split(".").str[1]
data["END"] = pd.to_datetime(data["Year"] + "-11-01")
data["START"] = pd.to_datetime(data["Year"] + "-01-01")

print(f"Phenotype rows: {data.shape[0]}")

# ------------------------------------------------------------------------------
# 3. Weather array
# ------------------------------------------------------------------------------
# print("Loading daily weather...")
# all_weather = pd.read_csv(weather_file)

# # expected columns from your B2:
# # Location, Day, Temperature_Dmean, Humidity_Dmean, Radiation_Dsum
# all_weather["Day"] = pd.to_datetime(all_weather["Day"])
# all_weather["Year"] = all_weather["Day"].dt.year.astype(str)
# all_weather["Envir"] = all_weather["Location"] + "." + all_weather["Year"]

# # repo logic: Jan 1 to Nov 1 minus 4 days => ~300 days
# n_days = (data.iloc[0]["END"] - data.iloc[0]["START"]).days - 4
# len_interval = 1
# n_samples = data.shape[0]
# n_variables = 3
# n_timesteps = int(n_days / len_interval)

# X_weather = np.zeros((n_samples, n_timesteps, n_variables), dtype=float)

# print(f"Building weather.npy with shape ({n_samples}, {n_timesteps}, {n_variables}) ...")

# for x in range(n_samples):
#     env = data.loc[x, "Envir"]
#     end_date = data.loc[x, "END"]

#     env_df = all_weather[all_weather["Envir"] == env].copy()
#     env_df = env_df[env_df["Day"] < end_date].sort_values("Day")

#     # tail to keep exactly n_days
#     df = env_df.tail(n_days)

#     if df.shape[0] != n_days:
#         raise ValueError(
#             f"Weather rows mismatch for sample {x}, env {env}: "
#             f"expected {n_days}, found {df.shape[0]}"
#         )

#     temp = df["Temperature_Dmean"].to_numpy()
#     hum = df["Humidity_Dmean"].to_numpy()
#     rad = df["Radiation_Dsum"].to_numpy()

#     data_reshaped = np.zeros((n_timesteps, n_variables), dtype=float)

#     for i in range(n_timesteps):
#         r1 = i * len_interval
#         r2 = (i + 1) * len_interval

#         data_reshaped[i, 0] = np.mean(temp[r1:r2])
#         data_reshaped[i, 1] = np.mean(hum[r1:r2])
#         data_reshaped[i, 2] = np.sum(rad[r1:r2])

#     X_weather[x, :, :] = data_reshaped

# np.save(os.path.join(data_dir, "weather.npy"), X_weather)
# print("Saved weather.npy")

print("Loading daily weather...")
all_weather = pd.read_csv(weather_file)

# expected columns:
# Location, Day, Temperature_Dmean, Humidity_Dmean, Radiation_Dsum
all_weather["Day"] = pd.to_datetime(all_weather["Day"])
all_weather["Year"] = all_weather["Day"].dt.year.astype(str)
all_weather["Envir"] = all_weather["Location"] + "." + all_weather["Year"]

# repo logic:
# START = Jan 1
# END = Nov 1
# n_days = (END - START).days - 4 = 300
# so we explicitly use Jan 5 -> Oct 31 (300 days)
n_days = 300
len_interval = 1
n_samples = data.shape[0]
n_variables = 3
n_timesteps = int(n_days / len_interval)

X_weather = np.zeros((n_samples, n_timesteps, n_variables), dtype=float)

weather_missing_report = []

print(f"Building weather.npy with shape ({n_samples}, {n_timesteps}, {n_variables}) ...")

for x in range(n_samples):
    env = data.loc[x, "Envir"]
    year = int(data.loc[x, "Year"])

    # # expected 300-day window: Jan 5 -> Oct 31
    # start_date = pd.Timestamp(year=year, month=1, day=5)
    # end_date = pd.Timestamp(year=year, month=10, day=31)
    # expected_days = pd.date_range(start=start_date, end=end_date, freq="D")

    # env_df = all_weather[all_weather["Envir"] == env].copy()
    # env_df = env_df[["Day", "Temperature_Dmean", "Humidity_Dmean", "Radiation_Dsum"]]
    # env_df = env_df.sort_values("Day").drop_duplicates(subset="Day")

    # # align to expected daily grid
    # df = env_df.set_index("Day").reindex(expected_days)

    # n_missing_days = int(df["Temperature_Dmean"].isna().sum())

    # if n_missing_days > 0:
    #     weather_missing_report.append({
    #         "sample_index": x,
    #         "Envir": env,
    #         "missing_days": n_missing_days
    #     })

    #     # interpolate missing daily rows, then fill any edge cases
    #     df = df.interpolate(method="linear", limit_direction="both")
    #     df = df.ffill().bfill()

    # if df.shape[0] != n_days:
    #     raise ValueError(
    #         f"Weather row count mismatch after reindex for sample {x}, env {env}: "
    #         f"expected {n_days}, found {df.shape[0]}"
    #     )

    # full expected window: Jan 1 -> Oct 31
    # then keep the last 300 days, following repo logic more closely
    start_date = pd.Timestamp(year=year, month=1, day=1)
    end_date = pd.Timestamp(year=year, month=10, day=31)
    expected_days_full = pd.date_range(start=start_date, end=end_date, freq="D")

    env_df = all_weather[all_weather["Envir"] == env].copy()
    env_df = env_df[["Day", "Temperature_Dmean", "Humidity_Dmean", "Radiation_Dsum"]]
    env_df = env_df.sort_values("Day").drop_duplicates(subset="Day")

    # align to full daily grid
    df_full = env_df.set_index("Day").reindex(expected_days_full)

    n_missing_days = int(df_full["Temperature_Dmean"].isna().sum())

    if n_missing_days > 0:
        weather_missing_report.append({
            "sample_index": x,
            "Envir": env,
            "missing_days": n_missing_days
        })

        # interpolate missing daily rows, then fill any edge cases
        df_full = df_full.interpolate(method="linear", limit_direction="both")
        df_full = df_full.ffill().bfill()

    # now keep the last 300 days exactly
    df = df_full.tail(n_days)

    if df.shape[0] != n_days:
        raise ValueError(
            f"Weather row count mismatch after tail for sample {x}, env {env}: "
            f"expected {n_days}, found {df.shape[0]}"
        )

    temp = df["Temperature_Dmean"].to_numpy(dtype=float)
    hum = df["Humidity_Dmean"].to_numpy(dtype=float)
    rad = df["Radiation_Dsum"].to_numpy(dtype=float)

    data_reshaped = np.zeros((n_timesteps, n_variables), dtype=float)

    for i in range(n_timesteps):
        r1 = i * len_interval
        r2 = (i + 1) * len_interval

        data_reshaped[i, 0] = np.mean(temp[r1:r2])
        data_reshaped[i, 1] = np.mean(hum[r1:r2])
        data_reshaped[i, 2] = np.sum(rad[r1:r2])

    X_weather[x, :, :] = data_reshaped

np.save(os.path.join(data_dir, "weather.npy"), X_weather)
print("Saved weather.npy")

# save a small report of repaired missing days
weather_missing_df = pd.DataFrame(weather_missing_report).drop_duplicates()
weather_missing_df.to_csv(
    os.path.join(data_dir, "weather_missing_days_report.csv"),
    index=False
)

if weather_missing_df.shape[0] > 0:
    print("Saved weather_missing_days_report.csv")
    print(weather_missing_df.head())
else:
    print("No missing weather days needed repair.")

# ------------------------------------------------------------------------------
# 4. PCA array
# ------------------------------------------------------------------------------
print("Loading PCA...")
PCA = pd.read_csv(pca_file)
PCA["Genotype"] = PCA["Genotype"].astype(str)

pc_cols = [c for c in PCA.columns if c.startswith("PC")]
n_comp = len(pc_cols)

X_PCA = np.zeros((n_samples, n_comp), dtype=float)

print(f"Building pca.npy with shape ({n_samples}, {n_comp}) ...")

for x in range(n_samples):
    g = str(data.loc[x, "Genotype"])
    row = PCA[PCA["Genotype"] == g]

    if row.shape[0] != 1:
        raise ValueError(f"PCA row mismatch for genotype {g}: found {row.shape[0]} rows")

    X_PCA[x, :] = row[pc_cols].to_numpy()[0]

np.save(os.path.join(data_dir, "pca.npy"), X_PCA)
print("Saved pca.npy")

# ------------------------------------------------------------------------------
# 5. Soil array
# ------------------------------------------------------------------------------
print("Loading soil by environment...")
soil_data = pd.read_csv(soil_file)

soil_cols = [c for c in soil_data.columns if c != "Envir"]
n_soil_parameters = len(soil_cols)

X_Soil = np.zeros((n_samples, n_soil_parameters), dtype=float)

print(f"Building soil.npy with shape ({n_samples}, {n_soil_parameters}) ...")

for x in range(n_samples):
    env = data.loc[x, "Envir"]
    row = soil_data[soil_data["Envir"] == env]

    if row.shape[0] != 1:
        raise ValueError(f"Soil row mismatch for environment {env}: found {row.shape[0]} rows")

    X_Soil[x, :] = row[soil_cols].to_numpy()[0]

np.save(os.path.join(data_dir, "soil.npy"), X_Soil)
print("Saved soil.npy")

# ------------------------------------------------------------------------------
# 6. SNP arrays per split
# ------------------------------------------------------------------------------
print("Loading CV strategy...")
cv = pd.read_csv(cv_file)
split_cols = cv.columns[2:].tolist()

print(f"Generating SNP arrays for {len(split_cols)} splits...")

# here we use all samples from master table, as in repo logic for pheno rows
# for each sample row, we map the genotype to the split-specific geno file
for split in split_cols:
    print(f"Processing SNP array for {split} ...")

    geno_file = os.path.join(geno_split_dir, f"geno_{split}.csv")
    snp_trait_split = pd.read_csv(geno_file)

    # first column is unnamed rownames from R -> rename to Genotype
    snp_trait_split = snp_trait_split.rename(columns={snp_trait_split.columns[0]: "Genotype"})

    # remove G_ prefix
    snp_trait_split["Genotype"] = snp_trait_split["Genotype"].astype(str).str.slice(2)

    snp_cols = [c for c in snp_trait_split.columns if c != "Genotype"]
    n_snps = len(snp_cols)

    X_SNP_trait = np.zeros((n_samples, n_snps), dtype=float)

    geno_lookup = snp_trait_split.set_index("Genotype")

    for x in range(n_samples):
        g = str(data.loc[x, "Genotype"])

        if g not in geno_lookup.index:
            raise ValueError(f"Genotype {g} not found in {geno_file}")

        X_SNP_trait[x, :] = geno_lookup.loc[g, snp_cols].to_numpy(dtype=float)

    out_file = os.path.join(snp_dir, f"SNP_Harvest_date_{split}.npy")
    np.save(out_file, X_SNP_trait)
    print(f"Saved {out_file} with shape {X_SNP_trait.shape}")

# ------------------------------------------------------------------------------
# 7. Save sample metadata for reference
# ------------------------------------------------------------------------------
sample_meta = data[["Genotype", "Envir", "Harvest_date"]].copy()
sample_meta.to_csv(os.path.join(data_dir, "sample_metadata_harvest.csv"), index=False)

print("\nAll numpy arrays generated successfully.")
print(f"Output directory: {data_dir}")
