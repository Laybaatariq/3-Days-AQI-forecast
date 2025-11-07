# ==========================================================
# cleaned_fg_timestamp.py
# Author: Laiba Tariq
# ==========================================================

import hopsworks
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# ----------------------------------------------------------
# Connect to Hopsworks
# ----------------------------------------------------------
project = hopsworks.login()
fs = project.get_feature_store()
print("Connected to Hopsworks project")

# ----------------------------------------------------------
# Delete wrong Feature Group version (V2)
# ----------------------------------------------------------
CLEAN_FEATURE_GROUP = "cleaned_aqi_data"

try:
    fs.delete_feature_group(name=CLEAN_FEATURE_GROUP, version=2)
    print(" Deleted Feature Group 'cleaned_aqi_data' version 2 successfully.")
except Exception as e:
    print(f" Skipped delete (V2 may not exist): {e}")

# ----------------------------------------------------------
# Load raw feature group
# ----------------------------------------------------------
RAW_FEATURE_GROUP = "karachi_aqi_weather"
RAW_VERSION = 1

raw_fg = fs.get_feature_group(name=RAW_FEATURE_GROUP, version=RAW_VERSION)
df = raw_fg.read()
print(f" Raw data fetched (rows={df.shape[0]}, cols={df.shape[1]})")

# ----------------------------------------------------------
# Fix 'time' column to timestamp
# ----------------------------------------------------------
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday
    print(" Time column fixed and features extracted (timestamp dtype).")
else:
    raise ValueError(" 'time' column missing in raw dataset!")

# ----------------------------------------------------------
# Outlier Capping
# ----------------------------------------------------------
pollutant_cols = [
    "pm2_5", "pm10", "ozone",
    "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "temperature_2m", "wind_speed_10m", "pressure_msl", "relative_humidity_2m"
]

def remove_outliers_iqr(df, columns, method="cap"):
    df_clean = df.copy()
    for col in columns:
        if col not in df_clean.columns:
            continue
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        if method == "cap":
            df_clean[col] = np.where(df_clean[col] > upper, upper,
                                     np.where(df_clean[col] < lower, lower, df_clean[col]))
    return df_clean

df_cleaned = remove_outliers_iqr(df, pollutant_cols, method="cap").dropna()
print(f" Outliers capped. Cleaned shape: {df_cleaned.shape}")

# ----------------------------------------------------------
# Ensure correct dtypes
# ----------------------------------------------------------
for col in ['year', 'month', 'day', 'hour', 'weekday']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].astype(np.int64)

print(" Data types fixed (timestamp + bigint ints)")

# ----------------------------------------------------------
# Recreate and Upload to Feature Group (V1)
# ----------------------------------------------------------
CLEAN_VERSION = 1

try:
    cleaned_fg = fs.get_or_create_feature_group(
        name=CLEAN_FEATURE_GROUP,
        version=CLEAN_VERSION,
        primary_key=["time"],   #  timestamp PK
        description="Cleaned AQI data with timestamp type for time column",
        online_enabled=True
    )
    cleaned_fg.insert(df_cleaned, write_options={"wait_for_job": False})
    print(f" Successfully recreated and uploaded '{CLEAN_FEATURE_GROUP}' (version={CLEAN_VERSION})")

except Exception as e:
    print(f" Upload failed: {e}")
