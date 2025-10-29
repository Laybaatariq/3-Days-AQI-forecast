# ==========================================================
# clean_and_upload.py  — Final Version
# ==========================================================

import hopsworks
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# ==========================================================
# — Connect to Hopsworks
# ==========================================================
project = hopsworks.login()
fs = project.get_feature_store()
print("✅ Connected to Hopsworks project")

# ==========================================================
# — Load Raw Data Feature Group
# ==========================================================
RAW_FEATURE_GROUP = "karachi_aqi_weather"
RAW_VERSION = 1

raw_fg = fs.get_feature_group(name=RAW_FEATURE_GROUP, version=RAW_VERSION)
df = raw_fg.read()
print(f"✅ Raw data fetched (rows={df.shape[0]}, cols={df.shape[1]})")

# ==========================================================
# — Time-based Feature Engineering
# ==========================================================
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday
    print("🕒 Added datetime features: year, month, day, hour, weekday")
else:
    raise ValueError("❌ 'time' column missing in raw dataset!")

# ==========================================================
# — Preprocessing: Normalize skewed features
# ==========================================================
num_features = df.select_dtypes(include=["float64", "int64"]).columns.drop("cloud_cover", errors="ignore")

if len(num_features) == 0:
    raise ValueError("No numeric features found for PowerTransform!")

pt = PowerTransformer(method="yeo-johnson")
df[num_features] = pt.fit_transform(df[num_features])
print("✅ Skewed features normalized using PowerTransformer")

if "cloud_cover" in df.columns:
    df["cloud_cover"] = np.log1p(df["cloud_cover"])

# ==========================================================
# — Outlier Handling using IQR
# ==========================================================
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
        if method == "remove":
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        elif method == "cap":
            df_clean[col] = np.where(df_clean[col] > upper, upper,
                                     np.where(df_clean[col] < lower, lower, df_clean[col]))
    return df_clean

df_cleaned = remove_outliers_iqr(df, pollutant_cols, method="cap")
df_cleaned = df_cleaned.dropna()
print(f"✅ Outliers capped and NaNs dropped. Final shape: {df_cleaned.shape}")

# ==========================================================
# — Upload to Cleaned Feature Group (Keep same version)
# ==========================================================
CLEAN_FEATURE_GROUP = "cleaned_aqi_data"
CLEAN_VERSION = 1

# Ensure datetime type for 'time' (not string)
if "time" in df_cleaned.columns:
    df_cleaned["time"] = pd.to_datetime(df_cleaned["time"], errors="coerce")
else:
    raise KeyError("❌ 'time' column missing — required as primary key!")

# Drop any irrelevant columns
if "Unnamed: 0" in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=["Unnamed: 0"])

# Fix integer time columns
for col in ['year', 'month', 'day', 'hour', 'weekday']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].astype(np.int64)

print("✅ Datetime + integer columns formatted correctly.")

# Create or update Feature Group safely
cleaned_fg = fs.get_or_create_feature_group(
    name=CLEAN_FEATURE_GROUP,
    version=CLEAN_VERSION,
    primary_key=["time"],  # Keep 'time' as timestamp PK
    description="Cleaned AQI data with timestamp and capped pollutants",
    online_enabled=True
)

# Insert data into version 1 (no new version)
cleaned_fg.insert(df_cleaned, write_options={"wait_for_job": False})
print(f"🎉 Successfully uploaded cleaned data to version {CLEAN_VERSION}")
