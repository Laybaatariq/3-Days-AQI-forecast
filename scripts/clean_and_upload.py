# ==========================================================
# clean_and_upload.py
# Purpose: Read raw AQI data → Normalize → Remove outliers → Upload cleaned version
# Author: Laiba Tariq
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
RAW_FEATURE_GROUP = "Karachi_aqi_weather"   # ⚠️ Check this exact name in your Hopsworks dashboard
RAW_VERSION = 1                             # update if version differs

try:
    raw_fg = fs.get_feature_group(name=RAW_FEATURE_GROUP, version=RAW_VERSION)
    if raw_fg is None:
        raise ValueError("Feature group not found — returned None.")
    df = raw_fg.read()
    print(f"✅ Raw data fetched from feature group '{RAW_FEATURE_GROUP}' (rows={df.shape[0]}, cols={df.shape[1]})")
except Exception as e:
    raise RuntimeError(f"❌ Error fetching raw feature group '{RAW_FEATURE_GROUP}': {e}")

# ==========================================================
# — Preprocessing: Handle Skewness (PowerTransform)
# ==========================================================
# Select numeric features (excluding time and cloud_cover)
num_features = df.select_dtypes(include=['float64', 'int64']).columns.drop('cloud_cover', errors='ignore')

if len(num_features) == 0:
    raise ValueError("No numeric features found for transformation!")

pt = PowerTransformer(method='yeo-johnson')
df[num_features] = pt.fit_transform(df[num_features])
print("✅ Left-skewed features normalized using Yeo-Johnson PowerTransformer")

# Cloud cover (right-skewed) → log1p transform
if 'cloud_cover' in df.columns:
    df['cloud_cover'] = np.log1p(df['cloud_cover'])

# ==========================================================
# — Outlier Handling using IQR
# ==========================================================
pollutant_cols = [
    'pm2_5', 'pm10', 'ozone',
    'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide',
    'temperature_2m', 'wind_speed_10m', 'pressure_msl', 'relative_humidity_2m'
]

def remove_outliers_iqr(df, columns, method="cap"):
    """
    method = "remove"  -> drop rows containing outliers
    method = "cap"     -> cap outlier values to upper/lower bounds
    """
    df_clean = df.copy()
    for col in columns:
        if col not in df_clean.columns:
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        if method == "remove":
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        elif method == "cap":
            df_clean[col] = np.where(df_clean[col] > upper, upper,
                                     np.where(df_clean[col] < lower, lower, df_clean[col]))
    return df_clean

df_cleaned = remove_outliers_iqr(df, pollutant_cols, method="cap")
print(f"✅ Outlier capping completed (Before={df.shape}, After={df_cleaned.shape})")

# Drop missing values just in case
df_cleaned = df_cleaned.dropna()
print(f"✅ Cleaned data shape after dropping NAs: {df_cleaned.shape}")

# ==========================================================
# — Upload to Cleaned Feature Group
# ==========================================================
CLEAN_FEATURE_GROUP = "cleaned_aqi_data"
CLEAN_VERSION = 1

try:
    cleaned_fg = fs.get_feature_group(name=CLEAN_FEATURE_GROUP, version=CLEAN_VERSION)
    if cleaned_fg is not None:
        print(f"✅ Found existing feature group '{CLEAN_FEATURE_GROUP}' — appending new data")
    else:
        raise ValueError("Feature group not found.")
except:
    print(f"⚙️ Creating new feature group '{CLEAN_FEATURE_GROUP}'...")
    cleaned_fg = fs.create_feature_group(
        name=CLEAN_FEATURE_GROUP,
        version=CLEAN_VERSION,
        primary_key=["time"],
        description="Cleaned AQI data after PowerTransform normalization and outlier handling",
        online_enabled=True
    )

# Append cleaned data
cleaned_fg.insert(df_cleaned, write_options={"wait_for_job": False})
print("🎉 Cleaned data appended successfully to Feature Store!")
