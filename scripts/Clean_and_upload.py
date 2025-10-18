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
# STEP 1️⃣ — Connect to Hopsworks
# ==========================================================
project = hopsworks.login()
fs = project.get_feature_store()
print("✅ Connected to Hopsworks project")

# ==========================================================
# STEP 2️⃣ — Load Raw Data Feature Group
# ==========================================================
RAW_FEATURE_GROUP = "Karachi_aqi_weather"
RAW_VERSION = 1  # change if your version differs

raw_fg = fs.get_feature_group(name=RAW_FEATURE_GROUP, version=RAW_VERSION)
df = raw_fg.read()

print(f"✅ Raw data fetched from feature group '{RAW_FEATURE_GROUP}'")
print("Shape:", df.shape)

# ==========================================================
# STEP 3️⃣ — Preprocessing: Handle Skewness (PowerTransform)
# ==========================================================
# Select numeric features (excluding time and cloud_cover)
num_features = df.select_dtypes(include=['float64', 'int64']).columns.drop('cloud_cover')

pt = PowerTransformer(method='yeo-johnson')
df[num_features] = pt.fit_transform(df[num_features])

print("✅ Left-skewed features normalized using Yeo-Johnson PowerTransformer")

# Cloud cover (right-skewed) → log1p transform
df['cloud_cover'] = np.log1p(df['cloud_cover'])

# ==========================================================
# STEP 4️⃣ — Outlier Handling using IQR
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
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if method == "remove":
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        elif method == "cap":
            df_clean[col] = np.where(df_clean[col] > upper, upper,
                                     np.where(df_clean[col] < lower, lower, df_clean[col]))
        else:
            raise ValueError("method must be 'remove' or 'cap'")
    return df_clean

df_cleaned = remove_outliers_iqr(df, pollutant_cols, method="cap")

print("✅ Outlier capping completed")
print("Before:", df.shape, "| After:", df_cleaned.shape)

# Drop missing values just in case
df_cleaned = df_cleaned.dropna()

# ==========================================================
# STEP 5️⃣ — Upload to Cleaned Feature Group
# ==========================================================
CLEAN_FEATURE_GROUP = "cleaned_aqi_data"
CLEAN_VERSION = 1  # update if you recreate

try:
    cleaned_fg = fs.get_feature_group(name=CLEAN_FEATURE_GROUP, version=CLEAN_VERSION)
    print(f"✅ Found existing feature group '{CLEAN_FEATURE_GROUP}' — appending new data")
except:
    cleaned_fg = fs.create_feature_group(
        name=CLEAN_FEATURE_GROUP,
        version=CLEAN_VERSION,
        primary_key=["time"],  # your time column
        description="Cleaned AQI data after PowerTransform normalization and outlier handling",
        online_enabled=True
    )
    print(f"✅ Created new feature group '{CLEAN_FEATURE_GROUP}'")

# Append cleaned data
cleaned_fg.insert(df_cleaned, write_options={"wait_for_job": False})
print("✅Cleaned data appended successfully to Feature Store!")

