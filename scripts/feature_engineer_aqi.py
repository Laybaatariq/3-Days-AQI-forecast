# ============================================================
# 🧩 FEATURE ENGINEERING PIPELINE — AQI 3-Day Forecast (with Deduplication)
# ============================================================
# This script:
# 1. Fetches cleaned data from Feature Store (cleaned_aqi_data)
# 2. Runs feature engineering
# 3. Checks for duplicates
# 4. Uploads only NEW data to FS (aqi_feature_engineered)
# ============================================================

import hopsworks
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 🧠 Connect to Hopsworks & Fetch Cleaned Data
# ------------------------------------------------------------
project = hopsworks.login()
fs = project.get_feature_store()

print("📥 Fetching cleaned AQI data from Feature Store...")
cleaned_fg = fs.get_feature_group(name="cleaned_aqi_data", version=1)
df_cleaned = cleaned_fg.read()
print(f"✅ Cleaned data fetched — Shape: {df_cleaned.shape}")

# ------------------------------------------------------------
# 🧩 1️⃣ Datetime Feature Extraction
# ------------------------------------------------------------
def process_datetime_features(df):
    df = df.copy()
    time_col = None
    for col in df.columns:
        if col.lower() in ['time', 'timestamp', 'datetime', 'date']:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df['year'] = df[time_col].dt.year
        df['month'] = df[time_col].dt.month
        df['day'] = df[time_col].dt.day
        df['hour'] = df[time_col].dt.hour
        df['weekday'] = df[time_col].dt.weekday
        df['day_of_year'] = df[time_col].dt.dayofyear
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df.drop(columns=[time_col], inplace=True)
        print(f"✅ Extracted datetime features from '{time_col}'")
    else:
        print("⚠️ No time column found — skipping datetime extraction.")
    return df

# ------------------------------------------------------------
# 🧩 2️⃣ Season Encoding
# ------------------------------------------------------------
def add_and_encode_season(df):
    df = df.copy()
    if 'month' in df.columns and 'season' not in df.columns:
        def assign_season(m):
            if m in [12, 1, 2]:
                return 'winter'
            elif m in [3, 4, 5]:
                return 'spring'
            elif m in [6, 7, 8]:
                return 'summer'
            else:
                return 'autumn'
        df['season'] = df['month'].apply(assign_season)
        print("🌦 Added 'season' column from 'month'.")

    if 'season' in df.columns:
        df = pd.get_dummies(df, columns=['season'], prefix='season', dtype=float)
        print("✅ One-hot encoded 'season' column.")
    else:
        print("⚠️ 'season' column not found — skipping encoding.")
    return df

# ------------------------------------------------------------
# 🧩 3️⃣ Lag, Rolling, and Diff Features
# ------------------------------------------------------------
def add_aqi_lag_features(df):
    df = df.copy()
    aqi_col = None
    for col in df.columns:
        if col.lower() == 'aqi':
            aqi_col = col
            break
    if aqi_col:
        df[f'{aqi_col}_lag1'] = df[aqi_col].shift(1)
        df[f'{aqi_col}_lag2'] = df[aqi_col].shift(2)
        df[f'{aqi_col}_rolling_mean3'] = df[aqi_col].rolling(window=3).mean()
        df[f'{aqi_col}_diff'] = df[aqi_col].diff()
        print("✅ Added AQI lag and rolling features.")
    else:
        print("⚠️ 'AQI' column not found — skipping lag features.")
    return df

# ------------------------------------------------------------
# 🧩 4️⃣ Add Prediction Targets (for next 3 days)
# ------------------------------------------------------------
def add_prediction_targets(df):
    df = df.copy()
    for c in df.columns:
        if c.lower() == 'aqi':
            aqi_col = c
            break
    else:
        print("⚠️ 'AQI' column not found — skipping targets.")
        return df

    # FS-safe names (lowercase + underscores only)
    df['target_aqi_t1'] = df[aqi_col].shift(-1)
    df['target_aqi_t2'] = df[aqi_col].shift(-2)
    df['target_aqi_t3'] = df[aqi_col].shift(-3)
    print("🎯 Added 3-day ahead target columns for AQI.")
    return df

# ------------------------------------------------------------
# 🧩 5️⃣ Full Feature Engineering Pipeline
# ------------------------------------------------------------
def run_feature_engineering(df_cleaned):
    df = df_cleaned.copy()
    df = process_datetime_features(df)
    df = add_and_encode_season(df)
    df = add_aqi_lag_features(df)
    df = add_prediction_targets(df)

    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"⚠️ Dropping non-numeric columns: {list(non_numeric)}")
        df = df.drop(columns=non_numeric)

    df = df.dropna().reset_index(drop=True)
    print(f"\n✅ Final engineered feature matrix shape: {df.shape}")
    return df

# ------------------------------------------------------------
# 🚀 Run Feature Engineering
# ------------------------------------------------------------
df_feature = run_feature_engineering(df_cleaned)

# ------------------------------------------------------------
# 💾 Upload to Feature Store (with Deduplication)
# ------------------------------------------------------------
print("\n📤 Preparing to upload engineered features to Feature Store...")

aqi_fg = fs.get_or_create_feature_group(
    name="aqi_feature_engineered",
    version=1,
    primary_key=["year", "month", "day", "hour"],
    description="Feature engineered AQI data for 3-day forecasting",
    online_enabled=True
)

# 🧱 Read existing data (if any)
try:
    df_existing = aqi_fg.read()
    print(f"📂 Existing records found: {df_existing.shape[0]}")
except Exception as e:
    print("⚠️ No existing data found — creating new feature group.")
    df_existing = pd.DataFrame(columns=df_feature.columns)

# 🧮 Deduplicate based on primary key
merge_keys = ["year", "month", "day", "hour"]
if not df_existing.empty:
    df_merged = df_feature.merge(df_existing[merge_keys], on=merge_keys, how="left", indicator=True)
    df_new = df_merged[df_merged["_merge"] == "left_only"].drop(columns=["_merge"])
else:
    df_new = df_feature

print(f"🧩 New unseen rows to insert: {df_new.shape[0]}")

# Insert only new rows
if df_new.shape[0] > 0:
    aqi_fg.insert(df_new)
    print("✅ Successfully inserted new rows into Feature Store!")
else:
    print("✅ No new data to insert — feature store is already up to date.")
