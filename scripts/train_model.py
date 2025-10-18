# ==========================================================
# train_model.py
# Purpose: Train AQI prediction model from cleaned data in Hopsworks
# Author: Laiba Tariq
# ==========================================================

import hopsworks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

# ==========================================================
#  Connect to Hopsworks and Load Cleaned Data
# ==========================================================
project = hopsworks.login()
fs = project.get_feature_store()
print(" Connected to Hopsworks project")

FEATURE_GROUP = "cleaned_aqi_data"
VERSION = 1

fg = fs.get_feature_group(FEATURE_GROUP, version=VERSION)
df = fg.read()
print(f" Data fetched from '{FEATURE_GROUP}' (rows={df.shape[0]}, cols={df.shape[1]})")

# ==========================================================
#  Feature Engineering
# ==========================================================
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

# Time-based features
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour
df['weekday'] = df['time'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

# Drop NA values
df = df.dropna()

# ==========================================================
#  Target and Feature Split
# ==========================================================
target = 'aqi'
X = df.drop(columns=['aqi'])
y = df['aqi']

# Convert datetime → numeric timestamp
X['time'] = X['time'].astype('int64') // 10**9

# Chronological split
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ==========================================================
#  Scaling
# ==========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
#  Model Training (RF + XGBoost)
# ==========================================================
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    results[name] = {"r2_train": r2_train, "r2_test": r2_test, "mae": mae, "rmse": rmse}

# ==========================================================
#  Select Best Model
# ==========================================================
best_model_name = max(results, key=lambda k: results[k]['r2_test'])
best_model = models[best_model_name]
print(f"\n Best Model: {best_model_name}")
print(f"R² (Test): {results[best_model_name]['r2_test']:.4f}")
print(f"MAE: {results[best_model_name]['mae']:.4f}")
print(f"RMSE: {results[best_model_name]['rmse']:.4f}")

# ==========================================================
# Save Best Model
# ==========================================================
os.makedirs("models", exist_ok=True)
model_path = f"models/{best_model_name}_model.pkl"
scaler_path = "models/scaler.pkl"

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
print(f" Saved best model → {model_path}")
print(f"Saved scaler → {scaler_path}")

# ==========================================================
# Print Results
# ==========================================================
print("\n Model Comparison Results:")
for name, r in results.items():
    print(f"{name:15s} | R² Test: {r['r2_test']:.4f} | MAE: {r['mae']:.4f} | RMSE: {r['rmse']:.4f}")
