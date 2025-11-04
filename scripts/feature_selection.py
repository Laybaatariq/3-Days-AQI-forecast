import hopsworks
import pandas as pd
import joblib
import json
from pathlib import Path

def load_metadata():
    path = Path("models/feature_metadata.json")
    if not path.exists():
        raise FileNotFoundError("feature_metadata.json not found in models/")
    with open(path) as f:
        meta = json.load(f)
    return meta

def get_engineered_features():
    """Fetch the latest engineered data from Hopsworks Feature Store"""
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features_engineered", version=1)
    df = fg.read()
    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns from engineered features.")
    return df

def scale_and_select_features():
    """Scale and select only the features used by the trained model"""
    df = get_engineered_features()

    # Load scaler
    scaler = joblib.load("models/scaler.pkl")

    # Load feature metadata (selected features list)
    meta = load_metadata()
    selected_features = meta["selected_features"]

    # Ensure those features exist in df
    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in engineered data: {missing}")

    # Subset and scale
    df_selected = df[selected_features]
    df_scaled = pd.DataFrame(
        scaler.transform(df_selected),
        columns=selected_features
    )

    print("✅ Scaling + feature selection completed successfully.")
    return df_scaled

if __name__ == "__main__":
    df_scaled = scale_and_select_features()
    print(df_scaled.head())
