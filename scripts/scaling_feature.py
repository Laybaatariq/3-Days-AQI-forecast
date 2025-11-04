import hopsworks
import pandas as pd
import joblib

def scale_features():
    #  Step 1: Connect to Feature Store
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Step 2: Fetch latest feature-engineered data
    fg = fs.get_feature_group("aqi_features_engineered", version=1)
    df = fg.read()

    print(f" Loaded {df.shape[0]} rows and {df.shape[1]} columns from engineered features.")

    #  Step 3: Load saved scaler from training
    scaler = joblib.load("models/scaler.pkl")

    #  Step 4: Apply scaling
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )

    print(" Scaling completed successfully.")
    return df_scaled


if __name__ == "__main__":
    df_scaled = scale_features()
    print(df_scaled.head())
