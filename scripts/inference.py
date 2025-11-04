from feature_selection import scale_and_select_features
import joblib

def predict_next_3_days():
    # Step 1: get scaled and selected features
    df_scaled = scale_and_select_features()

    # Step 2: load model
    model = joblib.load("models/best_xgboost.pkl")

    # Step 3: make prediction
    preds = model.predict(df_scaled)

    print("✅ Predictions generated.")
    return preds
