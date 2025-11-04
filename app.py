# ============================================
# aqi_dashboard_colab.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
import joblib
import importlib.util
import hopsworks
import os

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ============================================
# LOAD FEATURE SELECTION SCRIPT
# ============================================
feature_selection_path = "feature_selection.py"  # path to your feature selection script
spec = importlib.util.spec_from_file_location("feature_selection", feature_selection_path)
feature_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_module)

# ============================================
# LOAD TRAINED MODEL
# ============================================
model_path = "best_lightgbm.pkl"
model = joblib.load(model_path)

# ============================================
# LOAD FEATURE STORE DATA (Hopsworks)
# ============================================
@st.cache_data
def load_feature_store():
    """Fetch latest engineered features from Hopsworks Feature Store."""
    try:
        st.info("🔗 Connecting to Hopsworks...")
        project = hopsworks.login()  # Uses environment or saved credentials
        fs = project.get_feature_store()

        st.info("📦 Loading feature group: 'aqi_feature_engineered' (v1)")
        aqi_fg = fs.get_feature_group(name="aqi_feature_engineered", version=1)

        # Read latest data
        df_features = aqi_fg.read()

        if df_features is None or df_features.empty:
            st.warning("⚠️ Feature Store returned an empty dataset.")
        else:
            st.success(f"✅ Loaded {len(df_features)} records from Feature Store!")

        df_features.columns = df_features.columns.str.lower()
        return df_features

    except Exception as e:
        st.error(f"❌ Failed to connect or fetch data from Hopsworks: {e}")

        # 🔁 Optional fallback: load local CSV if available
        local_path = "feature_store_data.csv"
        if os.path.exists(local_path):
            st.warning(f"📂 Using local fallback data: {local_path}")
            return pd.read_csv(local_path)
        else:
            st.error("🚫 No local fallback data found. Please check Hopsworks connection.")
            return pd.DataFrame()

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00E400"
    elif aqi <= 100:
        return "Moderate", "#FFFF00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"
    elif aqi <= 200:
        return "Unhealthy", "#FF0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97"
    else:
        return "Hazardous", "#7E0023"

def get_health_message(aqi):
    if aqi <= 50:
        return "🟢 Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "🟡 Air quality is acceptable. Some sensitive people may be affected."
    elif aqi <= 150:
        return "🟠 Sensitive groups may experience health effects."
    elif aqi <= 200:
        return "🔴 General public may experience health effects; sensitive groups more."
    elif aqi <= 300:
        return "🟣 Health alert: risk increased for everyone."
    else:
        return "🟤 Emergency: everyone likely affected."

def get_recommendations(aqi):
    if aqi <= 50:
        return ["✅ Great day for outdoor activities", "✅ Everyone can enjoy normal activities"]
    elif aqi <= 100:
        return ["⚠️ Sensitive people reduce prolonged outdoor exertion", "✅ Others normal activities"]
    elif aqi <= 150:
        return ["⚠️ Sensitive groups limit outdoor activity", "⚠️ Children/adults with respiratory issues limit outdoor activities"]
    elif aqi <= 200:
        return ["🚫 Everyone reduce outdoor exertion", "🚫 Sensitive groups avoid outdoor activities"]
    elif aqi <= 300:
        return ["🚫 Everyone avoid prolonged outdoor exertion", "🚫 Stay indoors as much as possible"]
    else:
        return ["⛔ Everyone avoid all outdoor exertion", "⛔ Remain indoors"]

def generate_hourly_data(daily_aqi):
    hours = list(range(24))
    hourly_variation = [daily_aqi + np.random.normal(0, 5) + 10 * np.sin((h - 8) * np.pi / 12) for h in hours]
    return hours, [max(0, val) for val in hourly_variation]

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_next_3_days():
    # Load features
    raw_features = load_feature_store()

    if raw_features is None or raw_features.empty:
        st.warning("⚠️ No features available for prediction.")
        return [0, 0, 0]

    # Apply feature selection
    selected_features = feature_module.select_features(raw_features)

    # Convert to numpy
    X = selected_features.values

    # Predict
    preds = model.predict(X)

    # Return 3 predictions
    if isinstance(preds, np.ndarray):
        if preds.ndim > 1:
            return preds[0][:3]
        else:
            return preds[:3]
    else:
        return list(preds)[:3]

# ============================================
# STREAMLIT DASHBOARD
# ============================================
def main():
    st.markdown('<h1 style="text-align:center; font-size:3rem;">🌍 AQI Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Sequential 3-Day AQI Forecast with Hourly Breakdown")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        st.info("📍 Location: Karachi, Pakistan")
        st.info(f"📅 Today's Date: {date.today().strftime('%B %d, %Y')}")
        view_mode = st.radio("Display Mode:", ["Sequential (One by One)", "All at Once"], index=0)
        show_hourly = st.checkbox("Show Hourly Breakdown", value=True)

    # Prediction button
    if st.button("🚀 Run AQI Prediction"):
        try:
            predictions = predict_next_3_days()
            next_days = [date.today() + timedelta(days=i + 1) for i in range(3)]
            result_df = pd.DataFrame({
                "Date": next_days,
                "Day": [d.strftime('%A') for d in next_days],
                "Predicted_AQI": [round(p, 1) for p in predictions]
            })

            st.success("✅ Prediction Complete!")

            # Sequential view
            if view_mode == "Sequential (One by One)":
                for idx, row in result_df.iterrows():
                    aqi_val = row['Predicted_AQI']
                    category, color = get_aqi_category(aqi_val)
                    st.markdown(f"### Day {idx + 1}: {row['Day']} — {row['Date'].strftime('%b %d')}")
                    st.metric("Predicted AQI", aqi_val)
                    st.info(get_health_message(aqi_val))
                    with st.expander("💡 Recommendations"):
                        for rec in get_recommendations(aqi_val):
                            st.write(f"- {rec}")

                    if show_hourly:
                        hours, hourly_aqi = generate_hourly_data(aqi_val)
                        hourly_df = pd.DataFrame({"Hour": [f"{h:02d}:00" for h in hours], "AQI": hourly_aqi})
                        fig = go.Figure(go.Scatter(x=hourly_df['Hour'], y=hourly_df['AQI'], mode='lines+markers', line=dict(color=color)))
                        st.plotly_chart(fig, use_container_width=True)

            # All-at-once view
            else:
                st.subheader("📊 3-Day Comparison")
                st.dataframe(result_df)

            # Download
            st.download_button("📥 Download CSV", data=result_df.to_csv(index=False),
                               file_name=f"aqi_predictions_{date.today()}.csv")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================
# MAIN ENTRY
# ============================================
if __name__ == "__main__":
    main()
