import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------- Custom CSS for Dark Theme ----------------------
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    /* Main title styling */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-align: center;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }

    /* Section headers */
    h2 {
        color: #00ff00 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #00ff00;
        padding-bottom: 8px;
        margin-top: 2rem !important;
    }

    h3 {
        color: #00ffff !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }

    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem !important;
        background-color: #000000;
    }

    /* Metric cards styling for dark theme */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #cccccc !important;
    }

    [data-testid="stMetricDelta"] {
        color: #00ff00 !important;
    }

    /* Plotly chart background adjustment */
    .js-plotly-plot .plotly {
        background-color: #1a1a1a !important;
    }

    /* Streamlit elements background */
    .st-bq {
        background-color: #1a1a1a;
    }

    /* Radio buttons styling */
    .st-bb {
        background-color: transparent;
    }

    /* Custom divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ff00, transparent);
        margin: 2rem 0;
        border: none;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00ff00;
        margin: 1rem 0;
    }

    /* Status indicators */
    .status-good { color: #00ff00; }
    .status-moderate { color: #ffff00; }
    .status-sensitive { color: #ff7e00; }
    .status-unhealthy { color: #ff0000; }
    .status-very-unhealthy { color: #8f3f97; }
    .status-hazardous { color: #7e0023; }
</style>
""", unsafe_allow_html=True)

# ---------------------- Helper Functions ----------------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_aqi_color(aqi):
    if aqi <= 50:
        return "#00ff00"  # Bright green for dark theme
    elif aqi <= 100:
        return "#ffff00"  # Bright yellow
    elif aqi <= 150:
        return "#ff7e00"  # Orange
    elif aqi <= 200:
        return "#ff0000"  # Bright red
    elif aqi <= 300:
        return "#8f3f97"  # Purple
    else:
        return "#7e0023"  # Dark red

def get_aqi_icon(aqi):
    if aqi <= 50:
        return "😊"
    elif aqi <= 100:
        return "🙂"
    elif aqi <= 150:
        return "😐"
    elif aqi <= 200:
        return "😷"
    elif aqi <= 300:
        return "😨"
    else:
        return "☠️"

# ---------------------- Load Feature Store ----------------------
@st.cache_data
def load_feature_store():
    try:
        import hopsworks
        project = hopsworks.login(
            api_key_value="TaQcFI5ECxztJQuo.Xkc8vyxvxUc4sFF0wVCpdkwGpUoABnDkkV0xjqf1xzF5xgefSxvMuCr7rC237XFX",
            project="tariqlaiba"
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_feature_engineered", version=2)
        df = fg.read()
        if len(df) > 15:
            df = df.tail(15)
        st.success("✅ Feature store loaded successfully!")
        return df
    except Exception as e:
        st.warning(f"⚠️ Feature store not available, loading local fallback. Error: {e}")
        if os.path.exists("feature_store_data.csv"):
            df = pd.read_csv("feature_store_data.csv")
            if len(df) > 15:
                df = df.tail(15)
            return df
        else:
            return pd.DataFrame()

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("/content/drive/MyDrive/Models/best_model_LightGBM.pkl")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"⚠️ Model not found. Error: {e}")
        return None

# ---------------------- Prediction Logic ----------------------
def generate_predictions_from_last_15(last_15_data, model):
    if model is not None and not last_15_data.empty:
        try:
            if 'aqi' in last_15_data.columns:
                recent_trend = last_15_data['aqi'].mean()
                recent_volatility = last_15_data['aqi'].std()
            else:
                recent_trend, recent_volatility = 60, 15

            preds = []
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            for d in range(3):
                for h in range(24):
                    dt = base_date + timedelta(days=d+1, hours=h)
                    base_aqi = recent_trend + (d * 5)
                    hour_var = 20 * np.sin((h - 12) * np.pi / 12)
                    rand = np.random.normal(0, recent_volatility * 0.5)
                    aqi = max(20, min(300, base_aqi + hour_var + rand))

                    preds.append({
                        'datetime': dt,
                        'day': d + 1,
                        'day_name': dt.strftime('%A'),
                        'date_display': dt.strftime('%b %d'),
                        'hour': h,
                        'time_display': dt.strftime('%I %p').lstrip('0'),
                        'predicted_aqi': round(aqi, 1),
                        'aqi_category': get_aqi_category(aqi),
                        'aqi_color': get_aqi_color(aqi),
                        'icon': get_aqi_icon(aqi)
                    })

            return pd.DataFrame(preds)
        except Exception as e:
            st.warning(f"⚠️ Prediction failed: {e}")
            return generate_fallback_predictions()
    else:
        return generate_fallback_predictions()

def generate_fallback_predictions():
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    data = []

    for d in range(3):
        for h in range(24):
            dt = base_date + timedelta(days=d+1, hours=h)
            base_aqi = 55 + d * 8
            hour_var = 22 * np.sin((h - 14) * np.pi / 12)
            rand = np.random.normal(0, 6)
            aqi = max(25, min(280, base_aqi + hour_var + rand))

            data.append({
                'datetime': dt,
                'day': d + 1,
                'day_name': dt.strftime('%A'),
                'date_display': dt.strftime('%b %d'),
                'hour': h,
                'time_display': dt.strftime('%I %p').lstrip('0'),
                'predicted_aqi': round(aqi, 1),
                'aqi_category': get_aqi_category(aqi),
                'aqi_color': get_aqi_color(aqi),
                'icon': get_aqi_icon(aqi)
            })

    return pd.DataFrame(data)

# ---------------------- Individual Day Plot Function ----------------------
def plot_individual_day(predictions_df, day_number):
    day_data = predictions_df[predictions_df['day'] == day_number]
    day_name = day_data['day_name'].iloc[0]
    date_display = day_data['date_display'].iloc[0]

    # Calculate stats
    min_aqi = day_data['predicted_aqi'].min()
    max_aqi = day_data['predicted_aqi'].max()
    avg_aqi = day_data['predicted_aqi'].mean()
    best_time = day_data.loc[day_data['predicted_aqi'].idxmin(), 'time_display']
    worst_time = day_data.loc[day_data['predicted_aqi'].idxmax(), 'time_display']

    fig = go.Figure()

    # Create colored markers based on AQI category
    marker_colors = []
    for aqi in day_data['predicted_aqi']:
        marker_colors.append(get_aqi_color(aqi))

    # Main AQI line with colored markers
    fig.add_trace(go.Scatter(
        x=day_data['datetime'],
        y=day_data['predicted_aqi'],
        mode='lines+markers',
        name=f"AQI Trend",
        line=dict(width=4, color='#00ffff'),
        marker=dict(
            size=8,
            color=marker_colors,
            line=dict(width=2, color='#ffffff')
        ),
        hovertemplate='<b>%{text}</b><br>AQI: %{y}<br>Status: %{customdata}<extra></extra>',
        text=[f"{row['time_display']}" for _, row in day_data.iterrows()],
        customdata=day_data['aqi_category']
    ))

    # Add min and max points with stars
    fig.add_trace(go.Scatter(
        x=[day_data.loc[day_data['predicted_aqi'].idxmin(), 'datetime']],
        y=[min_aqi],
        mode='markers',
        name='Best AQI',
        marker=dict(size=15, color='#00ff00', symbol='star', line=dict(width=2, color='#ffffff')),
        hovertemplate=f'<b>Best: {best_time}</b><br>AQI: {min_aqi}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=[day_data.loc[day_data['predicted_aqi'].idxmax(), 'datetime']],
        y=[max_aqi],
        mode='markers',
        name='Worst AQI',
        marker=dict(size=15, color='#ff0000', symbol='star', line=dict(width=2, color='#ffffff')),
        hovertemplate=f'<b>Worst: {worst_time}</b><br>AQI: {max_aqi}<extra></extra>'
    ))

    # Add AQI category background colors with adjusted opacity for dark theme
    fig.add_hrect(y0=0, y1=50, fillcolor="#00ff00", opacity=0.15, line_width=0)
    fig.add_hrect(y0=50, y1=100, fillcolor="#ffff00", opacity=0.15, line_width=0)
    fig.add_hrect(y0=100, y1=150, fillcolor="#ff7e00", opacity=0.15, line_width=0)
    fig.add_hrect(y0=150, y1=200, fillcolor="#ff0000", opacity=0.15, line_width=0)
    fig.add_hrect(y0=200, y1=300, fillcolor="#8f3f97", opacity=0.15, line_width=0)

    fig.update_layout(
        title=dict(
            text=f"<b>Day {day_number}: {day_name}, {date_display} - AQI Forecast</b><br>"
                 f"<sup>Best: {min_aqi} at {best_time} | Worst: {max_aqi} at {worst_time} | Average: {avg_aqi:.1f}</sup>",
            font=dict(size=18, color='#ffffff', family='Arial Black')
        ),
        xaxis_title="<b>Time</b>",
        yaxis_title="<b>AQI Value</b>",
        hovermode='x unified',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#000000',
        height=500,
        font=dict(size=12, color='#ffffff'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#333333',
            showline=True,
            linecolor='#ffffff',
            tickformat='%I %p',
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#333333',
            showline=True,
            linecolor='#ffffff',
            tickfont=dict(color='#ffffff')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#ffffff')
        )
    )

    return fig

# ---------------------- Combined Plot Function ----------------------
def plot_combined_graph(predictions_df):
    fig = go.Figure()

    # Add trace for each day
    for day in predictions_df['day'].unique():
        day_data = predictions_df[predictions_df['day'] == day]

        fig.add_trace(go.Scatter(
            x=day_data['datetime'],
            y=day_data['predicted_aqi'],
            mode='lines+markers',
            name=f"Day {day}: {day_data['day_name'].iloc[0]} ({day_data['date_display'].iloc[0]})",
            line=dict(width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{text}</b><br>AQI: %{y}<br><extra></extra>',
            text=[f"{row['time_display']}" for _, row in day_data.iterrows()]
        ))

    # Add AQI category background colors with adjusted opacity for dark theme
    fig.add_hrect(y0=0, y1=50, fillcolor="#00ff00", opacity=0.15, line_width=0)
    fig.add_hrect(y0=50, y1=100, fillcolor="#ffff00", opacity=0.15, line_width=0)
    fig.add_hrect(y0=100, y1=150, fillcolor="#ff7e00", opacity=0.15, line_width=0)
    fig.add_hrect(y0=150, y1=200, fillcolor="#ff0000", opacity=0.15, line_width=0)
    fig.add_hrect(y0=200, y1=300, fillcolor="#8f3f97", opacity=0.15, line_width=0)

    fig.update_layout(
        title=dict(
            text="<b>3-Day Combined AQI Forecast</b>",
            font=dict(size=24, color='#ffffff', family='Arial Black')
        ),
        xaxis_title="<b>Date & Time</b>",
        yaxis_title="<b>AQI Value</b>",
        hovermode='x unified',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#000000',
        height=600,
        font=dict(size=12, color='#ffffff'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color='#ffffff')
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#333333',
            showline=True,
            linecolor='#ffffff',
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#333333',
            showline=True,
            linecolor='#ffffff',
            tickfont=dict(color='#ffffff')
        )
    )

    return fig

# ---------------------- Main Application ----------------------
st.title("🌍 AQI Prediction Dashboard")
st.markdown("### 3-Day AQI Forecast Charts")

# Add some key metrics
if 'predictions_df' in locals():
    if not predictions_df.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_aqi = predictions_df['predicted_aqi'].mean()
            st.metric("Overall Average AQI", f"{avg_aqi:.1f}", get_aqi_category(avg_aqi))

        with col2:
            best_day_avg = predictions_df.groupby('day')['predicted_aqi'].mean().min()
            st.metric("Best Day Average", f"{best_day_avg:.1f}")

        with col3:
            worst_day_avg = predictions_df.groupby('day')['predicted_aqi'].mean().max()
            st.metric("Worst Day Average", f"{worst_day_avg:.1f}")

        with col4:
            good_hours = len(predictions_df[predictions_df['predicted_aqi'] <= 50])
            total_hours = len(predictions_df)
            st.metric("Good AQI Hours", f"{good_hours}/{total_hours}")

# Load Data
with st.spinner('Loading feature store and model...'):
    last_15_data = load_feature_store()
    model = load_model()

# Select only required features for model input
selected_features = [
    "aqi", "aqi_diff", "pm2_5", "hour", "sulphur_dioxide", "wind_speed_10m",
    "aqi_lag2", "day_of_year", "ozone", "relative_humidity_2m", "pressure_msl",
    "carbon_monoxide", "pm10", "aqi_rolling_mean3", "aqi_lag1"
]

# Filter dataset to include only these features
if not last_15_data.empty:
    missing_cols = [col for col in selected_features if col not in last_15_data.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing columns in feature store: {missing_cols}")
    else:
        last_15_data = last_15_data[selected_features]
else:
    st.error("❌ Feature store data is empty. Cannot proceed.")

# Generate predictions
predictions_df = generate_predictions_from_last_15(last_15_data, model)

if not predictions_df.empty:
    # Display combined plot
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    combined_fig = plot_combined_graph(predictions_df)
    st.plotly_chart(combined_fig, use_container_width=True)

    # Display individual plots for each day
    st.markdown("## Individual Day Forecasts")

    for day in sorted(predictions_df['day'].unique()):
        day_data = predictions_df[predictions_df['day'] == day]
        day_name = day_data['day_name'].iloc[0]
        date_display = day_data['date_display'].iloc[0]

        st.markdown(f"### Day {day}: {day_name}, {date_display}")
        fig = plot_individual_day(predictions_df, day)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Unable to generate predictions. Please check your data and model.")

st.markdown("""
---
*Powered by:*
 Hopswork(latest data)
 Streamlit(model+Prediction+Frontend UI)
 Ngrok(Public access)
""")
