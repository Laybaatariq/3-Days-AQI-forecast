# -*- coding: utf-8 -*-
import os
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import hopsworks

# ----------------------------
# 1️⃣ Configuration
# ----------------------------
latitude = 24.8608   # Karachi
longitude = 67.0104

# Fetch last 48 hours
end_date = datetime.now(timezone.utc).date()
start_date = end_date - timedelta(days=2)

# AQI API (Open-Meteo Air Quality)
aqi_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
)

# Weather API (Open-Meteo Archive)
weather_url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=temperature_2m,relative_humidity_2m,pressure_msl,"
    "wind_speed_10m,cloud_cover"
)

# ----------------------------
# 2️⃣ Fetch Data
# ----------------------------
print(f"📡 Fetching AQI and weather data from {start_date} to {end_date}...")
aqi_data = requests.get(aqi_url).json()
weather_data = requests.get(weather_url).json()

aqi_df = pd.DataFrame(aqi_data.get("hourly", {}))
weather_df = pd.DataFrame(weather_data.get("hourly", {}))

if "time" not in aqi_df.columns or "time" not in weather_df.columns:
    raise ValueError("❌ Missing 'time' column from one of the APIs.")

# Convert timestamps
aqi_df["time"] = pd.to_datetime(aqi_df["time"])
weather_df["time"] = pd.to_datetime(weather_df["time"])

# ----------------------------
# 3️⃣ Merge AQI + Weather Data
# ----------------------------
merged_df = pd.merge(aqi_df, weather_df, on="time", how="inner")

# ----------------------------
# 4️⃣ AQI Breakpoints (US EPA Standard)
# ----------------------------
breakpoints = {
    "pm2_5": [(0.0,12.0,0,50), (12.1,35.4,51,100), (35.5,55.4,101,150),
              (55.5,150.4,151,200), (150.5,250.4,201,300), (250.5,350.4,301,400), (350.5,500.4,401,500)],
    "pm10":  [(0,54,0,50), (55,154,51,100), (155,254,101,150),
              (255,354,151,200), (355,424,201,300), (425,504,301,400), (505,604,401,500)],
    "ozone": [(0,0.054,0,50), (0.055,0.070,51,100), (0.071,0.085,101,150),
              (0.086,0.105,151,200), (0.106,0.200,201,300)],
    "carbon_monoxide": [(0.0,4.4,0,50), (4.5,9.4,51,100), (9.5,12.4,101,150),
                        (12.5,15.4,151,200), (15.5,30.4,201,300), (30.5,40.4,301,400), (40.5,50.4,401,500)],
    "nitrogen_dioxide": [(0,53,0,50), (54,100,51,100), (101,360,101,150),
                         (361,649,151,200), (650,1249,201,300), (1250,1649,301,400), (1650,2049,401,500)],
    "sulphur_dioxide": [(0,35,0,50), (36,75,51,100), (76,185,101,150),
                        (186,304,151,200), (305,604,201,300), (605,804,301,400), (805,1004,401,500)],
}

def compute_aqi_for_pollutant(pollutant, conc):
    if pollutant not in breakpoints or pd.isna(conc):
        return None
    for c_lo, c_hi, i_lo, i_hi in breakpoints[pollutant]:
        if c_lo <= conc <= c_hi:
            if (c_hi - c_lo) == 0:
                return i_lo if conc == c_lo else None
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo
    return None

def calculate_overall_aqi(row):
    pollutants = ["pm2_5", "pm10", "ozone", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide"]
    aqi_values = []
    for p in pollutants:
        val = compute_aqi_for_pollutant(p, row.get(p))
        if val is not None:
            aqi_values.append(val)
    return max(aqi_values) if aqi_values else None

merged_df["AQI"] = merged_df.apply(calculate_overall_aqi, axis=1)
merged_df.columns = merged_df.columns.str.lower()

# ----------------------------
# 5️⃣ Reorder Columns
# ----------------------------
column_order = [
    "time", "aqi", "pm2_5", "pm10", "ozone",
    "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "temperature_2m", "relative_humidity_2m",
    "pressure_msl", "wind_speed_10m", "cloud_cover"
]
merged_df = merged_df[[col for col in column_order if col in merged_df.columns]]

print("✅ Preview of merged data:")
print(merged_df.head())

# ----------------------------
# 6️⃣ Upload to Hopsworks
# ----------------------------
print("\n🚀 Connecting to Hopsworks...")
api_key = os.getenv("HOPSWORKS_API_KEY")  # Use secret from GitHub Actions
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

aqi_fg = fs.get_or_create_feature_group(
    name="karachi_aqi_weather",
    version=1,
    primary_key=["time"],
    description="Hourly AQI (EPA-standard) + weather data for Karachi (Open-Meteo)",
    online_enabled=True
)

try:
    aqi_fg.insert(merged_df)
    print("🎉 Data successfully uploaded to Hopsworks Feature Store!")
except Exception as e:
    print(f"❌ Failed to insert data into Hopsworks: {e}")
