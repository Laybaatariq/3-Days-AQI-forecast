# 3-Day Hourly AQI Prediction Dashboard

This project is an interactive **Machine Learning and Streamlit-based Dashboard** designed to forecast the **Air Quality Index (AQI)** for the next **3 days (hourly basis)**.  
It integrates **machine learning models**, **Hopsworks feature storage**, and a **CI/CD pipeline** via **GitHub Actions**, ensuring continuous updates and deployment of the system.

---

## Project Overview

Air quality forecasting plays a vital role in understanding environmental health risks.  
This project predicts **hourly AQI values for the next 72 hours**, using multiple machine learning algorithms and automatically updating via a data pipeline.  

The system architecture combines:
- **ML models (RF, XGB, LGBM)** for prediction  
- **Hopsworks feature store** for feature management  
- **GitHub-based CI/CD** for automation  
- **Streamlit frontend** for visualization

---

##  System Architecture

**Data Source & Feature Store (Hopsworks):**
- Historical AQI and meteorological features (e.g., PM2.5, PM10, temperature, humidity, rainfall, wind speed, etc.) are stored in **Hopsworks**.
- Automated scripts fetch the most recent features from the **Hopsworks Feature Store** before each model retraining.
- Version-controlled datasets ensure consistent and reproducible results.

**Model Training & Forecasting:**
- Three ensemble algorithms are used:
  -  **Random Forest (RF)**
  -  **XGBoost (XGB)**
  -  **LightGBM (LGBM)**
- The models predict **hourly AQI** for 3 consecutive days (target_aqi_t1, t2, t3).

**CI/CD Automation (GitHub Actions):**
- **CI/CD pipeline** triggers automatically when:
- New data is pushed to Hopsworks.
- Code updates occur in the repository.
- The pipeline:
- Fetches data from Hopsworks.
- Retrains models.
- Evaluates results.
- Deploys the updated model and dashboard to Streamlit Cloud (or other hosting services).

**Streamlit Frontend:**
- Interactive dashboard displays:
- 3-day hourly AQI predictions.
- Color-coded categories.
- Trend graphs using Plotly.

---

 **Observation:**
- All models perform well for short-term (t1) forecasts.  
- LightGBM and XGBoost achieve the highest generalization accuracy across all targets.  
- Random Forest provides good baseline stability.  
- Ensemble prediction yields the smoothest and most reliable 3-day AQI trends.

---

## Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Frontend** | Streamlit, Plotly |
| **Backend** | Python, scikit-learn |
| **ML Models** | RandomForest, XGBoost, LightGBM |
| **Feature Store** | Hopsworks |
| **CI/CD** | GitHub Actions |
| **Explainability** | SHAP |
| **Visualization** | Plotly, Matplotlib |
| **Utilities** | Pandas, NumPy, tqdm, python-dotenv |

---

##  AQI Color Scale

| AQI Category | Range | Color |
|---------------|--------|--------|
| Good | 0–50 | 🟩 Green |
| Moderate | 51–100 | 🟨 Yellow |
| Unhealthy for Sensitive Groups | 101–150 | 🟧 Orange |
| Unhealthy | 151–200 | 🟥 Red |
| Very Unhealthy | 201–300 | 🟪 Purple |
| Hazardous | 301+ | 🟫 Maroon |

---

##  Workflow Summary

1. **Data Fetching:** Hopsworks feature store → pipeline fetches updated data  
2. **Model Training:** Train RF, XGB, LGBM on AQI + meteorological features  
3. **Model Evaluation:** Compute R², MAE, RMSE for each target (t1, t2, t3)  
4. **Deployment:** CI/CD pipeline pushes updates to Streamlit dashboard  
5. **Visualization:** Streamlit + Plotly render hourly AQI forecast (3 days)

---

##  Future Enhancements

-  Integrate **real-time sensor or API-based AQI data**
-  Add **auto-retraining scheduling** via GitHub Actions CRON jobs
-  Mobile-optimized Streamlit layout
-  Multi-cloud deployment (Streamlit Cloud, Render, Hugging Face Spaces)

---

> _“Clean air, clear data — predicting tomorrow’s atmosphere today.”_
