import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# --- Page Config ---
st.set_page_config(page_title="Sales Forecaster Pro", layout="wide")
st.title("📈 Sales Forecasting & Time-Series Analysis")
st.markdown("""
This application compares **ARIMA** and **Facebook Prophet** models to predict future sales trends based on historical data.
""")

# --- Sidebar: User Controls ---
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Days to Predict", 7, 90, 30)
noise_level = st.sidebar.slider("Data Volatility (Noise)", 1, 10, 5)

# --- Step 1: Data Generation & EDA ---
@st.cache_data
def load_data(noise):
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    # Trend + Yearly Seasonality + User-defined Noise
    sales = 100 + (np.arange(len(dates)) * 0.05) + \
            (np.sin(np.arange(len(dates)) * (2 * np.pi / 365)) * 20) + \
            np.random.normal(0, noise, len(dates))
    return pd.DataFrame({'ds': dates, 'y': sales})

df = load_data(noise_level)
train = df.iloc[:-30]
test = df.iloc[-30:]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Historical Sales Data")
    st.line_chart(df.set_index('ds'))

# --- Step 2: Modeling ---
st.divider()
st.header("Model Performance & Comparison")

# --- Prophet Model ---
prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False)
prophet_model.fit(train)
future = prophet_model.make_future_dataframe(periods=forecast_days)
forecast = prophet_model.predict(future)

# --- ARIMA Model ---
arima_model = ARIMA(train['y'], order=(5, 1, 0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=forecast_days)

# --- Metrics Calculation ---
p_mape = mean_absolute_percentage_error(test['y'], forecast.iloc[len(train):len(train)+30]['yhat'])
a_mape = mean_absolute_percentage_error(test['y'], arima_forecast[:30])

m1, m2, m3 = st.columns(3)
m1.metric("Prophet MAPE (Error)", f"{p_mape:.2%}")
m2.metric("ARIMA MAPE (Error)", f"{a_mape:.2%}")
m3.metric("Model Winner", "Prophet" if p_mape < a_mape else "ARIMA")

# --- Step 3: Visualization ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['ds'].iloc[-60:], df['y'].iloc[-60:], label="Actual", color="black")
ax.plot(forecast['ds'].iloc[-forecast_days:], forecast['yhat'].iloc[-forecast_days:], label="Prophet Forecast", color="blue", linestyle="--")
ax.fill_between(forecast['ds'].iloc[-forecast_days:], 
                forecast['yhat_lower'].iloc[-forecast_days:], 
                forecast['yhat_upper'].iloc[-forecast_days:], color='blue', alpha=0.2)
ax.set_title("Future Sales Forecast with Confidence Intervals")
ax.legend()
st.pyplot(fig)

# --- Data Table ---
if st.checkbox("Show Raw Forecast Data"):
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days))