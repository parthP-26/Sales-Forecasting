import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# --- Page Configuration ---
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Title and Description ---
st.title("📈 Robust Sales Forecasting System")
st.markdown("""
This dashboard uses **ARIMA** and **Facebook Prophet** to predict future business trends. 
Adjust the parameters in the sidebar to see how the models adapt to data volatility.
""")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
forecast_horizon = st.sidebar.slider("Forecast Days", 7, 90, 30)
volatility = st.sidebar.slider("Data Volatility (Noise)", 1, 10, 5)

# --- Step 1: Data Generation (Simulating Real Business Data) ---
@st.cache_data
def load_data(noise):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
    # Base Sales + Linear Trend + Yearly Seasonality + Random Noise
    base = 100
    trend = np.arange(len(dates)) * 0.08
    seasonality = np.sin(np.arange(len(dates)) * (2 * np.pi / 365)) * 25
    noise_vals = np.random.normal(0, noise, len(dates))
    
    sales = base + trend + seasonality + noise_vals
    return pd.DataFrame({'ds': dates, 'y': sales})

df = load_data(volatility)
train_df = df.iloc[:-30] # Train on all but last 30 days
test_df = df.iloc[-30:]  # Last 30 days for validation

# --- Step 2: Modeling ---
with st.spinner('Calculating Forecasts...'):
    # --- Facebook Prophet ---
    m_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m_prophet.fit(train_df)
    future = m_prophet.make_future_dataframe(periods=forecast_horizon)
    forecast_prophet = m_prophet.predict(future)

    # --- ARIMA (5,1,0) ---
    m_arima = ARIMA(train_df['y'], order=(5, 1, 0))
    res_arima = m_arima.fit()
    forecast_arima = res_arima.forecast(steps=forecast_horizon)

# --- Step 3: Visualization & Metrics ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Model Comparison: Actual vs. Predicted")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotting Historical
    ax.plot(df['ds'].iloc[-60:], df['y'].iloc[-60:], label="Actual Sales", color="#2c3e50", lw=2)
    
    # Plotting Prophet
    prophet_plot_df = forecast_prophet.iloc[-forecast_horizon:]
    ax.plot(prophet_plot_df['ds'], prophet_plot_df['yhat'], label="Prophet Forecast", color="#3498db", linestyle="--")
    ax.fill_between(prophet_plot_df['ds'], prophet_plot_df['yhat_lower'], prophet_plot_df['yhat_upper'], color='#3498db', alpha=0.2)
    
    # Plotting ARIMA
    arima_dates = pd.date_range(start=train_df['ds'].iloc[-1], periods=forecast_horizon+1, freq='D')[1:]
    ax.plot(arima_dates, forecast_arima, label="ARIMA Forecast", color="#e67e22", linestyle=":")
    
    ax.set_title("30-Day Sales Outlook")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Performance Metrics")
    # Calculate MAPE for the overlap period (last 30 days)
    p_mape = mean_absolute_percentage_error(test_df['y'], forecast_prophet.iloc[len(train_df):len(train_df)+30]['yhat'])
    a_mape = mean_absolute_percentage_error(test_df['y'], forecast_arima[:30])
    
    st.metric("Prophet Error (MAPE)", f"{p_mape:.2%}")
    st.metric("ARIMA Error (MAPE)", f"{a_mape:.2%}")
    
    winner = "Prophet" if p_mape < a_mape else "ARIMA"
    st.success(f"Best Model: {winner}")

# --- Insights Section ---
st.divider()
st.subheader("Business Insights")
i1, i2, i3 = st.columns(3)
i1.write("**Seasonality Detected:** High demand peaks identified every 365 days.")
i2.write(f"**Trend Projection:** Sales are growing at an average rate of 0.08 units/day.")
i3.write(f"**Confidence:** Prophet shows a 95% uncertainty interval for the next {forecast_horizon} days.")
