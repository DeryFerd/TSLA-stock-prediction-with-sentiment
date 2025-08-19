
import streamlit as st
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from joblib import load
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION & HELPER FUNCTIONS ---

STOCK_TICKER = 'TSLA'
TIME_STEP = 60
DATA_FILE = 'tesla_data_with_sentiment.csv'

@st.cache_resource
def load_artifacts():
    """Loads the pre-trained model and scalers."""
    try:
        model = load_model('stock_prediction_lstm.h5')
        scaler_features = load('scaler_features.pkl')
        scaler_target = load('scaler_target.pkl')
        return model, scaler_features, scaler_target
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return None, None, None

@st.cache_data
def load_historical_data(filepath):
    """Loads historical data from the CSV file."""
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        return df
    except Exception as e:
        st.error(f"Failed to load historical data file '{filepath}': {e}")
        return None

def fetch_data(ticker, start_date, end_date):
    """Fetches live stock data from yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty: return None
        return data
    except Exception as e:
        st.error(f"Failed to fetch stock data: {e}")
        return None

def forecast_future(model, last_60_days_features_scaled, num_days_to_forecast, scaler_target):
    """Performs an iterative multi-day forecast."""
    future_predictions_scaled = []
    current_window = last_60_days_features_scaled.copy()

    for _ in range(num_days_to_forecast):
        X_predict = np.reshape(current_window, (1, TIME_STEP, current_window.shape[1]))
        prediction_scaled = model.predict(X_predict, verbose=0)
        future_predictions_scaled.append(prediction_scaled[0][0])
        new_day_features_scaled = np.append(current_window[-1, 1:], prediction_scaled[0][0])
        current_window = np.append(current_window[1:], [new_day_features_scaled], axis=0)

    future_predictions = scaler_target.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    return future_predictions


# --- STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="LSTM Stock Prediction", layout="wide")
st.title(f'🚀 Stock Price Prediction & Forecast for {STOCK_TICKER}')
st.write("This app uses an LSTM model to predict and forecast the closing stock price. The model was trained on historical price data enriched with daily news sentiment scores.")

model, scaler_features, scaler_target = load_artifacts()
historical_data = load_historical_data(DATA_FILE)

if historical_data is not None:
    # --- ADDED DESCRIPTIONS ---
    start_year = historical_data.index.min().year
    end_year = historical_data.index.max().year
    st.subheader(f"Historical Closing Price ({start_year} - {end_year})")
    st.line_chart(historical_data['close'])
    
    st.subheader("Latest Data Snapshot")
    st.caption("This table shows the last 10 days of data used for training the model, including the calculated daily news sentiment score (1: Positive, -1: Negative, 0: Neutral).")
    st.dataframe(historical_data[['close', 'volume', 'sentiment_score']].tail(10))
else:
    st.info("Waiting for the historical data file...")

st.divider()

if model and scaler_features and scaler_target:
    st.header("🔮 Make a New Forecast")
    st.caption("Select the number of days you want to forecast into the future. Note that predictions for longer periods are simulations and may become less accurate due to compounding errors.")
    
    days_to_forecast = st.number_input("Days to forecast:", min_value=1, max_value=100, value=7)

    if st.button(f'Forecast Next {days_to_forecast} Days', type="primary"):
        with st.spinner('Fetching latest data and running forecast...'):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=150)
            live_data = fetch_data(STOCK_TICKER, start_date, end_date)
            
            if live_data is not None and len(live_data) >= TIME_STEP:
                live_data['sentiment_score'] = 0
                features_to_scale = live_data[['Open', 'High', 'Low', 'Volume', 'sentiment_score']]
                scaled_features = scaler_features.transform(features_to_scale)
                last_60_days_features_only_scaled = scaled_features[-TIME_STEP:]

                forecasted_prices = forecast_future(model, last_60_days_features_only_scaled, days_to_forecast, scaler_target)
                
                st.success("Forecast complete!")
                
                last_date = live_data.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_forecast)
                forecast_df = pd.DataFrame(forecasted_prices, index=future_dates, columns=['Forecasted Price'])

                st.subheader("Forecast Results Table")
                st.dataframe(forecast_df)

                st.subheader("Forecast Results Chart")
                st.line_chart(forecast_df['Forecasted Price'])
            else:
                st.warning(f"Not enough historical data to forecast. A minimum of {TIME_STEP} days is required.")
else:
    st.error("Model artifacts (.h5, .pkl files) could not be loaded.")
