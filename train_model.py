import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# List of companies to train
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS"]

def train_all_models():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    for symbol in TICKERS:
        print(f"\n--- Training model for {symbol} ---")
        try:
            df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.copy().dropna()

            # Feature Engineering
            df['Prev_Close'] = df['Close'].shift(1)
            df['MA_5'] = df['Close'].rolling(5).mean()
            df['MA_10'] = df['Close'].rolling(10).mean()
            df['Daily_Return'] = df['Close'].pct_change()
            df.dropna(inplace=True)

            features = ['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']
            X = df[features]
            y = df['Close']

            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X.values, y.values)

            # Save model with symbol name
            model_path = f"models/{symbol}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Successfully saved model for {symbol}")

        except Exception as e:
            print(f"Error training {symbol}: {e}")

if __name__ == "__main__":
    train_all_models()