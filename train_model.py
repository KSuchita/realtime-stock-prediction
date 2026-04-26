import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Create models directory
os.makedirs("models", exist_ok=True)

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS"]

def train_all_models():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    # Lists to store ALL test results for global evaluation
    global_y_test = []
    global_predictions = []

    for symbol in TICKERS:
        print(f"Training {symbol}...")
        try:
            df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
            
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

            # Split (No shuffle for time-series)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

            # Train
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train.values, y_train.values)

            # Generate predictions for this specific stock
            preds = model.predict(X_test.values)

            # Store these in our global lists
            global_y_test.extend(y_test.tolist())
            global_predictions.extend(preds.tolist())

            # Save the individual model
            joblib.dump(model, f"models/{symbol}_model.pkl")

        except Exception as e:
            print(f"Error training {symbol}: {e}")

    # --- GLOBAL EVALUATION (After the loop) ---
    if global_y_test:
        global_y_test = np.array(global_y_test)
        global_predictions = np.array(global_predictions)

        mae = mean_absolute_error(global_y_test, global_predictions)
        r2 = r2_score(global_y_test, global_predictions)
        
        # Mean Absolute Percentage Error (MAPE) is best for global "Accuracy"
        mape = np.mean(np.abs((global_y_test - global_predictions) / global_y_test)) * 100
        global_accuracy = 100 - mape

        print("\n" + "="*40)
        print("OVERALL MODEL PERFORMANCE (ALL COMPANIES)")
        print("="*40)
        print(f"Global Mean Absolute Error:  ₹{mae:.2f}")
        print(f"Global R-squared Score:      {r2:.4f}")
        print(f"Global System Accuracy:      {global_accuracy:.2f}%")
        print("="*40)

        # Save global metrics to a single file
        with open("models/global_metrics.txt", "w") as f:
            f.write(f"Global MAE: {mae}\nGlobal R2: {r2}\nGlobal Accuracy: {global_accuracy}%")

if __name__ == "__main__":
    train_all_models()