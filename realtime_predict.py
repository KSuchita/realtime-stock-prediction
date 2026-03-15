import yfinance as yf
import pandas as pd
import joblib
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# --- CONFIGURATION ---
# Add your desired NSE tickers here
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS"]
MODEL_DIR = "models"
PREDICTION_FILE = "data/predictions.csv"

# Setup Folders
Path("data").mkdir(exist_ok=True)

def run_multi_predictions():
    today_date = datetime.today().date()
    current_weekday = today_date.weekday()
    now = datetime.now()
    
    market_closed_time = 15 
    market_closed_minute = 30
    
    all_results = []

    print(f"Starting Multi-Stock Analysis for {today_date}...")

    for symbol in TICKERS:
        print(f"--- Analyzing {symbol} ---")
        
        # 1. Load Specific Model
        model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
        if not os.path.exists(model_path):
            print(f"Warning: Model for {symbol} not found at {model_path}. Skipping.")
            continue
        
        model = joblib.load(model_path)

        # 2. Download Data
        df = yf.download(symbol, period="30d", interval="1d", progress=False)
        if df.empty:
            print(f"Error: No data for {symbol}. Skipping.")
            continue

        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # 3. Feature Engineering
        df['Prev_Close'] = df['Close'].shift(1)
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_10'] = df['Close'].rolling(10).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df = df.dropna()

        # 4. Weekend & Target Date Logic
        last_data_date = df.index[-1].date()
        
        if current_weekday >= 5: # Weekend
            target_date = today_date + timedelta(days=(7 - current_weekday))
        elif last_data_date != today_date: # Weekday morning
            target_date = today_date
        else:
            # Check market hours
            if now.hour < market_closed_time or (now.hour == market_closed_time and now.minute < market_closed_minute):
                target_date = today_date + (timedelta(days=3) if current_weekday == 4 else timedelta(days=1))
            else:
                target_date = today_date + (timedelta(days=3) if current_weekday == 4 else timedelta(days=1))

        # 5. Predict
        last_row = df.iloc[-1:]
        last_close = float(last_row["Close"].iloc[0])
        X = last_row[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']]
        predicted_price = float(model.predict(X.values)[0])

        # 6. Historical Scorecard Lookup
        yesterday_pred = "N/A"
        if os.path.exists(PREDICTION_FILE):
            old_preds = pd.read_csv(PREDICTION_FILE)
            # Find what we predicted for this specific ticker for today
            match = old_preds[(old_preds["Ticker"] == symbol) & 
                              (old_preds["Target_Date"] == last_data_date.strftime("%Y-%m-%d"))]
            if not match.empty:
                yesterday_pred = float(match.iloc[-1]["Predicted_Next_Close"])

        # 7. Collect Data
        all_results.append({
            "Ticker": symbol,
            "Prediction_Date": today_date.strftime("%Y-%m-%d"),
            "Target_Date": target_date.strftime("%Y-%m-%d"),
            "Last_Close": round(last_close, 2),
            "Predicted_Next_Close": round(predicted_price, 2),
            "Prev_Prediction_For_Today": yesterday_pred
        })

    # 8. Save/Update CSV
    new_results_df = pd.DataFrame(all_results)
    if os.path.exists(PREDICTION_FILE):
        old_df = pd.read_csv(PREDICTION_FILE)
        # Keep old history but remove today's entries if rerunning script
        filtered_df = old_df[~((old_df["Prediction_Date"] == today_date.strftime("%Y-%m-%d")))]
        final_df = pd.concat([filtered_df, new_results_df], ignore_index=True)
    else:
        final_df = new_results_df
    
    final_df.to_csv(PREDICTION_FILE, index=False)
    print(f"\nSuccess! Predictions for {len(all_results)} stocks saved to {PREDICTION_FILE}")

if __name__ == "__main__":
    run_multi_predictions()