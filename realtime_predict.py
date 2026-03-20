import yfinance as yf
import pandas as pd
import joblib
import os
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from pathlib import Path

# --- CONFIGURATION ---
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS"]
MODEL_DIR = "models"
PREDICTION_FILE = "data/predictions.csv"

Path("data").mkdir(exist_ok=True)

def get_next_trading_day(reference_date):
    nse = mcal.get_calendar('NSE')
    # Look ahead 10 days to find the next valid session
    schedule = nse.schedule(start_date=reference_date, end_date=reference_date + timedelta(days=10))
    if schedule.empty:
        return reference_date + timedelta(days=1)
    return schedule.index[0].date()

def run_multi_predictions():
    today_dt = datetime.now()
    today_date = today_dt.date()
    nse = mcal.get_calendar('NSE')
    
    # Check if market is open today
    is_open_today = not nse.schedule(start_date=today_date, end_date=today_date).empty
    
    all_results = []
    print(f"Starting Multi-Stock Analysis for {today_date}...")

    for symbol in TICKERS:
        model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
        if not os.path.exists(model_path): continue
        
        model = joblib.load(model_path)
        df = yf.download(symbol, period="60d", interval="1d", progress=False)
        if df.empty: continue

        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df['Prev_Close'] = df['Close'].shift(1)
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_10'] = df['Close'].rolling(10).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df = df.dropna()

        # HOLIDAY LOGIC
        last_data_date = df.index[-1].date()
        
        if not is_open_today:
            # If holiday, target is the next available session
            target_date = get_next_trading_day(today_date)
        else:
            # If market is open, check if we are post 3:30 PM
            if today_dt.hour < 15 or (today_dt.hour == 15 and today_dt.minute < 30):
                target_date = today_date # Prediction is for today's close
            else:
                target_date = get_next_trading_day(today_date + timedelta(days=1))

        last_row = df.iloc[-1:]
        last_close = float(last_row["Close"].iloc[0])
        X = last_row[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']]
        predicted_price = float(model.predict(X.values)[0])

        # Historical Lookup
        yesterday_pred = "N/A"
        if os.path.exists(PREDICTION_FILE):
            old_preds = pd.read_csv(PREDICTION_FILE)
            match = old_preds[(old_preds["Ticker"] == symbol) & 
                              (old_preds["Target_Date"] == last_data_date.strftime("%Y-%m-%d"))]
            if not match.empty:
                yesterday_pred = float(match.iloc[-1]["Predicted_Next_Close"])

        all_results.append({
            "Ticker": symbol,
            "Prediction_Date": today_date.strftime("%Y-%m-%d"),
            "Target_Date": target_date.strftime("%Y-%m-%d"),
            "Last_Close": round(last_close, 2),
            "Predicted_Next_Close": round(predicted_price, 2),
            "Prev_Prediction_For_Today": yesterday_pred
        })

    new_results_df = pd.DataFrame(all_results)
    if os.path.exists(PREDICTION_FILE):
        old_df = pd.read_csv(PREDICTION_FILE)
        filtered_df = old_df[old_df["Prediction_Date"] != today_date.strftime("%Y-%m-%d")]
        final_df = pd.concat([filtered_df, new_results_df], ignore_index=True)
    else:
        final_df = new_results_df
    
    final_df.to_csv(PREDICTION_FILE, index=False)
    print(f"Success! Predictions saved.")

if __name__ == "__main__":
    run_multi_predictions()