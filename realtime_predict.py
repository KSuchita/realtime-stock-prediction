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
    schedule = nse.schedule(start_date=reference_date, end_date=reference_date + timedelta(days=10))
    if schedule.empty:
        return reference_date + timedelta(days=1)
    return schedule.index[0].date()

def run_multi_predictions():
    today_dt = datetime.now()
    today_date = today_dt.date()
    nse = mcal.get_calendar('NSE')
    
    # 1. Market Status & Target Date Determination
    schedule_today = nse.schedule(start_date=today_date, end_date=today_date)
    is_open_today = not schedule_today.empty
    
    if not is_open_today:
        target_date = get_next_trading_day(today_date)
    else:
        # Market closes at 15:30 (3:30 PM)
        if today_dt.hour < 15 or (today_dt.hour == 15 and today_dt.minute < 30):
            target_date = today_date 
        else:
            target_date = get_next_trading_day(today_date + timedelta(days=1))

    # 2. Load Existing Data for Duplicate Checking
    existing_df = pd.DataFrame()
    if os.path.exists(PREDICTION_FILE):
        existing_df = pd.read_csv(PREDICTION_FILE)

    print(f"Starting Multi-Stock Analysis for {today_date}")
    print(f"Target Trading Session: {target_date}")
    print("-" * 40)

    all_results = []

    for symbol in TICKERS:
        # --- THE DUPLICATE BLOCKER ---
        if not existing_df.empty:
            # Check if ticker + target_date already exists
            is_predicted = existing_df[(existing_df["Ticker"] == symbol) & 
                                      (existing_df["Target_Date"] == str(target_date))]
            
            if not is_predicted.empty:
                print(f"--- [SKIP] {symbol} ---")
                print(f"Already predicted for {target_date}. Blocking redundant process.")
                continue

        # --- PROCESSING ---
        print(f"--- Analyzing {symbol} ---")
        
        model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
        if not os.path.exists(model_path):
            print(f"Error: Model for {symbol} not found in {MODEL_DIR}")
            continue
        
        model = joblib.load(model_path)
        df = yf.download(symbol, period="60d", interval="1d", progress=False)
        
        if df.empty:
            print(f"Error: Failed to fetch data for {symbol}")
            continue

        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Feature Engineering
        df['Prev_Close'] = df['Close'].shift(1)
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_10'] = df['Close'].rolling(10).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df = df.dropna()

        last_data_date = df.index[-1].date()
        last_row = df.iloc[-1:]
        last_close = float(last_row["Close"].iloc[0])
        X = last_row[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']]
        predicted_price = float(model.predict(X.values)[0])

        # Historical Lookup for UI (What did we predict for 'last_data_date'?)
        yesterday_pred = "N/A"
        if not existing_df.empty:
            match = existing_df[(existing_df["Ticker"] == symbol) & 
                                (existing_df["Target_Date"] == last_data_date.strftime("%Y-%m-%d"))]
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

    # 3. Final Save
    if all_results:
        new_results_df = pd.DataFrame(all_results)
        final_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        final_df.to_csv(PREDICTION_FILE, index=False)
        print(f"\nSuccess! {len(all_results)} new predictions saved to {PREDICTION_FILE}")
    else:
        print("\nProcess Stopped: All tickers already have predictions for this session.")

if __name__ == "__main__":
    run_multi_predictions()