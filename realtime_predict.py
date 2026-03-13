import yfinance as yf
import pandas as pd
import joblib
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

STOCK_SYMBOL = "RELIANCE.NS"
MODEL_PATH = "realtime_rf_model.pkl"
PREDICTION_FILE = "data/predictions.csv"
CHECKPOINT_DIR = "data/checkpoints"

# 1. Create folders if not exist
Path("data").mkdir(exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# 2. Load Model
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found. Train the model first.")
    sys.exit()

model = joblib.load(MODEL_PATH)

print(f"Fetching data for {STOCK_SYMBOL}...")

# 3. Download Data (30d to ensure MA_10 has enough history)
df = yf.download(STOCK_SYMBOL, period="30d", interval="1d")

if df.empty:
    print("No data fetched. Exiting.")
    sys.exit()

# Flatten MultiIndex columns from yfinance
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# ---------------------------------------------------------
# 4. Feature Engineering (Must match your training features)
# ---------------------------------------------------------
df['Prev_Close'] = df['Close'].shift(1)
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_10'] = df['Close'].rolling(10).mean()
df['Daily_Return'] = df['Close'].pct_change()
df = df.dropna()

# 5. Market Open Check (Skips Weekends/Holidays)
last_data_date = df.index[-1].date()
today_date = datetime.today().date()
current_hour = datetime.now().hour
current_minute = datetime.now().minute

print(f"Latest market data available from: {last_data_date}")

# Logic: Only predict for 'Next Day' if it's after Market Close (3:30 PM / 15:30)
# Otherwise, predict for 'Today' using yesterday's data.
market_closed_time = 15 # 3 PM
market_closed_minute = 30

if last_data_date != today_date:
    print(f"Market data for today ({today_date}) not yet available. Predicting for TODAY.")
    target_date = today_date
elif last_data_date == today_date and (current_hour < market_closed_time or (current_hour == market_closed_time and current_minute < market_closed_minute)):
    print("Market is currently OPEN. Prediction for Monday will be based on LIVE price (Not Final).")
    # You can choose to sys.exit() here if you only want final end-of-day predictions
    target_date = today_date + timedelta(days=3) if today_date.weekday() == 4 else today_date + timedelta(days=1)
else:
    print("Market is CLOSED. Predicting for the next session.")
    target_date = today_date + timedelta(days=3) if today_date.weekday() == 4 else today_date + timedelta(days=1)


# ---------------------------------------------------------
# 6. Prepare Prediction for the NEXT session
# ---------------------------------------------------------
last_row = df.iloc[-1:]
last_close = float(last_row["Close"].iloc[0])

# Ensure these columns match exactly what you used to train your .pkl
X = last_row[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']]
predicted_price = float(model.predict(X.values)[0])

# Determine next target trading day (Simple weekend skip)
if today_date.weekday() == 4: # Friday
    target_date = today_date + timedelta(days=3)
else:
    target_date = today_date + timedelta(days=1)

# ---------------------------------------------------------
# 7. Historical Comparison Logic (The Scorecard)
# ---------------------------------------------------------
yesterday_pred = "N/A"
accuracy_gap = "N/A"

if os.path.exists(PREDICTION_FILE):
    old_preds = pd.read_csv(PREDICTION_FILE)
    today_str = today_date.strftime("%Y-%m-%d")
    
    # Find what we predicted yesterday for TODAY
    match = old_preds[old_preds["Target_Date"] == today_str]
    
    if not match.empty:
        yesterday_pred = float(match.iloc[-1]["Predicted_Next_Close"])
        accuracy_gap = round(last_close - yesterday_pred, 2)

# ---------------------------------------------------------
# 8. Save/Update Prediction
# ---------------------------------------------------------
new_row = pd.DataFrame({
    "Prediction_Date": [today_date.strftime("%Y-%m-%d")],
    "Target_Date": [target_date.strftime("%Y-%m-%d")],
    "Last_Close": [round(last_close, 2)],
    "Predicted_Next_Close": [round(predicted_price, 2)],
    "Prev_Prediction_For_Today": [yesterday_pred]
})

if os.path.exists(PREDICTION_FILE):
    old_preds = pd.read_csv(PREDICTION_FILE)
    # Prevent duplicates if script is run twice on the same day
    updated_history = old_preds[old_preds["Prediction_Date"] != today_date.strftime("%Y-%m-%d")]
    final_df = pd.concat([updated_history, new_row], ignore_index=True)
else:
    final_df = new_row

final_df.to_csv(PREDICTION_FILE, index=False)

# 9. Save Checkpoint
checkpoint_path = f"{CHECKPOINT_DIR}/{STOCK_SYMBOL}_{today_date}.csv"
last_row.to_csv(checkpoint_path)

# ----------------------------
# 10. Performance Review Printout
# ----------------------------
print("\n" + "="*45)
print(f"PERFORMANCE REVIEW: {today_date}")
print(f"Actual Close Today   : ₹{last_close:.2f}")
print(f"Yesterday's Forecast : ₹{yesterday_pred if yesterday_pred == 'N/A' else f'{yesterday_pred:.2f}'}")
if accuracy_gap != "N/A":
    status = "OVER" if accuracy_gap > 0 else "UNDER"
    print(f"Accuracy Gap         : ₹{abs(accuracy_gap)} ({status} prediction)")
print("-" * 45)
print(f"NEW FORECAST FOR     : {target_date.strftime('%A, %b %d')}")
print(f"TARGET PRICE         : ₹{predicted_price:.2f}")
print("="*45)