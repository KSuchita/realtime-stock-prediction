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

# 1. Setup Folders
Path("data").mkdir(exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# 2. Load Model
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found.")
    sys.exit()
model = joblib.load(MODEL_PATH)

print(f"Fetching data for {STOCK_SYMBOL}...")

# 3. Download Data
df = yf.download(STOCK_SYMBOL, period="30d", interval="1d")
if df.empty:
    sys.exit("No data fetched.")

df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# 4. Feature Engineering
df['Prev_Close'] = df['Close'].shift(1)
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_10'] = df['Close'].rolling(10).mean()
df['Daily_Return'] = df['Close'].pct_change()
df = df.dropna()

# ---------------------------------------------------------
# 5. Smart Weekend & Market Logic
# ---------------------------------------------------------
last_data_date = df.index[-1].date() # This will be Friday's date if run on a weekend
today_date = datetime.today().date()
current_weekday = today_date.weekday() # 0=Mon, 4=Fri, 5=Sat, 6=Sun

market_closed_time = 15 
market_closed_minute = 30
now = datetime.now()

# Logic to determine the TARGET DATE
if current_weekday >= 5: # It's Saturday or Sunday
    # Target is the coming Monday
    days_to_monday = 7 - current_weekday
    target_date = today_date + timedelta(days=days_to_monday)
    print(f"It's the weekend. Analysis based on Friday's close ({last_data_date}). Target: Monday.")

elif last_data_date != today_date:
    # It's a weekday morning before data is updated
    target_date = today_date
    print("Market not yet updated today. Predicting for TODAY.")

else:
    # Standard Weekday Logic
    if now.hour < market_closed_time or (now.hour == market_closed_time and now.minute < market_closed_minute):
        print("Market is OPEN. Target is tomorrow (or Monday).")
    else:
        print("Market is CLOSED. Target is the next session.")
    
    # Friday skip to Monday
    if today_date.weekday() == 4:
        target_date = today_date + timedelta(days=3)
    else:
        target_date = today_date + timedelta(days=1)

# ---------------------------------------------------------
# 6. Prepare Prediction
# ---------------------------------------------------------
last_row = df.iloc[-1:]
last_close = float(last_row["Close"].iloc[0])
X = last_row[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']]
predicted_price = float(model.predict(X.values)[0])

# ---------------------------------------------------------
# 7. Scorecard Logic (Fixed for Weekends)
# ---------------------------------------------------------
yesterday_pred = "N/A"
accuracy_gap = "N/A"

if os.path.exists(PREDICTION_FILE):
    old_preds = pd.read_csv(PREDICTION_FILE)
    # Match the prediction that targeted the LAST AVAILABLE MARKET DATA DATE
    # (On Sunday, this looks for the prediction made for Friday)
    match = old_preds[old_preds["Target_Date"] == last_data_date.strftime("%Y-%m-%d")]
    
    if not match.empty:
        yesterday_pred = float(match.iloc[-1]["Predicted_Next_Close"])
        accuracy_gap = round(last_close - yesterday_pred, 2)

# ---------------------------------------------------------
# 8. Save/Update CSV
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
    # Only keep one entry per run date
    updated_history = old_preds[old_preds["Prediction_Date"] != today_date.strftime("%Y-%m-%d")]
    final_df = pd.concat([updated_history, new_row], ignore_index=True)
else:
    final_df = new_row

final_df.to_csv(PREDICTION_FILE, index=False)

# 9. Performance Printout
print("\n" + "="*45)
print(f"PERFORMANCE REVIEW: {last_data_date}")
print(f"Actual Close Today   : ₹{last_close:.2f}")
print(f"Yesterday's Forecast : ₹{yesterday_pred if yesterday_pred == 'N/A' else f'{yesterday_pred:.2f}'}")
if accuracy_gap != "N/A":
    status = "OVER" if accuracy_gap > 0 else "UNDER"
    print(f"Accuracy Gap         : ₹{abs(accuracy_gap)} ({status} prediction)")
print("-" * 45)
print(f"NEW FORECAST FOR     : {target_date.strftime('%A, %b %d')}")
print(f"TARGET PRICE         : ₹{predicted_price:.2f}")
print("="*45)