import yfinance as yf
import pandas as pd
import joblib
import os
import sys
from datetime import datetime
from pathlib import Path

STOCK_SYMBOL = "RELIANCE.NS"
MODEL_PATH = "realtime_rf_model.pkl"
PREDICTION_FILE = "data/predictions.csv"
CHECKPOINT_DIR = "data/checkpoints"

Path("data").mkdir(exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

model = joblib.load(MODEL_PATH)

print(f"Fetching real-time data for {STOCK_SYMBOL}...")

df = yf.download(STOCK_SYMBOL, period="30d", interval="1d")

if df.empty:
    print("No data fetched. Exiting.")
    sys.exit()

df = df.dropna()

# Feature Engineering
df['Prev_Close'] = df['Close'].shift(1)
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_10'] = df['Close'].rolling(10).mean()
df['Daily_Return'] = df['Close'].pct_change()

df.dropna(inplace=True)

last_data_date = df.index[-1].date()
today_date = datetime.today().date()

if last_data_date != today_date:
    print("Market closed today. Prediction skipped.")
    sys.exit()

# Prevent duplicate prediction
if os.path.exists(PREDICTION_FILE):
    old_preds = pd.read_csv(PREDICTION_FILE)
    if today_date.strftime("%Y-%m-%d") in old_preds["Prediction_Date"].astype(str).values:
        print("Prediction already made today.")
        sys.exit()

last_row = df.iloc[-1:]

X = last_row[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']]
last_close = last_row["Close"].iloc[0].item()

predicted_price = model.predict(X)[0]

checkpoint_path = f"{CHECKPOINT_DIR}/{STOCK_SYMBOL}_{today_date}.csv"
last_row.to_csv(checkpoint_path)

print(f"Checkpoint saved: {checkpoint_path}")

new_prediction = pd.DataFrame({
    "Prediction_Date": [today_date.strftime("%Y-%m-%d")],
    "Predicted_Price": [predicted_price]
})

if os.path.exists(PREDICTION_FILE):
    new_prediction.to_csv(PREDICTION_FILE, mode="a", header=False, index=False)
else:
    new_prediction.to_csv(PREDICTION_FILE, index=False)

print("\n==============================")
print(f"Stock                : {STOCK_SYMBOL}")
print(f"Last Closing Price   : ₹{last_close:.2f}")
print(f"Predicted Next Price : ₹{predicted_price:.2f}")

if predicted_price > last_close:
    print("Prediction Direction : UP 📈")
else:
    print("Prediction Direction : DOWN 📉")

print("==============================")