import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

STOCK_SYMBOL = "RELIANCE.NS"

def next_trading_day(date):

    next_day = date + timedelta(days=1)

    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    return next_day


pred_df = pd.read_csv("data/predictions.csv")

last_pred = pred_df.iloc[-1]

prediction_date = datetime.strptime(
    last_pred["Prediction_Date"],"%Y-%m-%d"
)

evaluation_date = next_trading_day(prediction_date)

df = yf.download(STOCK_SYMBOL,period="5d",interval="1d")

if df.empty:
    print("Market data not available yet")
    exit()

df = df.reset_index()

df["Date"] = pd.to_datetime(df["Date"]).dt.date

match = df[df["Date"] == evaluation_date.date()]

if match.empty:
    print("Market data not available yet")
    exit()

actual_price = match["Close"].iloc[0].item()

predicted_price = float(last_pred["Predicted_Price"])

absolute_error = abs(actual_price - predicted_price)

percentage_error = (absolute_error / actual_price) * 100

print("\n==============================")
print(f"Stock            : {STOCK_SYMBOL}")
print(f"Prediction Date  : {prediction_date.date()}")
print(f"Actual Date      : {evaluation_date.date()}")
print(f"Predicted Price  : ₹{predicted_price:.2f}")
print(f"Actual Price     : ₹{actual_price:.2f}")
print(f"Absolute Error   : ₹{absolute_error:.2f}")
print(f"Percentage Error : {percentage_error:.2f}%")
print("==============================")