from flask import Flask, render_template
import yfinance as yf
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

STOCK_SYMBOL = "RELIANCE.NS"
MODEL_PATH = "realtime_rf_model.pkl"

def get_predictions():
    # 1. Load Model
    model = joblib.load(MODEL_PATH)
    
    # 2. Fetch Data (Need enough for Moving Averages)
    df = yf.download(STOCK_SYMBOL, period="20d", interval="1d")
    if df.empty:
        return None
    
    # 3. Feature Engineering (Must match your Training script)
    df['Prev_Close'] = df['Close'].shift(1)
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Get the very latest row
    last_row = df.iloc[-1:]
    current_price = last_row["Close"].iloc[0]
    
    # Prepare features for prediction
    X = last_row[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']]
    
    # 4. Predict
    prediction = model.predict(X)[0]
    
    return {
        "symbol": STOCK_SYMBOL,
        "current": round(float(current_price), 2),
        "predicted": round(float(prediction), 2),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.route('/')
def index():
    data = get_predictions()
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)