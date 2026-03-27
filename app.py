from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime

app = Flask(__name__)
PREDICTION_FILE = "data/predictions.csv"

def get_base_data():
    if not os.path.exists(PREDICTION_FILE):
        return None, None
    df = pd.read_csv(PREDICTION_FILE).replace({np.nan: 'N/A'})
    if df.empty: return None, None
    # Get the absolute latest data regardless of when it was run
    latest_date = df["Prediction_Date"].max()
    return df[df["Prediction_Date"] == latest_date], latest_date

@app.route('/')
def index():
    latest_df, latest_date = get_base_data()
    if latest_df is None:
        return "<h1>Error: predictions.csv not found. Run realtime_predict.py first.</h1>"

    # NSE Holiday Logic for UI Banner
    nse = mcal.get_calendar('NSE')
    today = datetime.now().date()
    is_holiday = nse.schedule(start_date=today, end_date=today).empty

    all_tickers = sorted(latest_df["Ticker"].unique().tolist())
    selected_ticker = request.args.get('ticker', all_tickers[0])
    compare_ticker = request.args.get('compare', "None")

    def process_stock(symbol):
        if symbol == "None": return None
        row = latest_df[latest_df["Ticker"] == symbol].iloc[0]
        actual = float(row['Last_Close'])
        pred = float(row['Predicted_Next_Close'])
        
        yest_val = row.get('Prev_Prediction_For_Today', 'N/A')
        yest_pred = f"{float(yest_val):,.2f}" if yest_val != 'N/A' else "N/A"
        
        move_pct = ((pred - actual) / actual) * 100
        theme = "success" if move_pct > 0.3 else "danger" if move_pct < -0.3 else "neutral"
        rating = "Strong Buy" if move_pct > 0.5 else "Sell" if move_pct < -0.5 else "Hold"

        return {
            "symbol": symbol, "actual": f"{actual:,.2f}", "pred": f"{pred:,.2f}",
            "yest_pred": yest_pred, "pct": round(move_pct, 2), 
            "theme": theme, "rating": rating, "target_date": row['Target_Date'],
            "plan": f"The model predicts a movement of {round(move_pct, 2)}% based on current technical indicators."
        }

    dashboard_data = {
        "is_holiday": is_holiday,
        "tickers": all_tickers,
        "main": process_stock(selected_ticker),
        "comp": process_stock(compare_ticker)
    }

    return render_template('index.html', data=dashboard_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)