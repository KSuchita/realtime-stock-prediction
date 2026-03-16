from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)
PREDICTION_FILE = "data/predictions.csv"

def get_base_data():
    if not os.path.exists(PREDICTION_FILE):
        return None, None
    df = pd.read_csv(PREDICTION_FILE).replace({np.nan: 'N/A'})
    if df.empty:
        return None, None
    latest_date = df["Prediction_Date"].max()
    return df[df["Prediction_Date"] == latest_date], latest_date

def generate_investment_plan(symbol, move_pct):
    if move_pct > 1.0:
        return f"Institutional demand for {symbol} is surging. The AI identifies a high-conviction breakout pattern."
    elif move_pct > 0.3:
        return f"{symbol} is exhibiting steady organic growth. Market sentiment is cautiously optimistic."
    elif move_pct < -1.0:
        return f"Warning: {symbol} is approaching a liquidity trap. Technical indicators suggest a correction."
    elif move_pct < -0.3:
        return f"Short-term cooling detected for {symbol}. The asset is facing minor resistance."
    else:
        return f"{symbol} is currently range-bound. Minimal volatility expected. Recommend 'Hold'."

@app.route('/')
def index():
    latest_df, latest_date = get_base_data()
    if latest_df is None:
        return "<h1>No data found. Please run realtime_predict.py first.</h1>"

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
        
        if move_pct > 0.5: theme, rating = "success", "Strong Buy"
        elif move_pct < -0.5: theme, rating = "danger", "Sell / Avoid"
        else: theme, rating = "neutral", "Neutral / Hold"

        return {
            "symbol": symbol, "actual": f"{actual:,.2f}", "pred": f"{pred:,.2f}",
            "yest_pred": yest_pred, "pct": round(move_pct, 2), 
            "theme": theme, "rating": rating,
            "plan": generate_investment_plan(symbol, move_pct),
            "target_date": row['Target_Date']
        }

    dashboard_data = {
        "run_date": latest_date,
        "tickers": all_tickers,
        "main": process_stock(selected_ticker),
        "comp": process_stock(compare_ticker)
    }

    return render_template('index.html', data=dashboard_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)