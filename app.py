from flask import Flask, render_template
import pandas as pd
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)
PREDICTION_FILE = "data/predictions.csv"

def get_multi_stock_data():
    if not os.path.exists(PREDICTION_FILE):
        return None
    
    df = pd.read_csv(PREDICTION_FILE)
    if df.empty:
        return None

    # Replace NaN with 'N/A' for safe display
    df = df.replace({np.nan: 'N/A'})

    # 1. Get the most recent Prediction Date in the entire file
    latest_run_date = df["Prediction_Date"].max()
    
    # 2. Filter for all stocks predicted on that specific date
    latest_df = df[df["Prediction_Date"] == latest_run_date]
    
    stocks_results = []

    for _, row in latest_df.iterrows():
        # Clean numerical values
        try:
            actual_val = float(row['Last_Close'])
            next_pred = float(row['Predicted_Next_Close'])
            yesterday_pred = row['Prev_Prediction_For_Today']
            
            # --- Insight Engine Logic ---
            movement = ((next_pred - actual_val) / actual_val) * 100
            
            if movement > 0.5:
                advice_new = "BUY: Upward trend."
                advice_old = "HOLD: Growth expected."
                theme = "success"
            elif movement < -0.5:
                advice_new = "WAIT: Potential dip."
                advice_old = "SELL: Booking profit."
                theme = "danger"
            else:
                advice_new = "NEUTRAL: Sideways."
                advice_old = "HOLD: Stable outlook."
                theme = "warning"

            # Scorecard Color (Accuracy of previous prediction)
            status_color = "text-secondary"
            if yesterday_pred != 'N/A':
                gap = actual_val - float(yesterday_pred)
                status_color = "text-success" if abs(gap) < (actual_val * 0.01) else "text-danger"

            stocks_results.append({
                "symbol": row['Ticker'],
                "today_actual": actual_val,
                "yesterday_pred": yesterday_pred,
                "next_pred": next_pred,
                "target_date": row['Target_Date'],
                "status_color": status_color,
                "new_user_advice": advice_new,
                "old_user_advice": advice_old,
                "advice_theme": theme
            })
        except Exception as e:
            print(f"Error processing row for {row.get('Ticker')}: {e}")

    return {
        "run_date": latest_run_date,
        "stocks": stocks_results,
        "is_weekend": datetime.now().weekday() >= 5
    }

@app.route('/')
def index():
    dashboard_data = get_multi_stock_data()
    return render_template('index.html', data=dashboard_data)

if __name__ == '__main__':
    app.run(debug=True)