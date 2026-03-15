from flask import Flask, render_template
import pandas as pd
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)
PREDICTION_FILE = "data/predictions.csv"

def load_latest_data():
    if not os.path.exists(PREDICTION_FILE):
        return None
    df = pd.read_csv(PREDICTION_FILE)
    if df.empty:
        return None
    df = df.replace({np.nan: 'N/A'})
    return df.iloc[-1].to_dict()

@app.route('/')
def index():
    row = load_latest_data()
    if row:
        yesterday_val = row.get('Prev_Prediction_For_Today', 'N/A')
        actual_val = float(row.get('Last_Close', 0))
        next_pred = float(row.get('Predicted_Next_Close', 0))
        
        # --- Insight Engine Logic ---
        movement = ((next_pred - actual_val) / actual_val) * 100
        
        if movement > 0.5:  # Bullish
            new_user = "BUY: Model suggests an upward trend. Good entry point."
            old_user = "HOLD: Growth expected. Maintain position for gains."
            theme = "success"
        elif movement < -0.5:  # Bearish
            new_user = "WAIT: Potential dip ahead. Avoid buying at current price."
            old_user = "SELL/CAUTION: Trend is downward. Consider booking profit."
            theme = "danger"
        else:  # Neutral
            new_user = "NEUTRAL: Minimal movement expected. Range-bound."
            old_user = "HOLD: Stable outlook. No immediate action needed."
            theme = "warning"
        
        # Status Color for Scorecard
        status_color = "text-secondary"
        if yesterday_val != 'N/A':
            diff = actual_val - float(yesterday_val)
            status_color = "text-success" if abs(diff) < 10 else "text-danger"

        data = {
            "symbol": "RELIANCE.NS",
            "today_actual": actual_val,
            "yesterday_pred": yesterday_val,
            "next_pred": next_pred,
            "target_date": row.get('Target_Date', 'N/A'),
            "run_date": row.get('Prediction_Date', 'N/A'),
            "status_color": status_color,
            "is_weekend": datetime.now().weekday() >= 5,
            "new_user_advice": new_user,
            "old_user_advice": old_user,
            "advice_theme": theme
        }
    else:
        data = None

    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)