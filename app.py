from flask import Flask, render_template
import pandas as pd
import os
import json
import plotly
import plotly.graph_objs as go
from datetime import datetime

app = Flask(__name__)

PREDICTION_FILE = "data/predictions.csv"

def load_latest_data():
    if not os.path.exists(PREDICTION_FILE):
        return None
    df = pd.read_csv(PREDICTION_FILE)
    if df.empty:
        return None
    # Get the latest row which contains today's close and today's prediction
    return df.iloc[-1].to_dict()

@app.route('/')
def index():
    row = load_latest_data()
    
    if row:
        # Determine if we have a valid past prediction to compare against
        yesterday_val = row.get('Prev_Prediction_For_Today', 'N/A')
        actual_val = row.get('Last_Close', 0)
        
        # Calculate Accuracy Status
        status_color = "text-secondary"
        if yesterday_val != 'N/A':
            diff = float(actual_val) - float(yesterday_val)
            status_color = "text-success" if abs(diff) < 10 else "text-danger"

        data = {
            "symbol": "RELIANCE.NS",
            "today_actual": actual_val,
            "yesterday_pred": yesterday_val,
            "next_pred": row.get('Predicted_Next_Close', 0),
            "target_date": row.get('Target_Date', 'N/A'),
            "run_date": row.get('Prediction_Date', 'N/A'),
            "status_color": status_color
        }
    else:
        data = None

    # Simple Chart for visual appeal
    graphJSON = None
    if data:
        fig = go.Figure(data=[
            go.Bar(name='Yesterday Prediction', x=['Comparison'], y=[data['yesterday_pred'] if data['yesterday_pred'] != 'N/A' else 0], marker_color='#adb5bd'),
            go.Bar(name='Today Actual', x=['Comparison'], y=[data['today_actual']], marker_color='#0d6efd'),
            go.Bar(name='Next Prediction', x=['Comparison'], y=[data['next_pred']], marker_color='#198754')
        ])
        fig.update_layout(barmode='group', template="plotly_white", height=300, margin=dict(l=20, r=20, t=20, b=20))
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', data=data, graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)