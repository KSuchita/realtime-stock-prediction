import json
import joblib
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template
from datetime import datetime

app = Flask(__name__)

STOCK_SYMBOL = "RELIANCE.NS"
MODEL_PATH = "realtime_rf_model.pkl"

def get_comparison_data():
    # 1. Fetch data
    df = yf.download(STOCK_SYMBOL, period="60d", interval="1d")
    if df.empty:
        return None, None, None
    
    # 2. Fix Column Names
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # 3. Rebuild clean dataframe with absolute prices
    clean_df = pd.DataFrame({'Close': df.iloc[:, 3].values.flatten()}, index=df.index)
    clean_df['Prev_Close'] = clean_df['Close'].shift(1)
    clean_df['MA_5'] = clean_df['Close'].rolling(5).mean()
    clean_df['MA_10'] = clean_df['Close'].rolling(10).mean()
    clean_df['Daily_Return'] = clean_df['Close'].pct_change()
    clean_df = clean_df.dropna()
    
    # 4. Generate Historical Predictions vs Actuals
    model = joblib.load(MODEL_PATH)
    
    # We take the last 5 days
    history = clean_df.tail(5)
    dates = [d.strftime('%d %b') for d in history.index]
    actual_prices = history['Close'].values.tolist()
    
    # Calculate what the model predicted for THESE specific days
    # To predict 'Today', the model uses 'Yesterday's' features
    predicted_prices = []
    # Get a slightly larger slice to access previous day features
    full_slice = clean_df.tail(6) 
    for i in range(5):
        features = full_slice[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']].iloc[i:i+1].values
        pred = model.predict(features)[0]
        predicted_prices.append(round(float(pred), 2))

    # Current Prediction for Monday
    last_features = clean_df[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']].iloc[-1:].values
    monday_pred = round(float(model.predict(last_features)[0]), 2)
    current_price = round(float(clean_df['Close'].iloc[-1]), 2)

    # 5. Create Grouped Bar Chart
    fig = go.Figure()
    
    # Trace for Actual Prices
    fig.add_trace(go.Bar(
        x=dates,
        y=actual_prices,
        name='Actual Price',
        marker_color='#0d6efd' # Blue
    ))
    
    # Trace for Predicted Prices
    fig.add_trace(go.Bar(
        x=dates,
        y=predicted_prices,
        name='Predicted Price',
        marker_color='#198754' # Green
    ))

    # Zoom Y-axis to see small differences (e.g., between 1400 and 1410)
    y_min = min(min(actual_prices), min(predicted_prices)) * 0.99
    y_max = max(max(actual_prices), max(predicted_prices)) * 1.01

    fig.update_layout(
        title=f"Comparison: Actual vs Predicted (Last 5 Sessions)",
        barmode='group', # This forces them side-by-side
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        yaxis=dict(range=[y_min, y_max]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return current_price, monday_pred, graphJSON

@app.route('/')
def index():
    price, pred, graph = get_comparison_data()
    data = {
        "symbol": STOCK_SYMBOL,
        "current": price,
        "predicted": pred,
        "date": datetime.now().strftime("%A, %d %B %Y")
    }
    return render_template('index.html', data=data, graphJSON=graph)

if __name__ == '__main__':
    app.run(debug=True)