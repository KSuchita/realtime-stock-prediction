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
    
    # 2. Force Positional Extraction (3 is Close)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    raw_close = df.iloc[:, 3].values.flatten()
    
    # 3. Rebuild clean dataframe
    clean_df = pd.DataFrame({'Close': raw_close}, index=df.index)
    clean_df['Prev_Close'] = clean_df['Close'].shift(1)
    clean_df['MA_5'] = clean_df['Close'].rolling(5).mean()
    clean_df['MA_10'] = clean_df['Close'].rolling(10).mean()
    clean_df['Daily_Return'] = clean_df['Close'].pct_change()
    
    model_df = clean_df.dropna()
    
    # 4. Predict current & historical for comparison
    model = joblib.load(MODEL_PATH)
    
    # Get last 5 days for comparison
    comparison_slice = model_df.tail(6) 
    actual_prices = comparison_slice['Close'].iloc[1:].values # Last 5 actuals
    
    # Generate predictions for those same 5 days using previous day's data
    preds = []
    for i in range(len(comparison_slice) - 1):
        row = comparison_slice[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']].iloc[i:i+1]
        preds.append(model.predict(row.values)[0])
    
    # Current Prediction (for Monday)
    last_row = model_df.iloc[-1:]
    monday_pred = model.predict(last_row[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']].values)[0]
    current_price = last_row["Close"].iloc[0]

    # 5. Create COMPARATIVE BAR GRAPH
    dates = [d.strftime('%b %d') for d in model_df.index[-5:]]
    
    fig = go.Figure(data=[
        go.Bar(name='Actual Price', x=dates, y=actual_prices, marker_color='#0d6efd'),
        go.Bar(name='Predicted Price', x=dates, y=preds[1:], marker_color='#198754')
    ])
    
    fig.update_layout(
        title="Actual vs Predicted Price (Last 5 Sessions)",
        barmode='group',
        template="plotly_white",
        yaxis=dict(range=[min(actual_prices)*0.98, max(actual_prices)*1.02]) # Zoom in on the 1400 range
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return round(float(current_price), 2), round(float(monday_pred), 2), graphJSON

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