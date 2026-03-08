import json
import joblib
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.utils
from flask import Flask, render_template
from datetime import datetime

app = Flask(__name__)

STOCK_SYMBOL = "RELIANCE.NS"
MODEL_PATH = "realtime_rf_model.pkl"

def get_stock_data():
    # 1. Fetch data (30 days to calculate MAs)
    df = yf.download(STOCK_SYMBOL, period="30d", interval="1d")
    if df.empty:
        return None, None, None
    
    # 2. Fix MultiIndex Error
    df.columns = df.columns.get_level_values(0)

    # 3. Feature Engineering (Must match training exactly)
    df['Prev_Close'] = df['Close'].shift(1)
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    df_clean = df.dropna()
    last_row = df_clean.iloc[-1:]
    current_price = last_row["Close"].iloc[0]
    
    # 4. Predict using the same 4 features
    model = joblib.load(MODEL_PATH)
    features = last_row[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']]
    prediction = model.predict(features.values)[0]
    
    # 5. Create Interactive Plot
    plot_df = df_clean.tail(15).reset_index()
    fig = px.line(plot_df, x='Date', y='Close', 
                  title=f'{STOCK_SYMBOL} Price History (Last 15 Sessions)',
                  template="plotly_white")
    
    fig.update_traces(line_color='#0d6efd')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return round(float(current_price), 2), round(float(prediction), 2), graphJSON

@app.route('/')
def index():
    try:
        price, pred, graph = get_stock_data()
        data = {
            "symbol": STOCK_SYMBOL,
            "current": price,
            "predicted": pred,
            "date": datetime.now().strftime("%A, %d %B %Y"),
            "status": "Market Closed" if datetime.now().weekday() >= 5 else "Market Open"
        }
        return render_template('index.html', data=data, graphJSON=graph)
    except Exception as e:
        return f"Error loading dashboard: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)