from fastapi import FastAPI
import yfinance as yf
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load("../realtime_rf_model.pkl")

STOCK_OPTIONS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

@app.get("/stocks")
def get_stocks():
    return STOCK_OPTIONS

@app.get("/predict/{symbol}")
def predict(symbol: str):

    data = yf.download(symbol, period="30d")

    if data.empty:
        return {"error": "Invalid stock symbol"}

    latest = data.iloc[-1]

    features = [[
        latest["Open"],
        latest["High"],
        latest["Low"],
        latest["Close"],
        latest["Volume"]
    ]]

    prediction = model.predict(features)[0]

    return {
        "current_price": round(latest["Close"], 2),
        "predicted_price": round(prediction, 2),
        "history": list(data["Close"].tail(7))
    }