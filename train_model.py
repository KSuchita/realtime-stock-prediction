import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

from data_cleaning import clean_stock_data

STOCK_SYMBOL = "RELIANCE.NS"
MODEL_NAME = "realtime_rf_model.pkl"

print(f"Training model for {STOCK_SYMBOL}...")

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

df = yf.download(
    STOCK_SYMBOL,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d")
)

df = clean_stock_data(df)

# -----------------------------
# Feature Engineering
# -----------------------------
df['Prev_Close'] = df['Close'].shift(1)
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_10'] = df['Close'].rolling(10).mean()
df['Daily_Return'] = df['Close'].pct_change()

# TARGET = NEXT DAY CLOSE
df['Target'] = df['Close'].shift(-1)

df.dropna(inplace=True)

X = df[['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']]
y = df['Target'].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nTraining completed")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

joblib.dump(model, MODEL_NAME)

print(f"Model saved as {MODEL_NAME}")
print("STEP COMPLETED SUCCESSFULLY")