import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Use a try-except for the local import
try:
    from data_cleaning import clean_stock_data
except ImportError:
    def clean_stock_data(df): return df.dropna()

STOCK_SYMBOL = "RELIANCE.NS"
MODEL_NAME = "realtime_rf_model.pkl"

print(f"Training model for {STOCK_SYMBOL}...")

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# 1. Download Data
df = yf.download(
    STOCK_SYMBOL,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d")
)

# 2. CRITICAL FIX: Flatten MultiIndex columns
# This converts ('Close', 'RELIANCE.NS') to just 'Close'
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = clean_stock_data(df)

# 3. Feature Engineering
# We use .copy() to avoid SettingWithCopy warnings
df = df.copy()
df['Prev_Close'] = df['Close'].shift(1)
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_10'] = df['Close'].rolling(10).mean()
df['Daily_Return'] = df['Close'].pct_change()

df.dropna(inplace=True)

# 4. Define Features (X) and Target (y)
# Feature order MUST match the Flask app exactly
features = ['Prev_Close', 'MA_5', 'MA_10', 'Daily_Return']
X = df[features]
y = df['Close']

# 5. Split Data (Shuffle=False is correct for Time Series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 6. Train Model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train.values, y_train.values) # Using .values for purity

# 7. Evaluate
y_pred = model.predict(X_test.values)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Training Results ---")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Features used: {features}")

# 8. Save
joblib.dump(model, MODEL_NAME)
print(f"\nModel saved as {MODEL_NAME}")
print("STEP COMPLETED SUCCESSFULLY")