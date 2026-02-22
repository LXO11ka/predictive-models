import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
ticker = "META"
data = yf.download(ticker, start="2023-01-01")

# Fix for multi-index columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Save the absolute latest price before dropping rows
# This ensures we always have the real current price
actual_latest_price = float(data['Close'].iloc[-1])

# new goal: price in 8 days
forecast_out = 365
data['Target'] = data['Close'].shift(-forecast_out)

# Create a copy for training and drop the empty rows (the last 8 days)
train_data = data.dropna().copy()

# Train model:
X = train_data[['Close']].values 
y = train_data['Target'].values 

model = LinearRegression()
model.fit(X, y)

# predict the price in 8 days based on the actual latest price
predicted_price = model.predict([[actual_latest_price]])

print("-" * 30)
print(f"Today's price: {actual_latest_price:.2f}")
print(f"Predicted price for {forecast_out} days from now: {predicted_price[0]:.2f}")
print("-" * 30)