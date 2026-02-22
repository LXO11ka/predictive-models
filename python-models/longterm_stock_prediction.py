import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
import hashlib
import warnings

warnings.filterwarnings("ignore")

# 1. Setup fuer Langfrist-Analyse
ticker_symbol = "KO" 
forecast_out = 750 # Ca. 3 Jahre (Handelstage)


seed_value = int(hashlib.md5(ticker_symbol.encode()).hexdigest(), 16) % 10**8
np.random.seed(seed_value)

# 2. Daten-Download (Wir brauchen mehr Historie fuer 3 Jahre Prognose)
data_raw = yf.download([ticker_symbol, "^GSPC", "^TNX"], start="2020-01-01")['Close']
data = data_raw.ffill().dropna().copy()

# 3. Langfrist-Indikatoren
data['MA200'] = data[ticker_symbol].rolling(window=200).mean()
data['Yearly_Return'] = data[ticker_symbol].pct_change(periods=252) # 1-Jahr Rendite
data['Volatility_Yearly'] = data[ticker_symbol].pct_change().rolling(window=252).std()
data['Target'] = data[ticker_symbol].shift(-60) # Fokus auf 3-Monats-Trends

df_final = data.dropna()

# 4. KI-Training
features = [ticker_symbol, '^GSPC', '^TNX', 'MA200', 'Yearly_Return', 'Volatility_Yearly']
X = df_final[features].values
y = df_final['Target'].values

rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
rf_model.fit(X, y)

# 5. Langfrist-Simulation (Monte Carlo)
actual_price = float(data[ticker_symbol].iloc[-1])
last_row = data[features].iloc[-1:].values
prediction_base = float(rf_model.predict(last_row)[0])

# Berechnung des historischen Drifts (Wachstum pro Tag)
hist_annual_return = data[ticker_symbol].pct_change(252).mean()
daily_drift = (1 + hist_annual_return)**(1/252) - 1

n_sims = 250
future_dates = pd.date_range(start=data.index[-1], periods=forecast_out + 1)
daily_vol = data[ticker_symbol].pct_change().std()

sim_matrix = np.zeros((n_sims, forecast_out + 1))

for i in range(n_sims):
    # Geometrische Brownsche Bewegung (besser fuer Langfrist)
    shocks = np.random.normal(daily_drift, daily_vol, forecast_out + 1)
    price_path = actual_price * np.exp(np.cumsum(shocks))
    price_path[0] = actual_price
    sim_matrix[i, :] = price_path

# Rendite-Berechnung nach 3 Jahren
final_prices = sim_matrix[:, -1]
mean_final_price = np.mean(final_prices)
total_return_pct = ((mean_final_price - actual_price) / actual_price) * 100
annual_return_pct = ((1 + total_return_pct/100)**(1/3) - 1) * 100

# 6. Visualisierung (3 Jahre Ansicht)
plt.figure(figsize=(16, 9))
hist_view = data[ticker_symbol].tail(500) # Letzte 2 Jahre zur Einordnung
plt.plot(hist_view.index, hist_view.values, color='black', lw=3, label='Historie (Real)')

# Simulationen (Langzeit-Korridor)
plt.fill_between(future_dates, np.percentile(sim_matrix, 5, axis=0), 
                 np.percentile(sim_matrix, 95, axis=0), color='blue', alpha=0.1, label='90% Konfidenz-Zone')
plt.plot(future_dates, np.mean(sim_matrix, axis=0), color='blue', lw=3, label='Ø Erwarteter Trend')

plt.text(0.02, 0.90, f'Erwartete Rendite (3 Jahre): {total_return_pct:.2f}%', 
         transform=plt.gca().transAxes, fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
plt.text(0.02, 0.84, f'Ø Rendite pro Jahr: {annual_return_pct:.2f}%', 
         transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.title(f'Langfrist-Investment-Analyse: {ticker_symbol} (3 Jahre)', fontsize=16)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.15)
plt.show()

# 7. Langfrist-Bericht
print("\n" + "█"*95)
print(f" INVESTMENT-REPORT (3 JAHRE): {ticker_symbol} ".center(95, "█"))
print("█"*95)
print(f"Aktueller Preis:          {actual_price:>12.2f} USD")
print(f"Erwarteter Preis (2029):  {mean_final_price:>12.2f} USD")
print("-" * 95)
print(f"GESAMTRENDITE ERWARTET:   {total_return_pct:>12.2f} %")
print(f"Ø RENDITE PRO JAHR:       {annual_return_pct:>12.2f} %")
print("-" * 95)
print("Risiko-Einschätzung:")
print(f"Worst-Case (5% Perzentil): {np.percentile(final_prices, 5):>12.2f} USD")
print(f"Best-Case (95% Perzentil): {np.percentile(final_prices, 95):>12.2f} USD")
print("█"*95)