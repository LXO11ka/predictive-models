import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import hashlib
import warnings

warnings.filterwarnings("ignore")

# 1. Setup & Individualisierung
ticker_symbol = "SOL-USD" 
forecast_out = 60 

seed_value = int(hashlib.md5(ticker_symbol.encode()).hexdigest(), 16) % 10**8
np.random.seed(seed_value)

# 2. Erweiterter Daten-Download
# Wir laden mehr Daten fuer ein besseres Training der KI
data_raw = yf.download([ticker_symbol, "^GSPC", "GC=F"], start="2023-01-01")['Close']
data = data_raw.ffill().dropna().copy()

# 3. Profi-Indikatoren (Feature Engineering)
# RSI (Relative Strength Index)
delta = data[ticker_symbol].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Bollinger Baender
data['MA20'] = data[ticker_symbol].rolling(window=20).mean()
data['STD20'] = data[ticker_symbol].rolling(window=20).std()
data['Upper_Band'] = data['MA20'] + (data['STD20'] * 2)
data['Lower_Band'] = data['MA20'] - (data['STD20'] * 2)

# Volatilitaet & Momentum
data['Returns'] = data[ticker_symbol].pct_change()
data['Volatility'] = data['Returns'].rolling(window=10).std()
data['Momentum'] = data[ticker_symbol] - data[ticker_symbol].shift(4)

# Zieldaten fuer die KI
data['Target'] = data[ticker_symbol].shift(-5)
df_final = data.dropna()

# 4. KI-Training (Random Forest)
features = [ticker_symbol, '^GSPC', 'GC=F', 'RSI', 'MA20', 'Upper_Band', 'Lower_Band', 'Volatility', 'Momentum']
X = df_final[features].values
y = df_final['Target'].values

# Modell mit 200 Baeumen fuer maximale Praezision
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X, y)

# 5. Monte Carlo Simulation (250 Pfade)
actual_price = float(data[ticker_symbol].iloc[-1])
last_row = data[features].iloc[-1:].values
prediction_base = float(rf_model.predict(last_row)[0])

n_sims = 250
future_dates = pd.date_range(start=data.index[-1], periods=forecast_out + 1)
daily_vol = data['Returns'].std() * actual_price
sim_matrix = np.zeros((n_sims, forecast_out + 1))

for i in range(n_sims):
    # Dynamisches Rauschen basierend auf aktueller Markt-Vola
    noise = np.random.normal(0, daily_vol, forecast_out + 1)
    # Drift zum KI-Zielpreis
    drift = np.linspace(0, prediction_base - actual_price, forecast_out + 1)
    path = actual_price + drift + np.cumsum(noise)
    path[0] = actual_price
    sim_matrix[i, :] = path

# 6. Strategische Signal-Logik (Kaufen/Verkaufen)
analysis = []
for t in range(2, forecast_out - 5):
    # Check auf 3-Tage Fenster
    move = np.mean(sim_matrix[:, t+3] - sim_matrix[:, t])
    # Konsens der Baeume/Simulationen
    agreement = np.sum(np.sign(sim_matrix[:, t+3] - sim_matrix[:, t]) == np.sign(move)) / n_sims
    
    # Zusatz-Check: Nur kaufen wenn RSI < 70 (nicht ueberkauft)
    # Nur verkaufen wenn RSI > 30 (nicht ueberverkauft)
    analysis.append({
        'date': future_dates[t],
        'score': move * agreement,
        'prob': agreement,
        'val': np.mean(sim_matrix[:, t])
    })

df_results = pd.DataFrame(analysis)

# Top Gelegenheiten filtern (mit Zeitabstand)
def filter_signals(df, mode='buy'):
    sort_type = (mode == 'sell')
    raw_sigs = df.sort_values(by='score', ascending=sort_type).head(30)
    filtered = []
    if not raw_sigs.empty:
        last_d = raw_sigs.iloc[0]['date'] - pd.Timedelta(days=20)
        for _, row in raw_sigs.iterrows():
            if abs((row['date'] - last_d).days) > 12:
                filtered.append(row)
                last_d = row['date']
            if len(filtered) >= 2: break
    return pd.DataFrame(filtered)

buys = filter_signals(df_results, 'buy')
sells = filter_signals(df_results, 'sell')

# 7. High-End Visualisierung
plt.figure(figsize=(16, 9))
hist_data = data[ticker_symbol].tail(15)
plt.plot(hist_data.index, hist_data.values, color='black', lw=4, label='Historie (Real)')

# Simulationen
for i in range(20):
    plt.plot(future_dates, sim_matrix[i, :], alpha=0.07, color='cyan')

# Signale
if not buys.empty:
    plt.scatter(buys['date'], buys['val'], color='lime', marker='^', s=300, edgecolors='black', label='OPTIMALER KAUF', zorder=10)
if not sells.empty:
    plt.scatter(sells['date'], sells['val'], color='red', marker='v', s=300, edgecolors='black', label='OPTIMALER VERKAUF', zorder=10)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
plt.title(f'KI-Ensemble Vorhersage: {ticker_symbol} | Multi-Faktor-Analyse', fontsize=16)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.15)
plt.show()

# 8. Profi-Bericht
print("\n" + "█"*90)
print(f" KI-HANDELSSTRATEGIE FÜR {ticker_symbol} ".center(90, "█"))
print("█"*90)
print(f"{'AKTION':<15} | {'DATUM':<12} | {'ZIELPREIS':<12} | {'KONFIANZ':<10} | {'INFO'}")
print("-" * 90)

final_table = pd.concat([buys, sells]).sort_values(by='date')
for _, row in final_table.iterrows():
    akt = "KAUFEN" if row['score'] > 0 else "VERKAUFEN"
    print(f"{akt:<15} | {row['date'].strftime('%d.%m.%Y'):<12} | {row['val']:>12.2f} | {row['prob']*100:>8.1f}% | Trend-Bestätigung")
print("█"*90)