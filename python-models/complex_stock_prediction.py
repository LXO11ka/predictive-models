import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
import hashlib
import warnings

warnings.filterwarnings("ignore")

# 1. Setup
ticker_symbol = "META" 
forecast_out = 365

seed_value = int(hashlib.md5(ticker_symbol.encode()).hexdigest(), 16) % 10**8
np.random.seed(seed_value)

# 2. Daten-Download (Aktie, Markt, Anleihen für Zins-Einfluss)
# Wir laden TNX (10-jährige Staatsanleihen), da Zinsen den Aktienkurs massiv beeinflussen
data_raw = yf.download([ticker_symbol, "^GSPC", "^TNX"], start="2023-01-01")['Close']
data = data_raw.ffill().dropna().copy()

# 3. Profi-Indikatoren (Feature Engineering)
# RSI für Overbought/Oversold
delta = data[ticker_symbol].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
data['RSI'] = 100 - (100 / (1 + (gain / loss)))

# MACD (Trendstärke)
exp1 = data[ticker_symbol].ewm(span=12, adjust=False).mean()
exp2 = data[ticker_symbol].ewm(span=26, adjust=False).mean()
data['MACD'] = exp1 - exp2

# Volatilität und MA-Abweichung
data['MA50'] = data[ticker_symbol].rolling(window=50).mean()
data['Dist_MA50'] = (data[ticker_symbol] - data['MA50']) / data['MA50']
data['Target'] = data[ticker_symbol].shift(-5)

# Fundamentaldaten-Check (KGV)
ticker_info = yf.Ticker(ticker_symbol).info
pe_ratio = ticker_info.get('trailingPE', 25)

df_final = data.dropna()

# 4. KI-Training (Random Forest)
features = [ticker_symbol, '^GSPC', '^TNX', 'RSI', 'MACD', 'Dist_MA50']
X = df_final[features].values
y = df_final['Target'].values

rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X, y)

# 5. Monte Carlo Simulation (250 Pfade)
actual_price = float(data[ticker_symbol].iloc[-1])
last_row = data[features].iloc[-1:].values
prediction_base = float(rf_model.predict(last_row)[0])

# PE-Adjustment: Wenn KGV zu hoch, Zielpreis leicht dämpfen
if pe_ratio > 35: prediction_base *= 0.98

n_sims = 250
future_dates = pd.date_range(start=data.index[-1], periods=forecast_out + 1)
daily_vol = data[ticker_symbol].pct_change().std() * actual_price
sim_matrix = np.zeros((n_sims, forecast_out + 1))

for i in range(n_sims):
    noise = np.random.normal(0, daily_vol, forecast_out + 1)
    drift = np.linspace(0, prediction_base - actual_price, forecast_out + 1)
    path = actual_price + drift + np.cumsum(noise)
    path[0] = actual_price
    sim_matrix[i, :] = path

# 6. Signal-Logik (Kaufen/Verkaufen)
analysis = []
for t in range(2, forecast_out - 5):
    move = np.mean(sim_matrix[:, t+3] - sim_matrix[:, t])
    prob = np.sum(np.sign(sim_matrix[:, t+3] - sim_matrix[:, t]) == np.sign(move)) / n_sims
    analysis.append({
        'date': future_dates[t],
        'score': move * prob,
        'prob': prob,
        'val': np.mean(sim_matrix[:, t])
    })

df_results = pd.DataFrame(analysis)

def filter_signals(df, mode='buy'):
    sort_type = (mode == 'sell')
    raw_sigs = df.sort_values(by='score', ascending=sort_type).head(30)
    filtered = []
    if not raw_sigs.empty:
        last_d = raw_sigs.iloc[0]['date'] - pd.Timedelta(days=20)
        for _, row in raw_sigs.iterrows():
            if abs((row['date'] - last_d).days) > 15:
                filtered.append(row)
                last_d = row['date']
            if len(filtered) >= 2: break
    return pd.DataFrame(filtered)

buys = filter_signals(df_results, 'buy')
sells = filter_signals(df_results, 'sell')

# 7. Visualisierung
plt.figure(figsize=(16, 9))
hist_data = data[ticker_symbol].tail(20)
plt.plot(hist_data.index, hist_data.values, color='black', lw=4, label='Real-Kurs (20 Tage)')

for i in range(15):
    plt.plot(future_dates, sim_matrix[i, :], alpha=0.1, color='blue')

if not buys.empty:
    plt.scatter(buys['date'], buys['val'], color='lime', marker='^', s=350, edgecolors='black', label='BUY SIGNAL', zorder=10)
if not sells.empty:
    plt.scatter(sells['date'], sells['val'], color='red', marker='v', s=350, edgecolors='black', label='SELL SIGNAL', zorder=10)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
plt.title(f'KI-Equity-Engine: {ticker_symbol} | Fundamental & Technisch', fontsize=16)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.15)
plt.tight_layout()
plt.show()

# 8. Bericht
print("\n" + "█"*95)
print(f" STRATEGISCHER AKTIEN-REPORT: {ticker_symbol} ".center(95, "█"))
print("█"*95)
print(f"{'AKTION':<15} | {'DATUM':<12} | {'KURS-LEVEL':<12} | {'KONFIANZ':<10} | {'EXIT-VORSCHLAG'}")
print("-" * 95)

final_table = pd.concat([buys, sells]).sort_values(by='date')
for _, row in final_table.iterrows():
    akt = "KAUFEN" if row['score'] > 0 else "VERKAUFEN"
    exit_date = (row['date'] + pd.Timedelta(days=4)).strftime('%d.%m.%Y')
    print(f"{akt:<15} | {row['date'].strftime('%d.%m.%Y'):<12} | {row['val']:>12.2f} | {row['prob']*100:>8.1f}% | {exit_date}")
print("█"*95)