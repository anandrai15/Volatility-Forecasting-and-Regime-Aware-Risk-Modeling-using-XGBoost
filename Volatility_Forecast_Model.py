"""
Volatility Forecasting and Regime-Aware Risk Model

This script:
1. Downloads NIFTY & VIX data
2. Engineers features (lags, shocks, regimes)
3. Trains XGBoost model
4. Forecasts volatility
5. Applies risk-based position sizing
6. Evaluates strategy performance

Core Idea:
→ Predict risk (volatility), not price
→ Use prediction for exposure control
"""

# || IMPORTS ||

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# || DATA DOWNLOAD ||

# Download market data
Nifty = yf.download("^NSEI", start="2020-01-01")
VIX = yf.download("^INDIAVIX", start="2020-01-01")

# Clean column structure (important for yfinance)
if isinstance(Nifty.columns, pd.MultiIndex):
    Nifty.columns = Nifty.columns.get_level_values(0)

if isinstance(VIX.columns, pd.MultiIndex):
    VIX.columns = VIX.columns.get_level_values(0)

# Keep only closing prices
Nifty = Nifty[['Close']].rename(columns={'Close': 'Nifty_close'})
VIX = VIX[['Close']].rename(columns={'Close': 'VIX_close'})

# Merge datasets
data = Nifty.join(VIX, how='inner')

# Save raw data
data.to_csv("data/market_data.csv")
print("✅ Data saved successfully")


# || LOAD + BASIC FEATURES ||

data = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)

# Ensure numeric values
data["Nifty_close"] = pd.to_numeric(data["Nifty_close"], errors="coerce")
data["VIX_close"] = pd.to_numeric(data["VIX_close"], errors="coerce")

# || RETURNS ||

# Why: returns represent price movement (not raw price)
data["Nifty_return"] = data["Nifty_close"].pct_change()
data["VIX_return"] = data["VIX_close"].pct_change()

# || VOLATILITY ||

# Why: volatility = risk (what we want to predict)
data['volatility_21'] = data['Nifty_return'].rolling(21).std() * np.sqrt(252)

data.dropna(inplace=True)

# Correlation insight
correlation = data["Nifty_return"].corr(data["VIX_return"])
print(f"Correlation between Nifty and VIX returns: {correlation:.4f}")


# || FEATURE ENGINEERING ||

# Core idea:
# Convert time-series into supervised learning (X → y)

# --- Volatility Lags (MOST IMPORTANT SIGNAL) ---
data['vol_lag1'] = data['volatility_21'].shift(1)
data['vol_lag2'] = data['volatility_21'].shift(2)
data['vol_lag3'] = data['volatility_21'].shift(3)
data['vol_lag5'] = data['volatility_21'].shift(5)

# --- Return Lags ---
# Why: market movement affects future volatility
data['return_lag1'] = data['Nifty_return'].shift(1)
data['return_lag2'] = data['Nifty_return'].shift(2)
data['return_lag3'] = data['Nifty_return'].shift(3)
data['return_lag5'] = data['Nifty_return'].shift(5)

# --- Absolute Returns ---
# Why: magnitude matters more than direction
data['abs_return_lag1'] = data['return_lag1'].abs()
data['abs_return_lag2'] = data['return_lag2'].abs()
data['abs_return_lag3'] = data['return_lag3'].abs()
data['abs_return_lag5'] = data['return_lag5'].abs()

# --- VIX Lags ---
# Why: external volatility signal
data['vix_lag1'] = data['VIX_return'].shift(1)
data['vix_lag2'] = data['VIX_return'].shift(2)
data['vix_lag3'] = data['VIX_return'].shift(3)
data['vix_lag5'] = data['VIX_return'].shift(5)

# || TARGET ||

# Predict next day's volatility
data['target_vol'] = data['volatility_21'].shift(-1)


# || SHOCK + DYNAMICS FEATURES ||

# Detect unusual moves (shock)
data['abs_return_mean_20'] = data['Nifty_return'].rolling(20).mean().abs()
data['shock'] = (data['Nifty_return'].abs() > data['abs_return_mean_20']).astype(int)

# Volatility change (trend direction)
data['vol_change'] = data['volatility_21'] - data['volatility_21'].shift(1)

# Volatility momentum (short-term trend)
data['vol_momentum'] = data['volatility_21'].rolling(3).mean()


# || REGIME FEATURE ||

# Identify high vs low volatility environments
data['vol_mean_10'] = data['volatility_21'].rolling(10).mean()
data['high_vol_regime'] = (data['volatility_21'] > data['vol_mean_10']).astype(int)


data.dropna(inplace=True)

# Save processed data
data.to_csv("data/processed_data.csv")
print("✅ Processed data saved")


# || TRAIN-TEST SPLIT ||

data = pd.read_csv("data/processed_data.csv", index_col=0, parse_dates=True)

features = [
    "vol_lag1","vol_lag2","vol_lag3","vol_lag5",
    "vix_lag1","vix_lag2","vix_lag3","vix_lag5",
    "return_lag1","return_lag2","return_lag3","return_lag5",
    "abs_return_lag1","abs_return_lag2","abs_return_lag3","abs_return_lag5",
    "shock","vol_change","vol_momentum","high_vol_regime"
]

target = "target_vol"

# Important: no random split (time-series)
train_data = data[:'2023-01-01']
test_data = data['2023-01-01':]


# || MODEL TRAINING ||

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

model.fit(train_data[features], train_data[target])

prediction = model.predict(test_data[features])

rmse = np.sqrt(mean_squared_error(test_data[target], prediction))


# || MODEL RESULTS ||

print("\n" + "="*50)
print("MODEL RESULTS")
print("="*50)
print("XGBOOST RMSE:", rmse)


# || PLOT PREDICTION ||

plt.figure(figsize=(12,6))
plt.plot(test_data.index, test_data[target], label="Actual")
plt.plot(test_data.index, prediction, label="Predicted")
plt.legend()
plt.title("Volatility Prediction")
plt.savefig("data/actual_vs_predicted.png")
plt.close()


# || FEATURE IMPORTANCE ||

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values(by='importance', ascending=False)

print("\nFeature Importance:\n", feature_importance)

plt.figure(figsize=(8,5))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.savefig("data/feature_importance.png")
plt.close()


# || POSITION SIZING (CORE QUANT LOGIC) ||

# Convert prediction → exposure decision
test_data['predicted_vol'] = prediction

# Base rule: inverse volatility
test_data['position_size'] = 1 / (test_data['predicted_vol'] + 1e-6)

# Reduce exposure in risky conditions
test_data.loc[test_data['high_vol_regime'] == 1, 'position_size'] *= 0.7
test_data.loc[test_data['shock'] == 1, 'position_size'] *= 0.5

# Prevent leverage explosion
test_data['position_size'] = test_data['position_size'].clip(0.5, 1.2)

# Normalize exposure
test_data['position_size'] /= test_data['position_size'].mean()

print("Position sizing created")


# || STRATEGY RETURNS ||

test_data['strategy_return'] = test_data['position_size'] * test_data['Nifty_return']

test_data['cum_market'] = (1 + test_data['Nifty_return']).cumprod()
test_data['cum_strategy'] = (1 + test_data['strategy_return']).cumprod()


# || PERFORMANCE PLOT ||

plt.figure(figsize=(12,6))
plt.plot(test_data.index, test_data['cum_market'], label="Market")
plt.plot(test_data.index, test_data['cum_strategy'], label="Strategy")
plt.legend()
plt.title("Volatility Strategy Performance")
plt.savefig("data/strategy_performance.png")
plt.close()


# || METRICS ||

market_vol = test_data['Nifty_return'].std() * np.sqrt(252)
strategy_vol = test_data['strategy_return'].std() * np.sqrt(252)

market_sharpe = test_data['Nifty_return'].mean() / test_data['Nifty_return'].std()
strategy_sharpe = test_data['strategy_return'].mean() / test_data['strategy_return'].std()

print("\n========== PERFORMANCE ==========")
print("Market Return:", test_data['cum_market'].iloc[-1])
print("Strategy Return:", test_data['cum_strategy'].iloc[-1])

print("\nVolatility:")
print("Market:", market_vol)
print("Strategy:", strategy_vol)

print("\nSharpe:")
print("Market:", market_sharpe)
print("Strategy:", strategy_sharpe)