# Volatility Forecasting and Regime-Aware Risk Modeling

This project explores whether short-term market volatility can be forecasted using engineered time-series features and machine learning, and whether those forecasts can be used to improve risk management.

The workflow includes:
- data collection from NIFTY and India VIX
- feature engineering using volatility lags, return lags, shocks, and regime signals
- volatility forecasting with Linear Regression and XGBoost
- feature importance analysis
- volatility-targeted position sizing
- backtest-style evaluation versus the benchmark market series
