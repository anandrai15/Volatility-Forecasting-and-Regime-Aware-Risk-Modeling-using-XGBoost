# Volatility Forecasting and Regime-Aware Risk Modeling for NIFTY

This project is a quant-style volatility forecasting framework built on NIFTY and India VIX data. It uses **feature engineering**, **machine learning**, and **XGBoost** to forecast next-day market volatility and convert that forecast into a simple risk-based position sizing strategy.

## Project Overview

Instead of predicting price direction, the model focuses on forecasting **volatility**, which is more useful for risk management, exposure control, and portfolio sizing.

The workflow includes:
- downloading NIFTY and India VIX data
- computing daily returns and realized volatility
- building lagged, shock, and regime-aware features
- training machine learning models
- using XGBoost for volatility prediction
- evaluating feature importance and strategy performance

## Key Features

- **Feature Engineering**: volatility lags, return lags, absolute returns, shock flags, momentum, and regime indicators
- **Machine Learning**: time-series forecasting setup with train/test split based on time
- **XGBoost**: used as the main nonlinear regression model for volatility prediction
- **Risk Modeling**: predicted volatility is converted into a position-sizing rule
- **Strategy Evaluation**: benchmark comparison using return, volatility, and Sharpe ratio

## Main Findings

- Recent volatility is the strongest predictor of future volatility
- Volatility shows strong persistence and clustering
- Return magnitude is more useful than return direction
- Regime and shock features improve the realism of the model
- A volatility forecast is more useful for **risk control** than direct alpha generation

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- XGBoost
- yfinance

## How to Run

##bash
pip install yfinance pandas numpy matplotlib scikit-learn xgboost
python SRC/analysis.py
