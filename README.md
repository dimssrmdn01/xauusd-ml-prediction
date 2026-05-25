# gold-price-forecast-ml – ML Trading System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)](https://xgboost.ai/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

End-to-end machine learning pipeline for predicting XAUUSD (Gold vs USD) price movements.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Models & Performance](#models--performance)
- [Technologies Used](#technologies-used)
- [Feature Engineering](#feature-engineering)
- [Results Visualization](#results-visualization)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)
- [Contact](#contact)
- [License](#license)

---

## Overview

This project implements a complete machine learning pipeline for predicting XAUUSD (Gold/US Dollar) price direction. Using historical forex data, feature engineering, and advanced ensemble methods, the system aims to forecast short-term price movements.

Key Features:

- Automated Data Pipeline – Fetch, clean, and preprocess historical XAUUSD data
- Feature Engineering – Technical indicators (RSI, MACD, Bollinger Bands, Fibonacci)
- Multiple ML Models – XGBoost, Random Forest, LSTM Neural Networks
- Backtesting Engine – Simulate real trading with realistic transaction costs
- Performance Metrics – Sharpe ratio, max drawdown, win rate, profit factor

## Project Structure

```text
gold-price-forecast-ml/
├── data/
│   ├── raw/                 # Data historis mentah XAUUSD (misal dari OANDA/MT4)
│   ├── processed/           # Data bersih yang sudah dinormalisasi dan ditambah fitur teknikal
│   └── external/            # Data pendukung (misal: kalender ekonomi, data indeks DXY)
├── notebooks/               # Jupyter/R notebooks untuk EDA dan eksperimen awal
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_engineering.ipynb
├── src/                     # Source code utama (Pipeline)
│   ├── __init__.py
│   ├── data_loader.py       # Script penarikan data via API broker
│   ├── features.py          # Script pembuatan fitur prediktif (RSI, MACD, Price Action)
│   ├── train.py             # Script pelatihan model (XGBoost, dll)
│   └── evaluate.py          # Evaluasi metrik model (MSE, Cross-Validation, Sharpe Ratio)
├── models/                  # File model prediktif yang sudah dilatih dan divalidasi (misal: .pkl / .rds)
├── requirements.txt         # Daftar dependensi library (pandas, scikit-learn, xgboost, dll)
├── config.yaml              # Konfigurasi parameter (timeframe 5m/1h, hyperparameter model)
└── README.md                # Dokumentasi utama proyek
```


