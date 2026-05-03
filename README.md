# ₿ BTC/USDT — Next Hour Forecast

A live Bitcoin price range forecaster built with a **Cyber-GBM model** using FIGARCH volatility estimation and Student-t fat tails. Predicts the 95% confidence interval for BTC price one hour into the future.

> Built as part of the **AlphaI × Polaris Challenge** — a quantitative forecasting assignment for internship selection.

🔗 **[Live Dashboard](https://btc-forecaster-sg.streamlit.app/)**

---

## What it does

Every hour, a new Bitcoin candle closes. This system:

1. Fetches the latest 500 hourly BTCUSDT bars from Binance's public API
2. Fits a FIGARCH volatility model on recent returns
3. Runs 10,000 Monte Carlo simulations using Student-t distributed shocks
4. Reads off the 95% prediction interval from those simulations
5. Displays the live forecast on a dashboard — updated on every visit
6. Saves each prediction to a Google Sheet for historical tracking (Part C)

---

## Results

| Metric | Value |
|--------|-------|
| Coverage (target 0.95) | **0.9677** |
| Avg Range Width | **$1,383.98** |
| Mean Winkler Score | **1,662.67** |
| Bars backtested | **720 (30 days)** |

---

## The three core concepts

### 1. No Peeking
When predicting bar `N`, the model strictly uses only data up to bar `N-1`. The actual price at `N+1` is accessed only after the prediction is already locked in — purely for scoring. Implemented via:

```python
train_ret = log_ret.iloc[i - train : i]  # never includes bar i+1
```

Accidentally including the current bar in the training window makes backtests look artificially perfect (coverage > 0.99) while the live model fails completely. This is the #1 bug in beginner forecasters.

### 2. Fat Tails
Bitcoin has far more extreme hourly moves than a normal distribution predicts. The model uses a **Student-t distribution** with degrees of freedom fitted from actual BTC return residuals — giving heavier tails and more honest coverage.

```python
Z = np.random.standard_t(nu, size=n_sims) * np.sqrt((nu - 2) / nu)
```

### 3. Volatility Clustering
Calm hours cluster together; volatile hours cluster together. **FIGARCH** (Fractionally Integrated GARCH) captures this directly — the predicted range widens automatically during turbulent periods and narrows during calm ones. This is what makes the forecast adaptive rather than static.

---

## Project structure

```
btc-forecast/
├── app.py                   # Streamlit dashboard (Parts B + C)
├── requirements.txt         # Python dependencies
├── backtest_results.jsonl   # 720 predictions from 30-day backtest (Part A)
└── README.md
```

---

## How the model works

```
Binance API (500 hourly bars)
        │
        ▼
  Log Returns Calculation
        │
        ▼
  FIGARCH Model Fit  ──►  Conditional Volatility (sigma)
        │
        ▼
  Entropy + Momentum Filters  ──►  Crisis Detection
        │
        ▼
  10,000 Monte Carlo Paths  (Student-t shocks)
        │
        ▼
  2.5th / 97.5th Percentile  ──►  95% Prediction Range
```

---

## Backtest methodology (Part A)

```python
for i in range(train, train + 720):
    # STRICT: only use data before bar i
    train_ret = log_ret.iloc[i - train : i]

    # Fit FIGARCH, simulate 10,000 paths
    lower, upper = predict_range(train_ret)

    # Reveal actual — AFTER prediction is locked
    actual = prices.iloc[i + 1]

    # Score
    winkler = compute_winkler(lower, upper, actual, alpha=0.05)
```

The Winkler score penalises both wide ranges (higher width = higher score) and misses (penalty proportional to how far off the actual was). Lower is better.

---

## Dashboard features (Part B + C)

- **Live BTC price** fetched from Binance on every visit
- **95% predicted range** for the next hour
- **Interactive chart** — hover over any point to see exact price and timestamp
- **Countdown timer** showing when the next candle closes
- **Backtest metrics** (coverage, avg width, Winkler score) as headline numbers
- **Prediction history** — every visit is logged to Google Sheets with timestamp (IST), BTC price, lower, upper, and range width
- **Model explainer** — expandable section explaining no-peeking, fat tails, and volatility clustering

---

## Tech stack

| Component | Tool |
|-----------|------|
| Data source | Binance public API (no key needed) |
| Volatility model | FIGARCH via `arch` library |
| Simulation | Monte Carlo, Student-t, 10,000 paths |
| Dashboard | Streamlit |
| Charts | Plotly (interactive) |
| Persistence | Google Sheets via `gspread` |
| Hosting | Streamlit Community Cloud |

---



## Bugs found & fixed

**1. Starter notebook API key expired (401 error)**
The starter Colab used `eodhd.com` for USD/CHF data with a hardcoded API key that returned a 401. Replaced entirely with Binance's public geo-unblocked endpoint (`data-api.binance.vision`) — no API key required.

**2. FIGARCH convergence failures in backtest loop**
On certain 504-bar windows of BTC data, FIGARCH failed to converge and returned garbage volatility estimates silently — no error thrown. This caused coverage to drift unpredictably. Fixed by wrapping each fit in a try-except block with a fallback to simple rolling standard deviation.

**3. Data leakage risk in backtest indexing**
Initial backtest structure risked including bar `i`'s close in the training window for predicting bar `i+1`. Fixed by strict slicing `log_ret.iloc[i - train : i]` and only accessing `prices.iloc[i + 1]` after the prediction was saved.

---

## Submission details

- **Coverage:** 0.9677
- **Mean Winkler:** 1,662.67
- **Dashboard:** https://btc-forecaster-sg.streamlit.app/
- **Challenge:** AlphaI × Polaris Bitcoin Forecasting Challenge
