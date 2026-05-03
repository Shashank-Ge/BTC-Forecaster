import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from arch import arch_model
import json, os
import gspread
from google.oauth2.service_account
import Credentials

st.set_page_config(page_title="BTC Next-Hour Forecast", layout="wide")

@st.cache_data(ttl=3600)
def fetch_btc(limit=600):
    url = "https://data-api.binance.vision/api/v3/klines"
    r = requests.get(url, params={"symbol":"BTCUSDT","interval":"1h","limit":limit})
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df["close"].sort_index()

def rolling_entropy(x, window=60, bins=20):
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

def predict_next_hour(prices):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    am  = arch_model(log_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
    res = am.fit(disp='off')
    sigma_fig = res.conditional_volatility / 100
    resid = (log_ret * 100 - res.params['mu']) / res.conditional_volatility
    nu = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])

    H = rolling_entropy(resid)
    M = log_ret.abs().rolling(60).mean()
    bar_sigma2 = (sigma_fig**2).mean()
    redundancy = 1 + 0.1 * np.log1p(prices.rolling(5).var() / prices.rolling(20).var())
    info_filter = (H > H.mean()).astype(float)

    S0 = prices.iloc[-1]
    mu = log_ret.mean()

    H_max = H.max() if H.max() > 0 else 1.0
    M_max = M.max() if M.max() > 0 else 1.0
    H_val = min(H.iloc[-1] / H_max, 1.0)
    M_val = min(M.iloc[-1] / M_max, 1.0)
    crisis = (H_val > 0.8) or (M_val > 0.8)

    α0, δ0 = 0.5, 0.3
    sigma2_last = sigma_fig.iloc[-1]**2
    delta_t = δ0 if crisis else 0.0
    sigma2 = sigma2_last * (1 + α0 * H_val + delta_t * M_val)
    sigma2 = max(1e-6, min(sigma2, 0.5))

    n_sims = 10000
    Z = np.random.standard_t(nu, size=n_sims) * np.sqrt((nu - 2) / nu)
    S1 = S0 * np.exp((mu - 0.5 * sigma2) + np.sqrt(sigma2) * Z)
    lower, upper = np.percentile(S1, [2.5, 97.5])
    return S0, lower, upper

def load_backtest_metrics():
    if not os.path.exists("backtest_results.jsonl"):
        return None, None, None
    rows = []
    with open("backtest_results.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    coverage = df["coverage_95"].mean()
    avg_width = df["width_95"].mean()
    mean_winkler = df["winkler"].mean()
    return coverage, avg_width, mean_winkler

def get_sheet():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("BTC Predictions").sheet1
    return sheet

def save_prediction(sheet, S0, lower, upper):
    from datetime import datetime
    row = [
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        round(S0, 2),
        round(lower, 2),
        round(upper, 2),
        round(upper - lower, 2)
    ]
    sheet.append_row(row)

def load_history(sheet):
    records = sheet.get_all_records()
    return pd.DataFrame(records)


# --- UI ---
st.title("₿ BTC/USDT — Next Hour Forecast")
st.caption("Model: Cyber-GBM with FIGARCH volatility + Student-t tails")

with st.spinner("Fetching live BTC data and running model..."):
    prices = fetch_btc(limit=500)
    S0, lower, upper = predict_next_hour(prices)

# Headline metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current BTC Price", f"${S0:,.2f}")
c2.metric("Predicted Lower (95%)", f"${lower:,.2f}")
c3.metric("Predicted Upper (95%)", f"${upper:,.2f}")
c4.metric("Range Width", f"${upper - lower:,.2f}")

# Chart — last 50 bars + shaded range
st.subheader("Last 50 bars + predicted range")
last50 = prices.tail(50)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(last50.index, last50.values, color='steelblue', linewidth=1.5, label='BTC Price')
ax.axhspan(lower, upper, alpha=0.25, color='orange', label='95% next-hour range')
ax.axhline(S0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel("Price (USDT)")
ax.legend()
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
st.pyplot(fig)

# Backtest metrics
st.subheader("Backtest metrics (Part A)")
coverage, avg_width, mean_winkler = load_backtest_metrics()
if coverage is not None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Coverage (target 0.95)", f"{coverage:.4f}", delta=f"{coverage-0.95:+.4f}")
    m2.metric("Avg Range Width", f"${avg_width:,.2f}")
    m3.metric("Mean Winkler Score", f"{mean_winkler:,.2f}")
else:
    st.info("Run the backtest in Colab first and upload backtest_results.jsonl to see metrics here.")



# Part C — Prediction History
try:
    sheet = get_sheet()
    save_prediction(sheet, S0, lower, upper)
    history = load_history(sheet)

    st.subheader("Prediction History (Part C)")
    st.dataframe(history)

    if len(history) > 1:
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.plot(range(len(history)), history["S0"], label="BTC at prediction time", color="steelblue")
        ax2.fill_between(range(len(history)), history["lower_95"], history["upper_95"],
                         alpha=0.3, color="orange", label="Predicted range")
        ax2.set_xlabel("Visit number")
        ax2.set_ylabel("Price (USDT)")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
except Exception as e:
    st.warning(f"History unavailable: {e}")
