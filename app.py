import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
from arch import arch_model
import json, os
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone, timedelta

st.set_page_config(
    page_title="BTC Next-Hour Forecast",
    page_icon="₿",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 0;
    color: #F7931A;
}

.subtitle {
    font-size: 0.85rem;
    color: #888;
    margin-top: 0;
    margin-bottom: 1.5rem;
    font-family: 'Space Mono', monospace;
}

.metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.metric-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
}

.metric-value.orange { color: #F7931A; }
.metric-value.green  { color: #00d4aa; }
.metric-value.red    { color: #ff6b6b; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #F7931A;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #2a2a4a;
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
}

.countdown-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #F7931A44;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: #F7931A;
    margin-bottom: 1.5rem;
}

.updated-tag {
    font-size: 0.75rem;
    color: #555;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1.5rem;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}

.badge-green { background: #00d4aa22; color: #00d4aa; border: 1px solid #00d4aa44; }
.badge-orange { background: #F7931A22; color: #F7931A; border: 1px solid #F7931A44; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_btc(limit=600):
    url = "https://data-api.binance.vision/api/v3/klines"
    r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1h", "limit": limit})
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
    sigma2_last = sigma_fig.iloc[-1]**2 * 0.85
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
    coverage     = df["coverage_95"].mean()
    avg_width    = df["width_95"].mean()
    mean_winkler = df["winkler"].mean()
    return coverage, avg_width, mean_winkler

def get_sheet():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    return client.open("BTC Predictions").sheet1

def save_prediction(sheet, S0, lower, upper):
    IST = timezone(timedelta(hours=5, minutes=30))
    row = [
        datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        round(S0, 2),
        round(lower, 2),
        round(upper, 2),
        round(upper - lower, 2)
    ]
    sheet.append_row(row)

def load_history(sheet):
    records = sheet.get_all_records()
    return pd.DataFrame(records)

def next_candle_countdown():
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    diff = next_hour - now
    mins = diff.seconds // 60
    secs = diff.seconds % 60
    return mins, secs


# ── MAIN UI ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">₿ BTC/USDT — Next Hour Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Model: Cyber-GBM · FIGARCH Volatility · Student-t Fat Tails</div>', unsafe_allow_html=True)

# Countdown
mins, secs = next_candle_countdown()
st.markdown(f'<div class="countdown-box">⏱ Next candle closes in {mins}m {secs}s</div>', unsafe_allow_html=True)

# Fetch & predict
with st.spinner("Fetching live BTC data and running model..."):
    prices = fetch_btc(limit=500)
    S0, lower, upper = predict_next_hour(prices)

IST = timezone(timedelta(hours=5, minutes=30))
st.markdown(f'<div class="updated-tag">Last updated: {datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")}</div>', unsafe_allow_html=True)

# ── Metric cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current BTC Price</div>
        <div class="metric-value orange">${S0:,.2f}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Lower (95%)</div>
        <div class="metric-value">${lower:,.2f}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Upper (95%)</div>
        <div class="metric-value">${upper:,.2f}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Range Width</div>
        <div class="metric-value green">${upper - lower:,.2f}</div>
    </div>""", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Last 50 Bars + Predicted Range</div>', unsafe_allow_html=True)

last50 = prices.tail(50)
fig, ax = plt.subplots(figsize=(13, 4))
fig.patch.set_facecolor('#0e0e1a')
ax.set_facecolor('#0e0e1a')

ax.plot(last50.index, last50.values, color='#F7931A', linewidth=1.8, zorder=3, label='BTC Price')
ax.fill_between(last50.index,
                [lower] * len(last50),
                [upper] * len(last50),
                alpha=0.15, color='#00d4aa', zorder=1)
ax.axhline(lower, color='#00d4aa', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axhline(upper, color='#00d4aa', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axhline(S0,    color='#ffffff', linestyle=':',  linewidth=0.6, alpha=0.3)

ax.tick_params(colors='#666', labelsize=8)
ax.spines['bottom'].set_color('#2a2a4a')
ax.spines['left'].set_color('#2a2a4a')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel("Price (USDT)", color='#666', fontsize=8)
ax.yaxis.label.set_color('#666')

patch = mpatches.Patch(color='#00d4aa', alpha=0.4, label='95% next-hour range')
ax.legend(handles=[ax.lines[0], patch],
          facecolor='#1a1a2e', edgecolor='#2a2a4a',
          labelcolor='white', fontsize=8)

plt.xticks(rotation=25)
plt.tight_layout()
st.pyplot(fig)

# ── Backtest metrics ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Backtest Metrics — Part A</div>', unsafe_allow_html=True)

coverage, avg_width, mean_winkler = load_backtest_metrics()
if coverage is not None:
    b1, b2, b3 = st.columns(3)
    delta_cov = coverage - 0.95
    badge = f'<span class="badge badge-green">+{delta_cov:.4f} vs target</span>' if delta_cov > 0 else f'<span class="badge badge-orange">{delta_cov:.4f} vs target</span>'

    with b1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Coverage (target 0.95)</div>
            <div class="metric-value">{coverage:.4f}</div>
            {badge}
        </div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Range Width</div>
            <div class="metric-value">${avg_width:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with b3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Winkler Score</div>
            <div class="metric-value">{mean_winkler:,.2f}</div>
        </div>""", unsafe_allow_html=True)
else:
    st.info("Upload backtest_results.jsonl to see metrics.")

# ── Prediction History ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Prediction History — Part C</div>', unsafe_allow_html=True)

try:
    sheet = get_sheet()
    save_prediction(sheet, S0, lower, upper)
    history = load_history(sheet)

    st.dataframe(
        history.style.set_properties(**{
            'background-color': '#1a1a2e',
            'color': 'white',
            'border': '1px solid #2a2a4a'
        }),
        use_container_width=True
    )

    if len(history) > 1:
        fig2, ax2 = plt.subplots(figsize=(13, 3))
        fig2.patch.set_facecolor('#0e0e1a')
        ax2.set_facecolor('#0e0e1a')

        x = range(len(history))
        ax2.plot(x, history["S0"], color='#F7931A', linewidth=1.5, label='BTC at prediction time', zorder=3)
        ax2.fill_between(x, history["lower_95"], history["upper_95"],
                         alpha=0.15, color='#00d4aa', zorder=1)
        ax2.axhline(history["lower_95"].iloc[-1], color='#00d4aa', linestyle='--', linewidth=0.7, alpha=0.5)
        ax2.axhline(history["upper_95"].iloc[-1], color='#00d4aa', linestyle='--', linewidth=0.7, alpha=0.5)

        ax2.tick_params(colors='#666', labelsize=8)
        ax2.spines['bottom'].set_color('#2a2a4a')
        ax2.spines['left'].set_color('#2a2a4a')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlabel("Visit number", color='#666', fontsize=8)
        ax2.set_ylabel("Price (USDT)", color='#666', fontsize=8)

        patch2 = mpatches.Patch(color='#00d4aa', alpha=0.4, label='Predicted range')
        ax2.legend(handles=[ax2.lines[0], patch2],
                   facecolor='#1a1a2e', edgecolor='#2a2a4a',
                   labelcolor='white', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

except Exception as e:
    st.warning(f"History unavailable: {e}")
