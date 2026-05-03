import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 0;
    color: #F7931A;
}
.subtitle {
    font-size: 0.82rem;
    color: #888;
    margin-top: 0;
    margin-bottom: 1.2rem;
    font-family: 'Space Mono', monospace;
}
.metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    min-height: 100px;
}
.metric-label {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    font-family: 'Inter', sans-serif;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.45rem;
    font-weight: 700;
    color: #ffffff;
    word-break: break-word;
}
.metric-value.orange { color: #F7931A; }
.metric-value.green  { color: #00d4aa; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
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
    padding: 0.5rem 1.2rem;
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.88rem;
    color: #F7931A;
    margin-bottom: 1rem;
}
.updated-tag {
    font-size: 0.72rem;
    color: #555;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1.2rem;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.70rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}
.badge-green  { background: #00d4aa22; color: #00d4aa; border: 1px solid #00d4aa44; }
.badge-orange { background: #F7931A22; color: #F7931A; border: 1px solid #F7931A44; }

.coverage-bar-wrap {
    background: #2a2a4a;
    border-radius: 6px;
    height: 6px;
    margin-top: 10px;
    width: 100%;
}
.coverage-bar-fill {
    height: 6px;
    border-radius: 6px;
    background: linear-gradient(90deg, #00d4aa, #F7931A);
}
.model-pill {
    display: inline-block;
    background: #F7931A18;
    border: 1px solid #F7931A33;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    color: #F7931A;
    margin-right: 6px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor='#0e0e1a',
    plot_bgcolor='#0e0e1a',
    font=dict(color='#888', family='Space Mono, monospace', size=11),
    xaxis=dict(gridcolor='#1e1e30', showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='#1e1e30', showgrid=True, zeroline=False),
    legend=dict(bgcolor='#1a1a2e', bordercolor='#2a2a4a', borderwidth=1,
                font=dict(color='white', size=10)),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='#1a1a2e', bordercolor='#F7931A',
                    font=dict(color='white', family='Space Mono')),
    margin=dict(l=60, r=20, t=20, b=60),
)

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
    return df["coverage_95"].mean(), df["width_95"].mean(), df["winkler"].mean()

def get_sheet():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scope)
    return gspread.authorize(creds).open("BTC Predictions").sheet1

def save_prediction(sheet, S0, lower, upper):
    IST = timezone(timedelta(hours=5, minutes=30))
    sheet.append_row([
        datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        round(S0, 2), round(lower, 2), round(upper, 2), round(upper - lower, 2)
    ])

def load_history(sheet):
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    if not df.empty:
        for col in ["S0", "lower_95", "upper_95", "width"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    return df

def next_candle_countdown():
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    diff = next_hour - now
    return diff.seconds // 60, diff.seconds % 60


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">₿ BTC/USDT — Next Hour Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Model: Cyber-GBM · FIGARCH Volatility · Student-t Fat Tails</div>', unsafe_allow_html=True)

mins, secs = next_candle_countdown()
st.markdown(f'<div class="countdown-box">⏱ Next candle closes in {mins}m {secs}s</div>', unsafe_allow_html=True)

with st.spinner("Fetching live BTC data and running model..."):
    prices = fetch_btc(limit=500)
    S0, lower, upper = predict_next_hour(prices)

IST = timezone(timedelta(hours=5, minutes=30))
st.markdown(
    f'<div class="updated-tag">Last updated: {datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")}</div>',
    unsafe_allow_html=True
)

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "Current BTC Price",     f"${S0:,.2f}",          "orange"),
    (c2, "Predicted Lower (95%)", f"${lower:,.2f}",        ""),
    (c3, "Predicted Upper (95%)", f"${upper:,.2f}",        ""),
    (c4, "Range Width",           f"${upper-lower:,.2f}",  "green"),
]
for col, label, value, cls in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{value}</div>
        </div>""", unsafe_allow_html=True)

# ── Model info expander ───────────────────────────────────────────────────────
with st.expander("ℹ️ How this model works"):
    st.markdown("""
    <div style="font-size:0.85rem; color:#aaa; line-height:1.7;">
    <span class="model-pill">No Peeking</span>
    <span class="model-pill">Fat Tails</span>
    <span class="model-pill">Volatility Clustering</span>
    <br><br>
    <b style="color:#F7931A;">No Peeking</b> — When predicting bar N, the model only uses data up to bar N-1.
    The actual price is revealed only after the prediction is locked in, preventing any data leakage.<br><br>
    <b style="color:#F7931A;">Fat Tails</b> — Bitcoin has frequent large moves that a normal distribution underestimates.
    We use a Student-t distribution (fitted from residuals) to correctly account for extreme price swings.<br><br>
    <b style="color:#F7931A;">Volatility Clustering</b> — FIGARCH volatility model captures the fact that calm hours
    cluster together and volatile hours cluster together. The predicted range widens automatically during turbulent periods.
    </div>
    """, unsafe_allow_html=True)

# ── Main chart — INTERACTIVE ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Last 50 Bars + Predicted Range</div>', unsafe_allow_html=True)

last50 = prices.tail(50)
times  = last50.index.tolist()
closes = last50.values.tolist()

fig1 = go.Figure()

# Shaded range ribbon
fig1.add_trace(go.Scatter(
    x=times + times[::-1],
    y=[upper]*len(times) + [lower]*len(times),
    fill='toself',
    fillcolor='rgba(0,212,170,0.12)',
    line=dict(color='rgba(0,0,0,0)'),
    hoverinfo='skip',
    name='95% range',
    showlegend=True
))

# Upper bound dashed line
fig1.add_trace(go.Scatter(
    x=times, y=[upper]*len(times),
    mode='lines',
    line=dict(color='#00d4aa', width=1, dash='dash'),
    hovertemplate=f'Upper: ${upper:,.2f}<extra></extra>',
    name=f'Upper ${upper:,.2f}'
))

# Lower bound dashed line
fig1.add_trace(go.Scatter(
    x=times, y=[lower]*len(times),
    mode='lines',
    line=dict(color='#00d4aa', width=1, dash='dash'),
    hovertemplate=f'Lower: ${lower:,.2f}<extra></extra>',
    name=f'Lower ${lower:,.2f}'
))

# BTC price line
fig1.add_trace(go.Scatter(
    x=times, y=closes,
    mode='lines',
    line=dict(color='#F7931A', width=2),
    hovertemplate='%{x|%b %d %H:%M}<br>BTC: $%{y:,.2f}<extra></extra>',
    name='BTC Price'
))

fig1.update_layout(
    **PLOTLY_LAYOUT,
    height=380,
    yaxis_title="Price (USDT)",
)
st.plotly_chart(fig1, use_container_width=True)

# ── Backtest metrics ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Backtest Metrics — Part A</div>', unsafe_allow_html=True)

coverage, avg_width, mean_winkler = load_backtest_metrics()
if coverage is not None:
    b1, b2, b3 = st.columns(3)
    delta_cov = coverage - 0.95
    badge_cls = "badge-green" if delta_cov > 0 else "badge-orange"
    badge_txt = f"+{delta_cov:.4f} vs target" if delta_cov > 0 else f"{delta_cov:.4f} vs target"
    bar_pct   = min(coverage / 1.0 * 100, 100)

    with b1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Coverage (target 0.95)</div>
            <div class="metric-value">{coverage:.4f}</div>
            <span class="badge {badge_cls}">{badge_txt}</span>
            <div class="coverage-bar-wrap">
                <div class="coverage-bar-fill" style="width:{bar_pct}%"></div>
            </div>
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

# ── Prediction History — INTERACTIVE ─────────────────────────────────────────
st.markdown('<div class="section-header">Prediction History — Part C</div>', unsafe_allow_html=True)

try:
    sheet   = get_sheet()
    save_prediction(sheet, S0, lower, upper)
    history = load_history(sheet)

    st.dataframe(history, use_container_width=True)

    if len(history) > 1:
        fig2 = go.Figure()

        x_vals = list(range(len(history)))

        # Shaded predicted range
        fig2.add_trace(go.Scatter(
            x=x_vals + x_vals[::-1],
            y=history["upper_95"].tolist() + history["lower_95"].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,212,170,0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            name='Predicted range'
        ))

        # Upper / lower dashed
        fig2.add_trace(go.Scatter(
            x=x_vals, y=history["upper_95"],
            mode='lines',
            line=dict(color='#00d4aa', width=1, dash='dash'),
            hovertemplate='Upper: $%{y:,.2f}<extra></extra>',
            name='Upper 95%'
        ))
        fig2.add_trace(go.Scatter(
            x=x_vals, y=history["lower_95"],
            mode='lines',
            line=dict(color='#00d4aa', width=1, dash='dash'),
            hovertemplate='Lower: $%{y:,.2f}<extra></extra>',
            name='Lower 95%'
        ))

        # BTC price at prediction time
        fig2.add_trace(go.Scatter(
            x=x_vals, y=history["S0"],
            mode='lines+markers',
            line=dict(color='#F7931A', width=2),
            marker=dict(size=5, color='#F7931A'),
            hovertemplate=(
                'Visit %{x}<br>'
                'BTC: $%{y:,.2f}<extra></extra>'
            ),
            name='BTC at prediction time'
        ))

        fig2.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            xaxis_title="Visit number",
            yaxis_title="Price (USDT)",
        )
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.warning(f"History unavailable: {e}")
