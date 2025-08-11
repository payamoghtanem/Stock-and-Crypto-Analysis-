# =============================================================================
# Streamlit + Plotly app
# Yahoo Finance data (crypto/stocks) with 4h/1h/1d candles
# Indicators: SMA, Bollinger Bands, RSI, MACD, ATR
# Extras: Support/Resistance + breakout & pullback, date filter, CSV export
# NOTE: Exploratory tool. Not trading advice.
# =============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date, timedelta

# =============================================================================
# 1) PAGE SETUP
# =============================================================================
st.set_page_config(page_title="Crypto/Stocks Trend — Indicators", layout="wide")
st.title("Crypto/Stocks Trend — Interactive (Plotly + Streamlit)")

# ---- Sidebar: high-level controls -------------------------------------------
st.sidebar.header("Settings")
symbol  = st.sidebar.text_input("Symbol", value="BTC-USD",
                                help="Any Yahoo symbol, e.g. BTC-USD, ETH-USD, AAPL")
interval = st.sidebar.selectbox("Interval", ["4h", "1h", "1d"], index=0)
period   = st.sidebar.selectbox("Period (download window)",
                                ["30d", "90d", "365d", "730d"], index=3)

# ---- Sidebar: indicators -----------------------------------------------------
st.sidebar.subheader("Indicators")
show_ma20  = st.sidebar.checkbox("SMA 20", True)
show_ma50  = st.sidebar.checkbox("SMA 50", True)
show_ma200 = st.sidebar.checkbox("SMA 200", False)
show_bb    = st.sidebar.checkbox("Bollinger Bands (20,2)", True)
show_rsi   = st.sidebar.checkbox("RSI (14)", True)
show_macd  = st.sidebar.checkbox("MACD (12,26,9)", True)
show_atr   = st.sidebar.checkbox("ATR (14)", True)

# ---- Sidebar: S/R + breakouts -----------------------------------------------
st.sidebar.subheader("Support / Resistance")
sr_window = st.sidebar.slider("Swing window (bars)", 10, 200, 50,
                              help="Lookback to find rolling highs/lows.")
sr_merge_atr_mult = st.sidebar.slider("Merge tolerance (×ATR)", 0.1, 2.0, 0.5, 0.1,
                                      help="Nearby levels merged if within ATR × this value.")
detect_breakouts = st.sidebar.checkbox("Detect breakouts/returns", True)

use_log = st.sidebar.checkbox("Log scale (price)", False)
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()

# =============================================================================
# 2) DATA LOADING (cached)
# =============================================================================
@st.cache_data(ttl=60*15, show_spinner=False)
def load_data(sym: str, per: str, inter: str) -> pd.DataFrame:
    df = yf.download(sym, period=per, interval=inter, auto_adjust=False, progress=False)
    return df.dropna()

with st.spinner("Loading data…"):
    df = load_data(symbol, period, interval)

if df.empty:
    st.warning("No data returned. Try a shorter period or a different interval.")
    st.stop()

df = df.copy()

# =============================================================================
# 3) INDICATORS
# =============================================================================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

# -- SMAs
if show_ma20:  df["SMA20"]  = df["Close"].rolling(20).mean()
if show_ma50:  df["SMA50"]  = df["Close"].rolling(50).mean()
if show_ma200: df["SMA200"] = df["Close"].rolling(200).mean()

# -- Bollinger Bands (20, 2σ)
if show_bb:
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std(ddof=0)
    df["BB_mid"] = mid
    df["BB_up"]  = mid + 2*std
    df["BB_lo"]  = mid - 2*std

# -- RSI(14) (pandas-native → always 1-D)
if show_rsi:
    delta  = df["Close"].diff()
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_up = gains.ewm(alpha=1/14, adjust=False).mean()
    avg_dn = losses.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

# -- MACD(12,26,9)
if show_macd:
    ema12 = ema(df["Close"], 12)
    ema26 = ema(df["Close"], 26)
    df["MACD"]    = ema12 - ema26
    df["MACDsig"] = ema(df["MACD"], 9)
    df["MACDhist"]= df["MACD"] - df["MACDsig"]

# -- ATR(14) (also used for S/R merge tolerance)
if show_atr or detect_breakouts:
    pc = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()

# =============================================================================
# 4) DATE FILTER (in-memory, no re-download)
# =============================================================================
min_date = df.index.min().date()
max_date = df.index.max().date()
st.sidebar.subheader("Date range (filter)")
start_date, end_date = st.sidebar.date_input(
    "Start / End",
    value=(max(min_date, max_date - timedelta(days=180)), max_date),
    min_value=min_date, max_value=max_date
)
if isinstance(start_date, date) and isinstance(end_date, date) and start_date <= end_date:
    df = df.loc[str(start_date):str(end_date)]
if df.empty:
    st.warning("No data after filtering.")
    st.stop()

# =============================================================================
# 5) SUPPORT/RESISTANCE + BREAKOUTS  (robust: 1-D arrays + integer indexing)
# =============================================================================
sr_levels, breakouts = [], []
if sr_window and len(df) > sr_window + 5:
    # Rolling extremes; min_periods avoids early NaNs being treated as valid pivots
    roll_high = df["High"].rolling(sr_window, min_periods=sr_window).max()
    roll_low  = df["Low"].rolling(sr_window,  min_periods=sr_window).min()

    # 1) Build 1-D numpy arrays
    high_vals = np.asarray(df["High"]).ravel()
    low_vals  = np.asarray(df["Low"]).ravel()
    rh_vals   = np.asarray(roll_high).ravel()
    rl_vals   = np.asarray(roll_low).ravel()

    # 2) Valid masks: ensure same length + no-NaN comparisons
    valid_res = ~np.isnan(high_vals) & ~np.isnan(rh_vals)
    valid_sup = ~np.isnan(low_vals)  & ~np.isnan(rl_vals)

    mask_res = valid_res & (high_vals >= rh_vals)
    mask_sup = valid_sup & (low_vals  <= rl_vals)

    # 3) Convert masks to integer positions (avoids boolean-index length issues)
    idx_res = np.flatnonzero(mask_res)
    idx_sup = np.flatnonzero(mask_sup)

    # 4) Build Series with matching timestamps by integer take()
    piv_res = pd.Series(high_vals[idx_res], index=df.index.take(idx_res), name="High")
    piv_sup = pd.Series(low_vals[idx_sup],  index=df.index.take(idx_sup),  name="Low")

    # 5) Merge nearby levels using ATR-based tolerance (or small fallback)
    if "ATR14" in df.columns and not df["ATR14"].dropna().empty:
        tol = float(df["ATR14"].median()) * float(sr_merge_atr_mult)
    else:
        tol = float(df["Close"].median()) * 0.002  # ~0.2%

    def merge_levels(series: pd.Series, kind: str):
        s = series.dropna().sort_index()
        levels = []
        for ts, val in s.items():
            try:
                lvl = float(val)
            except Exception:
                continue
            if not levels or abs(lvl - levels[-1]["price"]) > tol:
                levels.append({"price": lvl, "time": ts, "kind": kind})
            else:
                # merge with previous (average price; latest time)
                levels[-1]["price"] = (levels[-1]["price"] + lvl) / 2.0
                levels[-1]["time"]  = ts
        return levels

    sr_levels = merge_levels(piv_sup, "S") + merge_levels(piv_res, "R")

    # 6) Breakout + pullback (simple rules)
    if detect_breakouts and sr_levels:
        last_S = max([l for l in sr_levels if l["kind"] == "S"], key=lambda x: x["time"], default=None)
        last_R = max([l for l in sr_levels if l["kind"] == "R"], key=lambda x: x["time"], default=None)

        c = df["Close"]
        if last_R is not None and (c.shift(1) <= last_R["price"]).iloc[-1] and (c > last_R["price"]).iloc[-1]:
            breakouts.append({"when": c.index[-1], "type": "Breakout ↑", "level": float(last_R["price"])})
        if last_S is not None and (c.shift(1) >= last_S["price"]).iloc[-1] and (c < last_S["price"]).iloc[-1]:
            breakouts.append({"when": c.index[-1], "type": "Breakdown ↓", "level": float(last_S["price"])})

        if "ATR14" in df.columns and not df["ATR14"].dropna().empty and breakouts:
            atr  = float(df["ATR14"].iloc[-1])
            last = float(df["Close"].iloc[-1])
            band = 0.5 * atr
            for b in breakouts:
                lvl = float(b["level"])
                if b["type"] == "Breakout ↑" and (lvl <= last <= lvl + band):
                    b["return"] = "Pullback to R→S"
                if b["type"] == "Breakdown ↓" and (lvl - band <= last <= lvl):
                    b["return"] = "Pullback to S→R"


# =============================================================================
# 6) PLOTTING
# =============================================================================
heights = [0.46, 0.18]            # price + volume are always present
if show_rsi:  heights.append(0.18)
if show_macd: heights.append(0.18)
if show_atr:  heights.append(0.18)
rows = len(heights)

fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    row_heights=heights, vertical_spacing=0.03)

row = 1
# -- Price panel
fig.add_trace(
    go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                   low=df["Low"], close=df["Close"], name="Price"),
    row=row, col=1
)
for col, dash in [("SMA20","solid"), ("SMA50","dot"), ("SMA200","dash")]:
    if col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                 mode="lines", line=dict(dash=dash)),
                      row=row, col=1)
if show_bb and {"BB_up","BB_lo"} <= set(df.columns):
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB up", mode="lines", opacity=0.6),
                  row=row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lo"], name="BB lo", mode="lines", opacity=0.6),
                  row=row, col=1)

# S/R lines + breakout markers
for lvl in sr_levels:
    fig.add_hline(y=lvl["price"], line_width=1, opacity=0.3,
                  line_dash="dot" if lvl["kind"] == "S" else "dash",
                  row=row, col=1)
for b in breakouts:
    label = b["type"] + (f" ({b.get('return')})" if b.get("return") else "")
    fig.add_trace(go.Scatter(x=[b["when"]], y=[b["level"]], mode="markers+text",
                             text=[label], textposition="top center", name=b["type"]),
                  row=row, col=1)

# -- Volume
row += 1
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.6),
              row=row, col=1)

# -- RSI
if show_rsi:
    row += 1
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", mode="lines"),
                  row=row, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2,
                  line_width=0, row=row, col=1)

# -- MACD
if show_macd:
    row += 1
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"),
                  row=row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACDsig"], name="Signal", mode="lines"),
                  row=row, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACDhist"], name="Hist", opacity=0.5),
                  row=row, col=1)

# -- ATR
if show_atr:
    row += 1
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR14"], name="ATR14", mode="lines"),
                  row=row, col=1)

fig.update_layout(
    title=f"{symbol} — {interval} ({period})",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True, type=("log" if use_log else "linear"), row=1, col=1)

st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 7) FOOTER
# =============================================================================
c1, c2, c3 = st.columns(3)
c1.metric("Last Close", f"{float(df['Close'].iloc[-1]):,.2f}")
c2.metric("Bars", f"{len(df):,}")
c3.metric("Start → End", f"{df.index[0].date()} → {df.index[-1].date()}")

with st.expander("Show data (last 1000 rows)"):
    st.dataframe(df.tail(1000), use_container_width=True)

st.download_button("Download CSV", df.to_csv().encode("utf-8"),
                   file_name=f"{symbol}_{interval}_{period}.csv")
