import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date, timedelta

# -----------------------------
# Page + Sidebar
# -----------------------------
st.set_page_config(page_title="Multi-Indicator Trend (4h/others)", layout="wide")
st.title("Crypto/Stocks Trend — Interactive (Plotly + Streamlit)")

st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Symbol", value="BTC-USD", help="Any Yahoo Finance symbol (e.g., BTC-USD, ETH-USD, AAPL)")
interval = st.sidebar.selectbox("Interval", ["4h", "1h", "1d"], index=0)
period = st.sidebar.selectbox("Period (download window)", ["30d", "90d", "365d", "730d"], index=3)

# Indicators config
st.sidebar.subheader("Indicators")
show_ma20  = st.sidebar.checkbox("SMA 20", True)
show_ma50  = st.sidebar.checkbox("SMA 50", True)
show_ma200 = st.sidebar.checkbox("SMA 200", False)
show_bb    = st.sidebar.checkbox("Bollinger Bands (20,2)", True)
show_rsi   = st.sidebar.checkbox("RSI (14)", True)
show_macd  = st.sidebar.checkbox("MACD (12,26,9)", True)
show_atr   = st.sidebar.checkbox("ATR (14)", True)

# S/R + breakout config
st.sidebar.subheader("Support / Resistance")
sr_window = st.sidebar.slider("Swing window (bars)", 10, 200, 50, help="Lookback for rolling highs/lows.")
sr_merge_atr_mult = st.sidebar.slider("Merge tolerance (×ATR)", 0.1, 2.0, 0.5, 0.1, help="Near levels are merged if within this ATR multiple.")
detect_breakouts = st.sidebar.checkbox("Detect breakouts/returns", True)

use_log = st.sidebar.checkbox("Log scale (price)", False)
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()

# -----------------------------
# Data
# -----------------------------
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

# -----------------------------
# Indicators
# -----------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

# SMAs
if show_ma20:  df["SMA20"]  = df["Close"].rolling(20).mean()
if show_ma50:  df["SMA50"]  = df["Close"].rolling(50).mean()
if show_ma200: df["SMA200"] = df["Close"].rolling(200).mean()

# Bollinger Bands (20,2)
if show_bb:
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std(ddof=0)
    df["BB_mid"] = mid
    df["BB_up"]  = mid + 2*std
    df["BB_lo"]  = mid - 2*std

# RSI(14)
if show_rsi:
    delta = df["Close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    roll_dn = pd.Series(down, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    df["RSI14"] = 100 - (100 / (1 + rs))

# MACD (12,26,9)
if show_macd:
    ema12 = ema(df["Close"], 12)
    ema26 = ema(df["Close"], 26)
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = ema(df["MACD"], 9)
    df["MACDhist"] = df["MACD"] - df["MACDsig"]

# ATR(14)
if show_atr or detect_breakouts:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()

# -----------------------------
# Date range filter (no re-download)
# -----------------------------
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

# -----------------------------
# Support/Resistance + Breakouts
# -----------------------------
sr_levels = []
breakouts = []

if sr_window and len(df) > sr_window + 5:
    # rolling highs/lows (potential R/S pivots)
    roll_high = df["High"].rolling(sr_window).max()
    roll_low  = df["Low"].rolling(sr_window).min()

    # mark pivots when current high equals rolling high, and current low equals rolling low
    piv_res = df["High"][(df["High"] >= roll_high)]
    piv_sup = df["Low"][(df["Low"] <= roll_low)]

    # merge nearby levels (within ATR multiple) to avoid clutter
    if "ATR14" in df.columns:
        tol = df["ATR14"].median() * sr_merge_atr_mult if not df["ATR14"].dropna().empty else 0
    else:
        tol = df["Close"].median() * 0.002  # small fallback tolerance

    def merge_levels(level_series, kind):
        levels = []
        for ts, lvl in level_series.dropna().items():
            if not levels:
                levels.append({"price": float(lvl), "time": ts, "kind": kind})
                continue
            if abs(lvl - levels[-1]["price"]) <= tol:
                # average nearby levels
                levels[-1]["price"] = (levels[-1]["price"] + float(lvl)) / 2.0
                levels[-1]["time"] = ts
            else:
                levels.append({"price": float(lvl), "time": ts, "kind": kind})
        return levels

    sr_levels = merge_levels(piv_sup, "S") + merge_levels(piv_res, "R")

    # breakout/return detection (simple)
    if detect_breakouts and sr_levels:
        # last known S and R
        last_S = max([l for l in sr_levels if l["kind"] == "S"], key=lambda x: x["time"], default=None)
        last_R = max([l for l in sr_levels if l["kind"] == "R"], key=lambda x: x["time"], default=None)

        c = df["Close"]
        if last_R and (c.shift(1) <= last_R["price"]) & (c > last_R["price"]).iloc[-1]:
            breakouts.append({"when": c.index[-1], "type": "Breakout ↑", "level": last_R["price"]})
        if last_S and (c.shift(1) >= last_S["price"]) & (c < last_S["price"]).iloc[-1]:
            breakouts.append({"when": c.index[-1], "type": "Breakdown ↓", "level": last_S["price"]})

        # pullback/return: price revisits broken level within 0.5*ATR after a breakout
        if "ATR14" in df.columns and breakouts:
            atr = df["ATR14"].iloc[-1]
            for b in breakouts:
                band = atr * 0.5
                lvl = b["level"]
