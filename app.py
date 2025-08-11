# =============================================================================
# Streamlit + Plotly Technical Chart App (Yahoo Finance)
# -----------------------------------------------------------------------------
# What this app does
# - Downloads OHLCV data for any Yahoo symbol (e.g., BTC-USD, ETH-USD, AAPL).
# - Shows price as Candles OR as a Close-line (toggle).
# - Adds indicators: SMA(20/50/200), Bollinger Bands(20,2), RSI(14), MACD(12,26,9), ATR(14).
# - Adds trendlines: EMA(N) and Linear (OLS) across the current visible window.
# - Estimates Support/Resistance using rolling extremes, merges nearby levels using ATR,
#   and marks simple breakouts & pullbacks.
# - Lets you filter the date range WITHOUT re-downloading data.
# - Exports the filtered dataset as CSV.
#
# Design choices for readability
# - Row heights are tuned so each panel is easy to read:
#     Price ≈ 45%, Volume ≈ 15%, RSI ≈ 20%, MACD ≈ 20%, ATR ≈ 20%
# - All panels share the same time axis for alignment.
#
# Notes
# - This is an educational/exploratory tool. Not trading advice.
# - Indicators and S/R are recomputed dynamically after the date filter.
# - The S/R logic is intentionally simple; swap for your preferred method if needed.
# =============================================================================

from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


# =============================================================================
# 1) PAGE + SIDEBAR CONTROLS
# =============================================================================
st.set_page_config(page_title="Crypto/Stocks Trend — Indicators", layout="wide")
st.title("Crypto/Stocks Trend — Interactive (Plotly + Streamlit)")

# --- Core inputs: symbol, interval, period -----------------------------------
# Symbol can be crypto, stock, index, ETF, etc., as long as Yahoo supports it.
st.sidebar.header("Settings")
symbol   = st.sidebar.text_input("Symbol", value="BTC-USD",
                                 help="Examples: BTC-USD, ETH-USD, AAPL, TSLA, ^GSPC")

# Interval controls the candle size. Yahoo limits history for intraday intervals.
interval = st.sidebar.selectbox("Interval", ["4h", "1h", "1d"], index=0)

# Period controls how much history we ask from Yahoo in one request.
period   = st.sidebar.selectbox("Period (download window)",
                                ["30d", "90d", "365d", "730d"], index=3)

# --- Price visualization options ---------------------------------------------
st.sidebar.subheader("Price view")
price_mode  = st.sidebar.selectbox("Display", ["Candles", "Line (Close)"], index=0,
                                   help="Candles show OHLC; Line is the Close series only.")
show_volume = st.sidebar.checkbox("Show volume panel", True)

# --- Trendlines (overlays on the price chart) --------------------------------
st.sidebar.subheader("Trend lines")
ema_len = st.sidebar.number_input("EMA length", min_value=5, max_value=400, value=50, step=5,
                                  help="Exponential Moving Average window.")
show_ema_trend = st.sidebar.checkbox("Show EMA trend", False)
show_lin_trend = st.sidebar.checkbox("Show linear trend (OLS)", False)

# --- Indicators toggles -------------------------------------------------------
st.sidebar.subheader("Indicators")
show_ma20  = st.sidebar.checkbox("SMA 20", True)
show_ma50  = st.sidebar.checkbox("SMA 50", True)
show_ma200 = st.sidebar.checkbox("SMA 200", False)
show_bb    = st.sidebar.checkbox("Bollinger Bands (20,2)", True)
show_rsi   = st.sidebar.checkbox("RSI (14)", True)
show_macd  = st.sidebar.checkbox("MACD (12,26,9)", True)
show_atr   = st.sidebar.checkbox("ATR (14)", True)

# --- Support/Resistance + breakout rules -------------------------------------
st.sidebar.subheader("Support / Resistance")
sr_window = st.sidebar.slider("Swing window (bars)", 10, 200, 50,
                              help="Rolling lookback for highs/lows used as S/R candidates.")
sr_merge_atr_mult = st.sidebar.slider("Merge tolerance (×ATR)", 0.1, 2.0, 0.5, 0.1,
                                      help="Nearby levels are merged if within ATR × this value.")
detect_breakouts = st.sidebar.checkbox("Detect breakouts/returns", True,
                                       help="Flags close cross above R or below S. Pullback = revisit within 0.5×ATR.")

# --- Misc UI ------------------------------------------------------------------
use_log = st.sidebar.checkbox("Log scale (price)", False)

# 'Refresh now' clears the cache so next run will re-download new bars.
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()


# =============================================================================
# 2) DATA LOADING (CACHED)
# =============================================================================
@st.cache_data(ttl=60*15, show_spinner=False)
def load_data(sym: str, per: str, inter: str) -> pd.DataFrame:
    """
    Download OHLCV with yfinance.
    - auto_adjust=False to keep raw OHLC (important for crypto and true volume).
    - Drop empty rows just in case.
    """
    df = yf.download(sym, period=per, interval=inter, auto_adjust=False, progress=False)
    return df.dropna()

with st.spinner("Loading data…"):
    df = load_data(symbol, period, interval)

if df.empty:
    st.warning("No data returned. Try a shorter period or a different interval.")
    st.stop()

# Work on a copy so we never mutate the cached object accidentally.
df = df.copy()


# =============================================================================
# 3) INDICATORS + TRENDLINES (computed on the full download, then filtered)
# =============================================================================
def ema(s: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average."""
    return s.ewm(span=n, adjust=False).mean()

# --- Simple Moving Averages (overlay on price) --------------------------------
if show_ma20:  df["SMA20"]  = df["Close"].rolling(20).mean()
if show_ma50:  df["SMA50"]  = df["Close"].rolling(50).mean()
if show_ma200: df["SMA200"] = df["Close"].rolling(200).mean()

# --- Bollinger Bands (20, 2σ)  (overlay on price) -----------------------------
if show_bb:
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std(ddof=0)
    df["BB_mid"] = mid
    df["BB_up"]  = mid + 2*std
    df["BB_lo"]  = mid - 2*std

# --- RSI(14) panel -------------------------------------------------------------
if show_rsi:
    delta  = df["Close"].diff()
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_up = gains.ewm(alpha=1/14, adjust=False).mean()
    avg_dn = losses.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

# --- MACD(12,26,9) panel ------------------------------------------------------
if show_macd:
    ema12 = ema(df["Close"], 12)
    ema26 = ema(df["Close"], 26)
    df["MACD"]    = ema12 - ema26
    df["MACDsig"] = ema(df["MACD"], 9)
    df["MACDhist"]= df["MACD"] - df["MACDsig"]

# --- ATR(14) panel + also used in S/R level merging ---------------------------
if show_atr or detect_breakouts:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),       # range of current bar
        (df["High"] - prev_close).abs(),      # gap up from prev close
        (df["Low"]  - prev_close).abs()       # gap down from prev close
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()

# --- EMA(N) trendline (overlay on price) --------------------------------------
if show_ema_trend:
    df[f"EMA{ema_len}"] = ema(df["Close"], int(ema_len))

# Linear (OLS) trendline will be computed AFTER we select the date window,
# so it reflects only the visible range the user chose.


# =============================================================================
# 4) DATE RANGE FILTER (NO RE-DOWNLOAD)
# =============================================================================
min_date = df.index.min().date()
max_date = df.index.max().date()

st.sidebar.subheader("Date range (filter)")
start_date, end_date = st.sidebar.date_input(
    "Start / End",
    value=(max(min_date, max_date - timedelta(days=180)), max_date),
    min_value=min_date, max_value=max_date
)

# Only keep rows within the chosen start/end (string slicing on DatetimeIndex).
if isinstance(start_date, date) and isinstance(end_date, date) and start_date <= end_date:
    df = df.loc[str(start_date):str(end_date)]

if df.empty:
    st.warning("No data after filtering. Expand the date range.")
    st.stop()

# Compute linear trendline on the filtered window (if turned on).
if show_lin_trend and len(df) >= 2:
    x = np.arange(len(df), dtype=float)
    m, b = np.polyfit(x, df["Close"].values.astype(float), 1)
    df["LIN_TREND"] = m * x + b


# =============================================================================
# 5) SUPPORT/RESISTANCE + BREAKOUTS (robust; 1-D arrays + integer indexing)
# =============================================================================
sr_levels, breakouts = [], []

if sr_window and len(df) > sr_window + 5:
    # Rolling extremes: where High equals rolling max, or Low equals rolling min
    roll_high = df["High"].rolling(sr_window, min_periods=sr_window).max()
    roll_low  = df["Low"].rolling(sr_window,  min_periods=sr_window).min()

    # Convert to 1-D arrays to avoid boolean-shape issues, then build masks.
    high_vals = np.asarray(df["High"]).ravel()
    low_vals  = np.asarray(df["Low"]).ravel()
    rh_vals   = np.asarray(roll_high).ravel()
    rl_vals   = np.asarray(roll_low).ravel()

    valid_res = ~np.isnan(high_vals) & ~np.isnan(rh_vals)
    valid_sup = ~np.isnan(low_vals)  & ~np.isnan(rl_vals)

    mask_res = valid_res & (high_vals >= rh_vals)  # candidate swing highs (R)
    mask_sup = valid_sup & (low_vals  <= rl_vals)  # candidate swing lows (S)

    # Integer positions keep index alignment correct.
    idx_res = np.flatnonzero(mask_res)
    idx_sup = np.flatnonzero(mask_sup)

    # Series of candidate levels with the correct datetime index.
    piv_res = pd.Series(high_vals[idx_res], index=df.index.take(idx_res), name="High")
    piv_sup = pd.Series(low_vals[idx_sup],  index=df.index.take(idx_sup),  name="Low")

    # Merge tolerance: prefer ATR; fallback to 0.2% of median close if ATR not available.
    if "ATR14" in df.columns and not df["ATR14"].dropna().empty:
        tol = float(df["ATR14"].median()) * float(sr_merge_atr_mult)
    else:
        tol = float(df["Close"].median()) * 0.002

    def merge_levels(series: pd.Series, kind: str):
        """
        Merge nearby S/R points into cleaner levels.
        - Input: Series of prices with timestamps where a swing occurred.
        - Output: list of dicts: {"price": float, "time": Timestamp, "kind": "S"|"R"}
        """
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
                # If very close to previous level, merge by averaging.
                levels[-1]["price"] = (levels[-1]["price"] + lvl) / 2.0
                levels[-1]["time"]  = ts
        return levels

    # Final set of S/R levels (supports + resistances).
    sr_levels = merge_levels(piv_sup, "S") + merge_levels(piv_res, "R")

    # --- Breakout detection (scalar-safe):
    # If the latest close crossed above the last R (breakout ↑) or below the last S (breakdown ↓).
    if detect_breakouts and sr_levels and len(df["Close"]) >= 2:
        last_S = max([l for l in sr_levels if l["kind"] == "S"], key=lambda x: x["time"], default=None)
        last_R = max([l for l in sr_levels if l["kind"] == "R"], key=lambda x: x["time"], default=None)

        prev_close = float(df["Close"].iloc[-2])
        last_close = float(df["Close"].iloc[-1])

        if last_R is not None:
            r_price = float(last_R["price"])
            if (prev_close <= r_price) and (last_close > r_price):
                breakouts.append({"when": df.index[-1], "type": "Breakout ↑", "level": r_price})

        if last_S is not None:
            s_price = float(last_S["price"])
            if (prev_close >= s_price) and (last_close < s_price):
                breakouts.append({"when": df.index[-1], "type": "Breakdown ↓", "level": s_price})

        # --- Pullback/return: after breakout, price revisits the level within 0.5×ATR.
        if "ATR14" in df.columns and not df["ATR14"].dropna().empty and breakouts:
            atr  = float(df["ATR14"].iloc[-1])
            band = 0.5 * atr
            for b in breakouts:
                lvl = float(b["level"])
                if b["type"] == "Breakout ↑" and (lvl <= last_close <= lvl + band):
                    b["return"] = "Pullback to R→S"
                if b["type"] == "Breakdown ↓" and (lvl - band <= last_close <= lvl):
                    b["return"] = "Pullback to S→R"


# =============================================================================
# 6) PLOTTING  (balanced row heights for readability)
# =============================================================================
# We build row heights dynamically based on which panels are visible.
# The weights below aim for a readable layout that still keeps panels aligned.

heights = []            # Collect panel height weights here (must sum to ~1; Plotly will normalize)
panel_rows = []         # We'll record the row index of each panel in order of creation.

# --- Price panel first (largest) ----------------------------------------------
heights.append(0.45)    # ~45% of the vertical space
panel_rows.append("price")

# --- Volume panel (optional) --------------------------------------------------
if show_volume:
    heights.append(0.15)    # ~15%
    panel_rows.append("volume")

# --- RSI panel (optional) -----------------------------------------------------
if show_rsi:
    heights.append(0.20)    # ~20%
    panel_rows.append("rsi")

# --- MACD panel (optional) ----------------------------------------------------
if show_macd:
    heights.append(0.20)    # ~20%
    panel_rows.append("macd")

# --- ATR panel (optional) -----------------------------------------------------
if show_atr:
    heights.append(0.20)    # ~20%
    panel_rows.append("atr")

rows = len(heights)

# Create subplots with shared X axis so time lines up across panels.
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    row_heights=heights, vertical_spacing=0.03)

# Helper: get the 1-based row index for a named panel.
def row_of(name: str) -> int:
    return panel_rows.index(name) + 1

# --- PRICE PANEL --------------------------------------------------------------
if price_mode == "Candles":
    fig.add_trace(
        go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                       low=df["Low"], close=df["Close"], name="Price"),
        row=row_of("price"), col=1
    )
else:
    # line chart of the Close series
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Price"),
        row=row_of("price"), col=1
    )

# Trendlines on price
if show_ema_trend and f"EMA{ema_len}" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA{ema_len}"], name=f"EMA{ema_len}", mode="lines"),
                  row=row_of("price"), col=1)
if show_lin_trend and "LIN_TREND" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["LIN_TREND"], name="Linear trend",
                             mode="lines", line=dict(dash="dash")),
                  row=row_of("price"), col=1)

# SMAs (overlay)
for col, dash in [("SMA20","solid"), ("SMA50","dot"), ("SMA200","dash")]:
    if col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                 mode="lines", line=dict(dash=dash)),
                      row=row_of("price"), col=1)

# Bollinger Bands (overlay)
if show_bb and {"BB_up","BB_lo"} <= set(df.columns):
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB up",
                             mode="lines", opacity=0.5),
                  row=row_of("price"), col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lo"], name="BB lo",
                             mode="lines", opacity=0.5),
                  row=row_of("price"), col=1)

# Draw S/R levels and breakout markers on the price panel.
for lvl in sr_levels:
    fig.add_hline(y=lvl["price"], line_width=1, opacity=0.3,
                  line_dash="dot" if lvl["kind"] == "S" else "dash",
                  row=row_of("price"), col=1)
for b in breakouts:
    label = b["type"] + (f" ({b.get('return')})" if b.get("return") else "")
    fig.add_trace(
        go.Scatter(x=[b["when"]], y=[b["level"]], mode="markers+text",
                   text=[label], textposition="top center", name=b["type"]),
        row=row_of("price"), col=1
    )

# --- VOLUME PANEL -------------------------------------------------------------
if show_volume:
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.6),
                  row=row_of("volume"), col=1)
    fig.update_yaxes(title_text="Volume", row=row_of("volume"), col=1)

# --- RSI PANEL ----------------------------------------------------------------
if show_rsi:
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", mode="lines"),
                  row=row_of("rsi"), col=1)
    # Neutral zone shading: 30-70
    fig.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2,
                  line_width=0, row=row_of("rsi"), col=1)

# --- MACD PANEL ---------------------------------------------------------------
if show_macd:
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"),
                  row=row_of("macd"), col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACDsig"], name="Signal", mode="lines"),
                  row=row_of("macd"), col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACDhist"], name="Hist", opacity=0.5),
                  row=row_of("macd"), col=1)

# --- ATR PANEL ----------------------------------------------------------------
if show_atr:
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR14"], name="ATR14", mode="lines"),
                  row=row_of("atr"), col=1)

# --- Global layout polishing --------------------------------------------------
fig.update_layout(
    title=f"{symbol} — {interval} ({period})",
    xaxis_rangeslider_visible=False,        # hide the small, default rangeslider
    hovermode="x unified",                  # one vertical hover across panels
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
# Price axis can be linear or log (others stay linear).
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True, type=("log" if use_log else "linear"), row=row_of("price"), col=1)

# Render the figure
st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 7) STATS, TABLE, AND CSV EXPORT
# =============================================================================
c1, c2, c3 = st.columns(3)
c1.metric("Last Close", f"{float(df['Close'].iloc[-1]):,.2f}")
c2.metric("Bars", f"{len(df):,}")
c3.metric("Start → End", f"{df.index[0].date()} → {df.index[-1].date()}")

with st.expander("Show data (last 1000 rows)"):
    st.dataframe(df.tail(1000), use_container_width=True)

st.download_button("Download CSV",
                   df.to_csv().encode("utf-8"),
                   file_name=f"{symbol}_{interval}_{period}.csv")
