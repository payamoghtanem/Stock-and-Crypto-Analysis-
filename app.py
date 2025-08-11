# =============================================================================
# Streamlit + Plotly Technical Charting (Yahoo Finance)
# Bigger, clearer charts using TABS + optional stacked view.
#
# Tabs:
#  1) Price & Volume (candles/line, SMAs, BB, trendlines, S/R, breakouts)
#  2) Oscillators (RSI, MACD)
#  3) Volatility (ATR, Bollinger band width)
#  4) Trends (EMA, Linear OLS, HP filter)
#  5) Rolling OLS (slope over a moving window)
#  6) STL Decomposition (trend/seasonal/residual)
#
# Notes:
# - All computations are dynamic after date filtering.
# - S/R detection is heuristic (rolling extremes + ATR merge).
# - Educational tool, not trading advice.
# =============================================================================

from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Crypto/Stocks Trend — Tabs", layout="wide")
st.title("Crypto/Stocks Trend — Interactive (Plotly + Streamlit)")

# -----------------------------------------------------------------------------
# Sidebar — Data & Layout
# -----------------------------------------------------------------------------
st.sidebar.header("Data")
symbol   = st.sidebar.text_input("Symbol", "BTC-USD",
                                 help="Any Yahoo symbol, e.g., BTC-USD, ETH-USD, AAPL, ^GSPC")
interval = st.sidebar.selectbox("Interval", ["4h", "1h", "1d"], index=0)
period   = st.sidebar.selectbox("Download window", ["30d", "90d", "365d", "730d"], index=3)

st.sidebar.header("Layout")
layout_mode = st.sidebar.radio("Chart layout", ["Tabbed charts", "Single stacked figure"], index=0,
                               help="Tabbed = largest, clearest. Stacked = all in one.")
price_mode  = st.sidebar.selectbox("Price display", ["Candles", "Line (Close)"], index=0)
use_log     = st.sidebar.checkbox("Log scale (price)", False)
show_volume = st.sidebar.checkbox("Show volume", True)

st.sidebar.header("Trend lines (on price)")
ema_len = st.sidebar.number_input("EMA length", 5, 400, 50, step=5)
show_ema_trend = st.sidebar.checkbox("Show EMA", False)
show_lin_trend = st.sidebar.checkbox("Show Linear (OLS) across visible window", False)

st.sidebar.header("Indicators")
show_ma20  = st.sidebar.checkbox("SMA 20", True)
show_ma50  = st.sidebar.checkbox("SMA 50", True)
show_ma200 = st.sidebar.checkbox("SMA 200", False)
show_bb    = st.sidebar.checkbox("Bollinger Bands (20,2)", True)
show_rsi   = st.sidebar.checkbox("RSI (14)", True)
show_macd  = st.sidebar.checkbox("MACD (12,26,9)", True)
show_atr   = st.sidebar.checkbox("ATR (14)", True)

st.sidebar.header("Support / Resistance")
sr_window = st.sidebar.slider("Swing window (bars)", 10, 300, 50,
                              help="Lookback used for rolling highs/lows.")
sr_merge_atr_mult = st.sidebar.slider("Merge tolerance (×ATR)", 0.1, 2.0, 0.5, 0.1,
                                      help="Nearby levels are merged if within this ATR multiple.")
detect_breakouts = st.sidebar.checkbox("Detect breakouts/returns", True)

st.sidebar.header("Advanced trends")
hp_lambda = st.sidebar.number_input("HP filter λ", 100.0, 200000.0, 1600.0, step=100.0,
                                    help="Higher λ → smoother trend (commonly 1600 for hourly-ish data).")
rols_win  = st.sidebar.slider("Rolling OLS window (bars)", 20, 500, 120,
                              help="Window size for slope computed over moving regression.")
stl_period = st.sidebar.slider("STL seasonal period (bars)", 10, 200, 42,
                               help="e.g., 42 bars ≈ 1 week for 4h candles.")

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()

# -----------------------------------------------------------------------------
# Data loading (cached)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Indicators/trend helpers (computed BEFORE date filter; re-evaluated after)
# -----------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

# SMAs
if show_ma20:  df["SMA20"]  = df["Close"].rolling(20).mean()
if show_ma50:  df["SMA50"]  = df["Close"].rolling(50).mean()
if show_ma200: df["SMA200"] = df["Close"].rolling(200).mean()

# Bollinger Bands (20,2) + Band Width (%)
if show_bb:
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std(ddof=0)
    df["BB_mid"] = mid
    df["BB_up"]  = mid + 2*std
    df["BB_lo"]  = mid - 2*std
    bw = (df["BB_up"] - df["BB_lo"]) / df["BB_mid"]
    df["BB_width_pct"] = (bw.replace([np.inf, -np.inf], np.nan) * 100)

# RSI(14)
if show_rsi:
    delta  = df["Close"].diff()
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_up = gains.ewm(alpha=1/14, adjust=False).mean()
    avg_dn = losses.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

# MACD(12,26,9)
if show_macd:
    ema12 = ema(df["Close"], 12)
    ema26 = ema(df["Close"], 26)
    df["MACD"]    = ema12 - ema26
    df["MACDsig"] = ema(df["MACD"], 9)
    df["MACDhist"]= df["MACD"] - df["MACDsig"]

# ATR(14)
if show_atr or detect_breakouts:
    pc = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()

# EMA trend
if show_ema_trend:
    df[f"EMA{ema_len}"] = ema(df["Close"], int(ema_len))

# -----------------------------------------------------------------------------
# Date filter (no re-download)
# -----------------------------------------------------------------------------
min_date = df.index.min().date()
max_date = df.index.max().date()

st.sidebar.header("Date range (filter)")
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

# Linear trendline across the *visible* window
if show_lin_trend and len(df) >= 2:
    x = np.arange(len(df), dtype=float)
    m, b = np.polyfit(x, df["Close"].astype(float), 1)
    df["LIN_TREND"] = m * x + b

# HP filter (trend & cycle) on visible window
close_float = df["Close"].astype(float)
hp_trend, hp_cycle = hpfilter(close_float, lamb=hp_lambda)
df["HP_trend"] = hp_trend
df["HP_cycle"] = hp_cycle

# Rolling OLS: slope per window (simple lambda → fine for a few thousand bars)
def rolling_slope(y: pd.Series, win: int) -> pd.Series:
    if len(y) < win:
        return pd.Series(index=y.index, dtype=float)
    return y.rolling(win).apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], raw=False)

df["ROLL_SLOPE"] = rolling_slope(close_float, rols_win)

# STL decomposition (trend/seasonal/resid) — needs a period guess
stl_components = None
try:
    stl = STL(close_float, period=int(stl_period), robust=True).fit()
    stl_components = {
        "trend": stl.trend,
        "seasonal": stl.seasonal,
        "resid": stl.resid,
    }
except Exception:
    # If STL fails (e.g., too few points), we simply skip plotting it.
    stl_components = None

# -----------------------------------------------------------------------------
# Support/Resistance + Breakouts  (robust, 1-D masks)
# -----------------------------------------------------------------------------
sr_levels, breakouts = [], []
if sr_window and len(df) > sr_window + 5:
    roll_high = df["High"].rolling(sr_window, min_periods=sr_window).max()
    roll_low  = df["Low"].rolling(sr_window,  min_periods=sr_window).min()

    hv = np.asarray(df["High"]).ravel()
    lv = np.asarray(df["Low"]).ravel()
    rh = np.asarray(roll_high).ravel()
    rl = np.asarray(roll_low).ravel()

    mask_res = (~np.isnan(hv) & ~np.isnan(rh) & (hv >= rh))
    mask_sup = (~np.isnan(lv) & ~np.isnan(rl) & (lv <= rl))

    idx_res = np.flatnonzero(mask_res)
    idx_sup = np.flatnonzero(mask_sup)

    piv_res = pd.Series(hv[idx_res], index=df.index.take(idx_res), name="High")
    piv_sup = pd.Series(lv[idx_sup],  index=df.index.take(idx_sup),  name="Low")

    if "ATR14" in df.columns and not df["ATR14"].dropna().empty:
        tol = float(df["ATR14"].median()) * float(sr_merge_atr_mult)
    else:
        tol = float(df["Close"].median()) * 0.002

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
                levels[-1]["price"] = (levels[-1]["price"] + lvl) / 2.0
                levels[-1]["time"]  = ts
        return levels

    sr_levels = merge_levels(piv_sup, "S") + merge_levels(piv_res, "R")

    if detect_breakouts and len(df) >= 2:
        last_S = max([l for l in sr_levels if l["kind"] == "S"], key=lambda x: x["time"], default=None)
        last_R = max([l for l in sr_levels if l["kind"] == "R"], key=lambda x: x["time"], default=None)
        prev_close = float(df["Close"].iloc[-2])
        last_close = float(df["Close"].iloc[-1])

        if last_R is not None:
            r = float(last_R["price"])
            if (prev_close <= r) and (last_close > r):
                breakouts.append({"when": df.index[-1], "type": "Breakout ↑", "level": r})
        if last_S is not None:
            s_ = float(last_S["price"])
            if (prev_close >= s_) and (last_close < s_):
                breakouts.append({"when": df.index[-1], "type": "Breakdown ↓", "level": s_})

        if "ATR14" in df.columns and not df["ATR14"].dropna().empty and breakouts:
            atr  = float(df["ATR14"].iloc[-1])
            band = 0.5 * atr
            for b in breakouts:
                lvl = float(b["level"])
                if b["type"] == "Breakout ↑" and (lvl <= last_close <= lvl + band):
                    b["return"] = "Pullback to R→S"
                if b["type"] == "Breakdown ↓" and (lvl - band <= last_close <= lvl):
                    b["return"] = "Pullback to S→R"

# -----------------------------------------------------------------------------
# Plotting helpers (each tab builds its own large figure)
# -----------------------------------------------------------------------------
def add_price_traces(fig, row=1, title_suffix=""):
    """Add price + overlays + S/R + breakouts to a figure."""
    if price_mode == "Candles":
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"], name="Price"),
                      row=row, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Price"),
                      row=row, col=1)

    # Trendlines
    if show_ema_trend and f"EMA{ema_len}" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA{ema_len}"], name=f"EMA{ema_len}", mode="lines"),
                      row=row, col=1)
    if show_lin_trend and "LIN_TREND" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["LIN_TREND"], name="Linear trend",
                                 mode="lines", line=dict(dash="dash")), row=row, col=1)

    # SMAs
    for col, dash in [("SMA20","solid"), ("SMA50","dot"), ("SMA200","dash")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                     mode="lines", line=dict(dash=dash)),
                          row=row, col=1)

    # Bollinger bands
    if show_bb and {"BB_up","BB_lo"} <= set(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB up", mode="lines", opacity=0.5),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lo"], name="BB lo", mode="lines", opacity=0.5),
                      row=row, col=1)

    # S/R lines + breakouts
    for lvl in sr_levels:
        fig.add_hline(y=lvl["price"], line_width=1, opacity=0.3,
                      line_dash="dot" if lvl["kind"] == "S" else "dash", row=row, col=1)
    for b in breakouts:
        label = b["type"] + (f" ({b.get('return')})" if b.get("return") else "")
        fig.add_trace(go.Scatter(x=[b["when"]], y=[b["level"]], mode="markers+text",
                                 text=[label], textposition="top center", name=b["type"]),
                      row=row, col=1)

def style(fig, title, height=650):
    """Common layout polish for all figures."""
    fig.update_layout(
        title=title,
        height=height,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, type=("log" if use_log else "linear"))
    return fig

# -----------------------------------------------------------------------------
# Render — either as TABS (recommended) or as a single stacked figure
# -----------------------------------------------------------------------------
if layout_mode == "Tabbed charts":
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Price & Volume",
        "Oscillators",
        "Volatility",
        "Trends (EMA/Linear/HP)",
        "Rolling OLS",
        "STL Decomposition",
    ])

    with tab1:
        fig = make_subplots(rows=2 if show_volume else 1, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25] if show_volume else [1.0])
        add_price_traces(fig, row=1)
        if show_volume:
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.6),
                          row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        style(fig, f"{symbol} — {interval} ({period})", height=700 if show_volume else 600)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tip: switch Candles/Line from the sidebar; toggle overlays to declutter.")

    with tab2:
        rows = int(show_rsi) + int(show_macd)
        if rows == 0:
            st.info("Turn on RSI or MACD from the sidebar to see this tab.")
        else:
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=[0.5]*rows)
            r = 1
            if show_rsi:
                fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", mode="lines"), row=r, col=1)
                fig.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2,
                              line_width=0, row=r, col=1)
                r += 1
            if show_macd:
                fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"), row=r, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["MACDsig"], name="Signal", mode="lines"), row=r, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df["MACDhist"], name="Hist", opacity=0.5), row=r, col=1)
            style(fig, "Oscillators (RSI, MACD)", height=650)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("RSI 30–70 band shaded. MACD shows line, signal, and histogram.")

    with tab3:
        rows = 1 + int(show_bb and "BB_width_pct" in df)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            row_heights=[0.65] + ([0.35] if rows == 2 else []))
        if show_atr:
            fig.add_trace(go.Scatter(x=df.index, y=df["ATR14"], name="ATR14", mode="lines"), row=1, col=1)
        if rows == 2:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_width_pct"], name="BB width (%)", mode="lines"),
                          row=2, col=1)
        style(fig, "Volatility (ATR + Bollinger width)", height=650)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ATR tracks absolute volatility. Bollinger band width (%) tracks relative volatility.")

    with tab4:
        fig = make_subplots(rows=1, cols=1)
        add_price_traces(fig, row=1)
        if "HP_trend" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["HP_trend"], name=f"HP trend (λ={int(hp_lambda)})",
                                     mode="lines", line=dict(dash="dot")), row=1, col=1)
        style(fig, "Trendlines (EMA / Linear OLS / HP filter)", height=650)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("EMA smooths price; Linear OLS fits a straight line; HP filter separates trend & cycle.")

    with tab5:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35])
        # upper: price (for reference)
        add_price_traces(fig, row=1)
        # lower: rolling OLS slope through time
        fig.add_trace(go.Scatter(x=df.index, y=df["ROLL_SLOPE"], name=f"Slope (win={rols_win})", mode="lines"),
                      row=2, col=1)
        style(fig, "Rolling OLS slope (per bar)", height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Positive slope = uptrend over the last window; negative = downtrend.")

    with tab6:
        if stl_components is None:
            st.info("STL needs enough data and a valid period. Try expanding the date range or adjust 'STL seasonal period'.")
        else:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.2, 0.2, 0.2])
            # original close
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines"), row=1, col=1)
            # STL components
            fig.add_trace(go.Scatter(x=df.index, y=stl_components["trend"], name="Trend", mode="lines"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=stl_components["seasonal"], name="Seasonal", mode="lines"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=stl_components["resid"], name="Residual", mode="lines"), row=4, col=1)
            style(fig, f"STL Decomposition (period={stl_period})", height=800)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("STL splits the series into Trend + Seasonal + Residual. Period = seasonal cycle length in bars.")

else:
    # --- Single stacked figure (for users who like everything aligned in one view)
    # Heights tuned to be readable (you can edit if you prefer different proportions)
    heights = [0.45]
    if show_volume: heights.append(0.15)
    if show_rsi:    heights.append(0.15)
    if show_macd:   heights.append(0.15)
    if show_atr:    heights.append(0.10)

    rows = len(heights)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03)

    r = 1
    add_price_traces(fig, row=r)
    if show_volume:
        r += 1
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.6), row=r, col=1)
        fig.update_yaxes(title_text="Volume", row=r, col=1)
    if show_rsi:
        r += 1
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", mode="lines"), row=r, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2, line_width=0, row=r, col=1)
    if show_macd:
        r += 1
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACDsig"], name="Signal", mode="lines"), row=r, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["MACDhist"], name="Hist", opacity=0.5), row=r, col=1)
    if show_atr:
        r += 1
        fig.add_trace(go.Scatter(x=df.index, y=df["ATR14"], name="ATR14", mode="lines"), row=r, col=1)

    style(fig, f"{symbol} — {interval} ({period})", height=900)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Footer — quick stats, table, CSV
# -----------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Last Close", f"{float(df['Close'].iloc[-1]):,.2f}")
c2.metric("Bars", f"{len(df):,}")
c3.metric("Start → End", f"{df.index[0].date()} → {df.index[-1].date()}")

with st.expander("Show data (last 1000 rows)"):
    st.dataframe(df.tail(1000), use_container_width=True)

st.download_button("Download CSV",
                   df.to_csv().encode("utf-8"),
                   file_name=f"{symbol}_{interval}_{period}.csv")
