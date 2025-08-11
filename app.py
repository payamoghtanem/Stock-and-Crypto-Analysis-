# =============================================================================
# Streamlit + Plotly Technical Charting (Yahoo Finance)
# Tabs + signal engine + last-value labels + zero baselines + help text
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

# ----------------------------- Page ------------------------------------------
st.set_page_config(page_title="Crypto/Stocks Trend — Tabs", layout="wide")
st.title("Crypto/Stocks Trend — Interactive (Plotly + Streamlit)")

# ----------------------------- Sidebar ---------------------------------------
st.sidebar.header("Data")
symbol   = st.sidebar.text_input("Symbol", "BTC-USD", help="Yahoo symbol: e.g., BTC-USD, ETH-USD, AAPL, ^GSPC")
interval = st.sidebar.selectbox("Interval", ["4h", "1h", "1d"], index=0)
period   = st.sidebar.selectbox("Download window", ["30d", "90d", "365d", "730d"], index=3)

st.sidebar.header("Layout")
layout_mode = st.sidebar.radio("Chart layout", ["Tabbed charts", "Single stacked figure"], index=0)
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
sr_window = st.sidebar.slider("Swing window (bars)", 10, 300, 50)
sr_merge_atr_mult = st.sidebar.slider("Merge tolerance (×ATR)", 0.1, 2.0, 0.5, 0.1)
detect_breakouts = st.sidebar.checkbox("Detect breakouts/returns", True)

st.sidebar.header("Advanced trends")
hp_lambda = st.sidebar.number_input("HP filter λ", 100.0, 200000.0, 1600.0, step=100.0,
                                    help="Higher λ → smoother trend (e.g., 1600 for 4h/daily).")
rols_win  = st.sidebar.slider("Rolling OLS window (bars)", 20, 500, 120)
stl_period = st.sidebar.slider("STL seasonal period (bars)", 10, 200, 42)

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()

# ----------------------------- Data (cached) ----------------------------------
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

# ----------------------------- Indicators ------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

# SMAs
if show_ma20:  df["SMA20"]  = df["Close"].rolling(20).mean()
if show_ma50:  df["SMA50"]  = df["Close"].rolling(50).mean()
if show_ma200: df["SMA200"] = df["Close"].rolling(200).mean()

# Bollinger bands + width
if show_bb:
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std(ddof=0)
    df["BB_mid"] = mid
    df["BB_up"]  = mid + 2*std
    df["BB_lo"]  = mid - 2*std
    width_pct = (df["BB_up"] - df["BB_lo"]) / df["BB_mid"] * 100
    df["BB_width_pct"] = width_pct.replace([np.inf, -np.inf], np.nan)

# RSI
if show_rsi:
    d   = df["Close"].diff()
    up  = d.clip(lower=0)
    dn  = (-d).clip(lower=0)
    avg_up = up.ewm(alpha=1/14, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

# MACD
if show_macd:
    ema12 = ema(df["Close"], 12)
    ema26 = ema(df["Close"], 26)
    df["MACD"]    = ema12 - ema26
    df["MACDsig"] = ema(df["MACD"], 9)
    df["MACDhist"]= df["MACD"] - df["MACDsig"]

# ATR (also used in S/R merging)
if show_atr or detect_breakouts:
    pc = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]).abs(),
                    (df["High"]-pc).abs(),
                    (df["Low"] -pc).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()

# EMA trend
if show_ema_trend:
    df[f"EMA{ema_len}"] = ema(df["Close"], int(ema_len))

# ----------------------------- Date filter -----------------------------------
min_date = df.index.min().date(); max_date = df.index.max().date()
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

# Linear trend across visible window
if show_lin_trend and len(df) >= 2:
    x = np.arange(len(df), dtype=float)
    m, b = np.polyfit(x, df["Close"].astype(float), 1)
    df["LIN_TREND"] = m * x + b

# HP filter & Rolling OLS & STL
close_f = df["Close"].astype(float)
hp_trend, hp_cycle = hpfilter(close_f, lamb=hp_lambda)
df["HP_trend"] = hp_trend
df["HP_cycle"] = hp_cycle

def rolling_slope(y: pd.Series, win: int) -> pd.Series:
    if len(y) < win: return pd.Series(index=y.index, dtype=float)
    return y.rolling(win).apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], raw=False)
df["ROLL_SLOPE"] = rolling_slope(close_f, rols_win)

try:
    _stl_check = STL(close_f, period=int(stl_period), robust=True).fit()
    STL_AVAILABLE = True
except Exception:
    STL_AVAILABLE = False

# ----------------------------- Support/Resistance + Breakouts ----------------
sr_levels, breakouts = [], []
if sr_window and len(df) > sr_window + 5:
    rh = df["High"].rolling(sr_window, min_periods=sr_window).max()
    rl = df["Low"].rolling(sr_window,  min_periods=sr_window).min()

    hv = np.asarray(df["High"]).ravel()
    lv = np.asarray(df["Low"]).ravel()
    rhv = np.asarray(rh).ravel()
    rlv = np.asarray(rl).ravel()

    idx_res = np.flatnonzero((~np.isnan(hv)) & (~np.isnan(rhv)) & (hv >= rhv))
    idx_sup = np.flatnonzero((~np.isnan(lv)) & (~np.isnan(rlv)) & (lv <= rlv))

    piv_res = pd.Series(hv[idx_res], index=df.index.take(idx_res), name="High")
    piv_sup = pd.Series(lv[idx_sup],  index=df.index.take(idx_sup),  name="Low")

    tol = (float(df["ATR14"].median()) * float(sr_merge_atr_mult)
           if "ATR14" in df and not df["ATR14"].dropna().empty
           else float(df["Close"].median()) * 0.002)

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
                levels[-1]["price"] = (levels[-1]["price"] + lvl)/2.0
                levels[-1]["time"]  = ts
        return levels

    sr_levels = merge_levels(piv_sup, "S") + merge_levels(piv_res, "R")

    if detect_breakouts and len(df) >= 2:
        last_S = max([l for l in sr_levels if l["kind"] == "S"], key=lambda x: x["time"], default=None)
        last_R = max([l for l in sr_levels if l["kind"] == "R"], key=lambda x: x["time"], default=None)
        prev_c, last_c = float(df["Close"].iloc[-2]), float(df["Close"].iloc[-1])

        if last_R is not None:
            r = float(last_R["price"])
            if (prev_c <= r) and (last_c > r):
                breakouts.append({"when": df.index[-1], "type": "Breakout ↑", "level": r})
        if last_S is not None:
            s_ = float(last_S["price"])
            if (prev_c >= s_) and (last_c < s_):
                breakouts.append({"when": df.index[-1], "type": "Breakdown ↓", "level": s_})

        if "ATR14" in df and not df["ATR14"].dropna().empty and breakouts:
            atr  = float(df["ATR14"].iloc[-1]); band = 0.5*atr
            for b in breakouts:
                lvl = float(b["level"])
                if b["type"] == "Breakout ↑" and (lvl <= last_c <= lvl + band):
                    b["return"] = "Pullback to R→S"
                if b["type"] == "Breakdown ↓" and (lvl - band <= last_c <= lvl):
                    b["return"] = "Pullback to S→R"

# ----------------------------- Signal Engine ---------------------------------
def last_cross(spread: pd.Series) -> int:
    if len(spread) < 2: return 0
    a, b = spread.iloc[-2], spread.iloc[-1]
    if (a <= 0) and (b > 0):  return +1
    if (a >= 0) and (b < 0):  return -1
    return 0

def compute_signal(df: pd.DataFrame) -> dict:
    sig = {}
    sig["close"]  = float(df["Close"].iloc[-1])
    sig["sma50"]  = float(df.get("SMA50", df["Close"]).iloc[-1])
    sig["sma200"] = float(df.get("SMA200", df["Close"]).iloc[-1])
    sig["rsi"]    = float(df.get("RSI14", pd.Series([np.nan])).iloc[-1])
    if {"MACD","MACDsig"}.issubset(df.columns):
        spread = df["MACD"] - df["MACDsig"]
        sig["macd"] = float(df["MACD"].iloc[-1])
        sig["macd_cross"] = last_cross(spread)
    else:
        sig["macd"] = np.nan; sig["macd_cross"] = 0

    buy = (
        (sig["close"] > sig["sma50"] > sig["sma200"]) and
        ("SMA50" not in df or df["SMA50"].diff().iloc[-1] > 0) and
        (sig["macd"] > 0) and (sig["macd_cross"] == +1) and
        (np.isnan(sig["rsi"]) or 45 <= sig["rsi"] <= 70)
    )
    sell = ((sig["close"] < sig["sma50"] < sig["sma200"]) or
            (sig["macd"] < 0 and sig["macd_cross"] == -1))
    label, emoji = ("Hold", "⏸️")
    if buy:  label, emoji = ("Buy (trend-following)", "🟢")
    elif sell: label, emoji = ("Sell / Reduce", "🔴")
    sig["label"], sig["emoji"] = label, emoji
    return sig

# ----------------------------- Plot helpers ----------------------------------
def add_last_label(fig, x, y, name, row=1, col=1):
    """
    Robust last-value marker + text.
    - Accepts list/ndarray/Series; coerces to 1-D float.
    - Skips silently if last value is not finite (NaN/inf) or arrays are empty.
    """
    try:
        y_arr = np.asarray(y, dtype=float).ravel()
        if y_arr.size == 0 or not np.isfinite(y_arr[-1]):  # nothing to label
            return
        x_last = x[-1]
        y_last = float(y_arr[-1])
    except Exception:
        return
    fig.add_trace(go.Scatter(
        x=[x_last], y=[y_last],
        mode="markers+text",
        marker=dict(size=9),
        text=[f"{y_last:,.2f}"],
        textposition="bottom right",
        name=f"{name} last"
    ), row=row, col=1)

def add_price_traces(fig, row=1):
    """Price + overlays + S/R + breakouts (and last labels)."""
    if price_mode == "Candles":
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"], name="Price"),
                      row=row, col=1)
        add_last_label(fig, df.index, df["Close"].values, "Price", row=row)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Price"),
                      row=row, col=1)
        add_last_label(fig, df.index, df["Close"].values, "Price", row=row)

    # Trendlines
    if show_ema_trend and f"EMA{ema_len}" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA{ema_len}"], name=f"EMA{ema_len}", mode="lines"), row=row, col=1)
        add_last_label(fig, df.index, df[f"EMA{ema_len}"].values, f"EMA{ema_len}", row=row)
    if show_lin_trend and "LIN_TREND" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["LIN_TREND"], name="Linear trend",
                                 mode="lines", line=dict(dash="dash")), row=row, col=1)
        add_last_label(fig, df.index, df["LIN_TREND"].values, "Linear", row=row)

    # SMAs
    for colname, dash in [("SMA20","solid"), ("SMA50","dot"), ("SMA200","dash")]:
        if colname in df:
            fig.add_trace(go.Scatter(x=df.index, y=df[colname], name=colname,
                                     mode="lines", line=dict(dash=dash)), row=row, col=1)
            add_last_label(fig, df.index, df[colname].values, colname, row=row)

    # BB
    if show_bb and {"BB_up","BB_lo"} <= set(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB up", mode="lines", opacity=0.5), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lo"], name="BB lo", mode="lines", opacity=0.5), row=row, col=1)

    # S/R + breakouts
    for lvl in sr_levels:
        fig.add_hline(y=lvl["price"], line_width=1, opacity=0.3,
                      line_dash="dot" if lvl["kind"] == "S" else "dash", row=row, col=1)
    for b in breakouts:
        label = b["type"] + (f" ({b.get('return')})" if b.get("return") else "")
        fig.add_trace(go.Scatter(x=[b["when"]], y=[b["level"]], mode="markers+text",
                                 text=[label], textposition="top center", name=b["type"]),
                      row=row, col=1)

def style(fig, title, height=650):
    fig.update_layout(
        title=title, height=height, xaxis_rangeslider_visible=False,
        hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, type=("log" if use_log else "linear"))
    return fig

# ----------------------------- Render ----------------------------------------
if layout_mode == "Tabbed charts":
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Price & Volume", "Oscillators", "Volatility",
        "Trends (EMA/Linear/HP)", "Rolling OLS", "STL Decomposition"
    ])

    # --------------- Tab 1: Price & Volume -----------------------------------
    with tab1:
        st.info("Why this chart matters:\n"
                "- Price trend and key moving averages (20/50/200).\n"
                "- Bollinger Bands for range and squeezes.\n"
                "- Support/Resistance with breakout and pullback tags.\n"
                "- Volume confirms the strength of moves.")
        sig = compute_signal(df)
        st.markdown(f"**Signal:** {sig['emoji']} **{sig['label']}**  "
                    f"| Close `{sig['close']:.2f}` | SMA50 `{sig['sma50']:.2f}` | "
                    f"SMA200 `{sig['sma200']:.2f}` | MACD `{sig['macd']:.3f}` | RSI `{sig['rsi']:.1f}`")

        rows = 2 if show_volume else 1
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25] if show_volume else [1.0])
        add_price_traces(fig, row=1)
        if show_volume:
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.6), row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        style(fig, f"{symbol} — {interval} ({period})", height=700 if show_volume else 620)
        st.plotly_chart(fig, use_container_width=True)

    # --------------- Tab 2: Oscillators --------------------------------------
    with tab2:
        st.info("Why this chart matters:\n"
                "- RSI: overbought/oversold timing; 30–70 neutral band.\n"
                "- MACD: momentum bias (above/below 0) and turns (crosses).")
        sig = compute_signal(df)
        st.markdown(f"**Signal:** {sig['emoji']} **{sig['label']}** | MACD `{sig['macd']:.3f}` | RSI `{sig['rsi']:.1f}`")

        rows = int(show_rsi) + int(show_macd)
        if rows == 0:
            st.info("Turn on RSI or MACD in the sidebar.")
        else:
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=[0.5]*rows)

            r = 1
            if show_rsi:
                fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", mode="lines"), row=r, col=1)
                add_last_label(fig, df.index, df["RSI14"].values, "RSI14", row=r)
                fig.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2, line_width=0, row=r, col=1)
                fig.update_yaxes(range=[0, 100], row=r, col=1)  # fixed RSI scale
                r += 1

            if show_macd:
                fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"), row=r, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["MACDsig"], name="Signal", mode="lines"), row=r, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df["MACDhist"], name="Hist", opacity=0.5), row=r, col=1)
                fig.add_hline(y=0, line_width=1, opacity=0.3, row=r, col=1)  # zero baseline
                add_last_label(fig, df.index, df["MACD"].values, "MACD", row=r)

            style(fig, "Oscillators (RSI, MACD)", height=680)
            st.plotly_chart(fig, use_container_width=True)

    # --------------- Tab 3: Volatility ---------------------------------------
    with tab3:
        st.info("Why this chart matters:\n"
                "- ATR = absolute volatility (how wide the daily swing is).\n"
                "- BB width (%) = relative volatility; very low width = squeeze → watch for break.")
        sig = compute_signal(df)
        st.markdown(f"**Signal:** {sig['emoji']} **{sig['label']}**  | "
                    f"{'ATR `' + str(df['ATR14'].iloc[-1]) + '`' if 'ATR14' in df else 'ATR N/A'}")

        rows = 1 + int(show_bb and "BB_width_pct" in df)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            row_heights=[0.65] + ([0.35] if rows == 2 else []))

        if show_atr and "ATR14" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["ATR14"], name="ATR14", mode="lines"), row=1, col=1)
            add_last_label(fig, df.index, df["ATR14"].values, "ATR14", row=1)
        if rows == 2:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_width_pct"], name="BB width (%)", mode="lines"),
                          row=2, col=1)
            add_last_label(fig, df.index, df["BB_width_pct"].values, "BB width", row=2)

        style(fig, "Volatility (ATR + Bollinger width)", height=680)
        st.plotly_chart(fig, use_container_width=True)

    # --------------- Tab 4: Trends (EMA/Linear/HP) ----------------------------
    with tab4:
        st.info("Why this chart matters:\n"
                "- EMA: fast, smooth trendline.\n"
                "- Linear OLS: straight-line drift over the visible window.\n"
                "- HP filter: separates trend from cycles; λ higher = smoother.")
        sig = compute_signal(df)
        slope_hint = df["LIN_TREND"].diff().iloc[-1] if "LIN_TREND" in df else 0.0
        st.markdown(f"**Signal:** {sig['emoji']} **{sig['label']}**  | Close `{sig['close']:.2f}`  | OLS slope `{slope_hint:.2f}`")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])
        add_price_traces(fig, row=1)

        # HP trend + cycle
        fig.add_trace(go.Scatter(x=df.index, y=df["HP_trend"], name=f"HP trend (λ={int(hp_lambda)})",
                                 mode="lines", line=dict(dash="dot")), row=1, col=1)
        add_last_label(fig, df.index, df["HP_trend"].values, "HP trend", row=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["HP_cycle"], name="HP cycle", mode="lines"), row=2, col=1)
        fig.add_hline(y=0, line_width=1, opacity=0.3, row=2, col=1)
        add_last_label(fig, df.index, df["HP_cycle"].values, "HP cycle", row=2)

        style(fig, "Trendlines (EMA / Linear OLS / HP filter)", height=720)
        st.plotly_chart(fig, use_container_width=True)

    # --------------- Tab 5: Rolling OLS slope --------------------------------
    with tab5:
        st.info("Why this chart matters:\n"
                "- Shows the recent slope of price over a moving window.\n"
                "- Above 0 = upward drift; below 0 = downward drift.")
        sig = compute_signal(df)
        st.markdown(f"**Signal:** {sig['emoji']} **{sig['label']}**  | Rolling slope `{df['ROLL_SLOPE'].iloc[-1]:.4f}`")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35])
        add_price_traces(fig, row=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ROLL_SLOPE"], name=f"Slope (win={rols_win})", mode="lines"),
                      row=2, col=1)
        fig.add_hline(y=0, line_width=1, opacity=0.3, row=2, col=1)
        add_last_label(fig, df.index, df["ROLL_SLOPE"].values, "Slope", row=2)
        style(fig, "Rolling OLS slope (per bar)", height=720)
        st.plotly_chart(fig, use_container_width=True)

    # --------------- Tab 6: STL ----------------------------------------------
    with tab6:
        st.info("Why this chart matters:\n"
                "- Decomposes price into Trend + Seasonal + Residual.\n"
                "- Helps separate slow drift from repeating patterns and noise.")
        sig = compute_signal(df)
        st.markdown(f"**Signal:** {sig['emoji']} **{sig['label']}**")

        if not STL_AVAILABLE:
            st.info("STL needs enough data and a valid 'STL seasonal period'. Try expanding the date range or adjust the period.")
        else:
            stl_fit = STL(close_f, period=int(stl_period), robust=True).fit()
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.2, 0.2, 0.2])
            # Close
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines"), row=1, col=1)
            add_last_label(fig, df.index, df["Close"].values, "Close", row=1)
            # Components
            fig.add_trace(go.Scatter(x=df.index, y=stl_fit.trend, name="Trend", mode="lines"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=stl_fit.seasonal, name="Seasonal", mode="lines"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=stl_fit.resid, name="Residual", mode="lines"), row=4, col=1)
            fig.add_hline(y=0, line_width=1, opacity=0.3, row=4, col=1)
            style(fig, f"STL Decomposition (period={stl_period})", height=820)
            st.plotly_chart(fig, use_container_width=True)

else:
    # ----------------------- Single stacked figure ----------------------------
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
        fig.update_yaxes(range=[0, 100], row=r, col=1)
        add_last_label(fig, df.index, df["RSI14"].values, "RSI14", row=r)
    if show_macd:
        r += 1
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACDsig"], name="Signal", mode="lines"), row=r, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["MACDhist"], name="Hist", opacity=0.5), row=r, col=1)
        fig.add_hline(y=0, line_width=1, opacity=0.3, row=r, col=1)
        add_last_label(fig, df.index, df["MACD"].values, "MACD", row=r)
    if show_atr:
        r += 1
        fig.add_trace(go.Scatter(x=df.index, y=df["ATR14"], name="ATR14", mode="lines"), row=r, col=1)
        add_last_label(fig, df.index, df["ATR14"].values, "ATR14", row=r)

    style(fig, f"{symbol} — {interval} ({period})", height=920)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Footer ----------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Last Close", f"{float(df['Close'].iloc[-1]):,.2f}")
c2.metric("Bars", f"{len(df):,}")
c3.metric("Start → End", f"{df.index[0].date()} → {df.index[-1].date()}")

with st.expander("Show data (last 1000 rows)"):
    st.dataframe(df.tail(1000), use_container_width=True)

st.download_button("Download CSV", df.to_csv().encode("utf-8"),
                   file_name=f"{symbol}_{interval}_{period}.csv")
