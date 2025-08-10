import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="BTC 4h Trend", layout="wide")
st.title("BTC Trend (4-Hour Candles)")

st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Symbol", value="BTC-USD")
interval = st.sidebar.selectbox("Interval", ["4h", "1h", "1d"], index=0)
period = st.sidebar.selectbox("Period", ["30d", "90d", "365d", "730d"], index=3)
show_ma20 = st.sidebar.checkbox("Show SMA 20", value=True)
show_ma50 = st.sidebar.checkbox("Show SMA 50", value=True)
show_ma200 = st.sidebar.checkbox("Show SMA 200", value=False)
use_log = st.sidebar.checkbox("Log scale (price)", value=False)
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()

@st.cache_data(ttl=60*15)
def load_data(sym, per, inter):
    df = yf.download(sym, period=per, interval=inter, auto_adjust=False, progress=False).dropna()
    return df

with st.spinner("Loading data…"):
    df = load_data(symbol, period, interval)

if df.empty:
    st.warning("No data returned. Try a shorter period or different interval.")
    st.stop()

# indicators
if show_ma20:  df["SMA20"]  = df["Close"].rolling(20).mean()
if show_ma50:  df["SMA50"]  = df["Close"].rolling(50).mean()
if show_ma200: df["SMA200"] = df["Close"].rolling(200).mean()

# plot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
for col, dash in [("SMA20","solid"),("SMA50","dot"),("SMA200","dash")]:
    if col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode="lines", line=dict(dash=dash)), row=1, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.6), row=2, col=1)
fig.update_layout(title=f"{symbol} — {interval} ({period})", xaxis_rangeslider_visible=False, hovermode="x unified",
                  margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h", y=1.02, x=0))
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True, type=("log" if use_log else "linear"), row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

last_close = float(df["Close"].iloc[-1])
c1,c2,c3 = st.columns(3)
c1.metric("Last Close", f"{last_close:,.2f}")
c2.metric("Bars", f"{len(df):,}")
c3.metric("Start → End", f"{df.index[0].date()} → {df.index[-1].date()}")

with st.expander("Show data (last 500 rows)"):
    st.dataframe(df.tail(500), use_container_width=True)

st.download_button("Download CSV", df.to_csv().encode("utf-8"), file_name=f"{symbol}_{interval}_{period}.csv")
