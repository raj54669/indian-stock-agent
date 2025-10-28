# dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
import datetime

st.set_page_config(page_title="Indian Stock Monitor", layout="wide")
st.title("📊 Indian Stock Live Monitor (EMA200 & RSI14)")

WATCHLIST_PATH = "watchlist.txt"
REFRESH_SECONDS = int(os.getenv("DASH_REFRESH", "60"))

def load_watchlist():
    if not os.path.exists(WATCHLIST_PATH):
        return []
    with open(WATCHLIST_PATH) as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

def fetch_stats(symbol):
    # daily indicators
    df = yf.download(symbol, period="1y", interval="1d", progress=False, threads=False)
    if df.empty:
        return None
    df["EMA200"] = ta.ema(df["Close"], length=200)
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    last_daily = df.iloc[-1]
    ema200 = float(last_daily["EMA200"])
    rsi14 = float(last_daily["RSI14"])
    daily_close = float(last_daily["Close"])
    # try intraday latest price
    try:
        intr = yf.download(symbol, period="2d", interval="1m", progress=False, threads=False)
        latest_close = float(intr["Close"].iloc[-1]) if (intr is not None and not intr.empty) else daily_close
        price_time = intr.index[-1].to_pydatetime().strftime("%Y-%m-%d %H:%M:%S") if (intr is not None and not intr.empty) else df.index[-1].strftime("%Y-%m-%d")
    except Exception:
        latest_close = daily_close
        price_time = df.index[-1].strftime("%Y-%m-%d")

    near_ema = (0.98 * ema200) < latest_close < (1.02 * ema200)
    rsi_ok = (30 < rsi14 < 40)
    triggered = near_ema and rsi_ok

    return {
        "Symbol": symbol,
        "Price": round(latest_close, 2),
        "Price Time": price_time,
        "EMA200": round(ema200, 2),
        "RSI14": round(rsi14, 2),
        "Near EMA?": "✅" if near_ema else "❌",
        "RSI 30-40?": "✅" if rsi_ok else "❌",
        "Triggered": "✅" if triggered else "❌"
    }

watchlist = load_watchlist()
if not watchlist:
    st.warning("watchlist.txt is empty or not present in the repository.")
else:
    data = []
    with st.spinner("Fetching data..."):
        for symbol in watchlist:
            stats = fetch_stats(symbol)
            if stats:
                data.append(stats)
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No data to display.")

st.caption(f"Auto-refresh every {REFRESH_SECONDS} seconds. Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# set auto-refresh using streamlit's experimental function
st.experimental_rerun()
