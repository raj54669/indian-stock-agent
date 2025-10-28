import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import time
from datetime import datetime

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Indian Stock Live Monitor", page_icon="📊", layout="wide")
st.title("📊 Indian Stock Live Monitor (EMA200 & RSI14)")
st.caption("Auto-refresh every 60 seconds | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# -------------------------------
# Load Watchlist
# -------------------------------
def load_watchlist(filename="watchlist.txt"):
    try:
        with open(filename, "r") as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        return symbols
    except FileNotFoundError:
        st.error("⚠️ 'watchlist.txt' not found in project directory.")
        return []
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")
        return []

# -------------------------------
# Fetch Stock Data
# -------------------------------
def fetch_stats(symbol):
    try:
        # Fetch 6 months of data (fast & sufficient)
        df = yf.download(symbol, period="6mo", interval="1d", progress=False, threads=False)
        if df.empty:
            return {"Symbol": symbol, "error": "No historical data"}

        # Compute indicators
        df["EMA200"] = ta.ema(df["Close"], length=200)
        df["RSI14"] = ta.rsi(df["Close"], length=14)

        last_row = df.iloc[-1]
        ema200 = last_row.get("EMA200")
        rsi14 = last_row.get("RSI14")
        close_price = last_row.get("Close")

        # Validate numbers
        if any(pd.isna(x) for x in [ema200, rsi14, close_price]):
            return {"Symbol": symbol, "error": "Insufficient data for EMA/RSI"}

        ema200 = float(ema200)
        rsi14 = float(rsi14)
        close_price = float(close_price)

        # Fetch recent 1-min price (optional quick check)
        try:
            intraday = yf.download(symbol, period="2d", interval="1m", progress=False, threads=False)
            if intraday is not None and not intraday.empty:
                latest_close = float(intraday["Close"].iloc[-1])
                price_time = intraday.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            else:
                latest_close = close_price
                price_time = df.index[-1].strftime("%Y-%m-%d")
        except Exception:
            latest_close = close_price
            price_time = df.index[-1].strftime("%Y-%m-%d")

        # Define triggers
        near_ema = (0.98 * ema200) <= latest_close <= (1.02 * ema200)
        rsi_ok = 30 <= rsi14 <= 40
        triggered = near_ema and rsi_ok

        return {
            "Symbol": symbol,
            "Price": round(latest_close, 2),
            "Price Time": price_time,
            "EMA200": round(ema200, 2),
            "RSI14": round(rsi14, 2),
            "Near EMA?": "✅" if near_ema else "❌",
            "RSI 30–40?": "✅" if rsi_ok else "❌",
            "Triggered": "✅" if triggered else "❌",
            "error": ""
        }

    except Exception as e:
        return {"Symbol": symbol, "error": str(e)}

# -------------------------------
# Dashboard Logic
# -------------------------------
watchlist = load_watchlist()

if not watchlist:
    st.warning("⚠️ Please add stock symbols (like INFY.NS, TCS.NS) in `watchlist.txt`.")
else:
    results = []
    progress = st.progress(0)
    for i, symbol in enumerate(watchlist):
        stats = fetch_stats(symbol)
        results.append(stats)
        progress.progress((i + 1) / len(watchlist))
        time.sleep(0.5)  # small delay to prevent rate-limit issues

    df = pd.DataFrame(results)

    valid_df = df[df["error"] == ""]
    error_df = df[df["error"] != ""]

    if not valid_df.empty:
        st.subheader("✅ Live Stock Status")
        st.dataframe(valid_df, use_container_width=True)
    else:
        st.info("No valid stock data currently available.")

    if not error_df.empty:
        with st.expander("⚠️ View Stocks with Errors"):
            st.dataframe(error_df, use_container_width=True)

# -------------------------------
# Auto Refresh
# -------------------------------
st_autorefresh = st.empty()
st_autorefresh.caption("🔄 Auto-refresh every 60 seconds")
time.sleep(60)
st.experimental_rerun()
