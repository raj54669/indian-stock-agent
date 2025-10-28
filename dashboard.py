# dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
import datetime
import time

# --- Page setup ---
st.set_page_config(page_title="Indian Stock Monitor", layout="wide")
st.title("📊 Indian Stock Live Monitor (EMA200 & RSI14)")

WATCHLIST_PATH = "watchlist.txt"
REFRESH_SECONDS = int(os.getenv("DASH_REFRESH", "60"))


# --- Load Watchlist ---
def load_watchlist():
    if not os.path.exists(WATCHLIST_PATH):
        return []
    with open(WATCHLIST_PATH) as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]


# --- Fetch Stock Data ---
def fetch_stats(symbol):
    try:
        # Get last 1 year daily data
        df = yf.download(symbol, period="1y", interval="1d", progress=False, threads=False)
        if df.empty:
            return {"Symbol": symbol, "error": "No data returned from Yahoo Finance"}

        # Calculate EMA200 & RSI14
        df["EMA200"] = ta.ema(df["Close"], length=200)
        df["RSI14"] = ta.rsi(df["Close"], length=14)

        # Get latest row
        last_daily = df.iloc[-1]

        # Safely extract values
        ema200_value = last_daily.get("EMA200", None)
        rsi_value = last_daily.get("RSI14", None)
        close_value = last_daily.get("Close", None)

        # Validate values before converting
        if ema200_value is None or pd.isna(ema200_value):
            return {"Symbol": symbol, "error": "EMA200 is None"}
        if rsi_value is None or pd.isna(rsi_value):
            return {"Symbol": symbol, "error": "RSI14 is None"}
        if close_value is None or pd.isna(close_value):
            return {"Symbol": symbol, "error": "Close price is None"}

        ema200 = float(ema200_value)
        rsi14 = float(rsi_value)
        daily_close = float(close_value)

        # Intraday latest price (optional)
        try:
            intr = yf.download(symbol, period="2d", interval="1m", progress=False, threads=False)
            if intr is not None and not intr.empty:
                latest_close = float(intr["Close"].iloc[-1])
                price_time = intr.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            else:
                latest_close = daily_close
                price_time = df.index[-1].strftime("%Y-%m-%d")
        except Exception:
            latest_close = daily_close
            price_time = df.index[-1].strftime("%Y-%m-%d")

        # Define trigger logic
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
            "Triggered": "✅" if triggered else "❌",
        }

    except Exception as e:
        return {"Symbol": symbol, "error": str(e)}


# --- Footer ---
st.caption(
    f"⏱️ Auto-refresh every {REFRESH_SECONDS} seconds | "
    f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

# --- Gentle Auto-Refresh ---
time.sleep(REFRESH_SECONDS)
st.experimental_rerun()
