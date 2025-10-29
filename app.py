import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import datetime
import requests
import time

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="📊 Indian Stock Monitor", layout="wide")
st.title("📈 Indian Stock Auto Alert (EMA200 & RSI14)")

# -------------------------------
# Load Secrets (Telegram)
# -------------------------------
TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

# -------------------------------
# Telegram Alert Function
# -------------------------------
def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        st.error(f"Telegram alert failed: {e}")

# -------------------------------
# Load Watchlist (Excel or Text)
# -------------------------------
def load_watchlist():
    try:
        # If Excel file present
        df = pd.read_excel("watchlist.xlsx")
        if "Symbol" in df.columns:
            return df["Symbol"].dropna().tolist()
    except Exception:
        pass

    # Fallback to text file
    try:
        with open("watchlist.txt") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []

# -------------------------------
# Fetch Stock Data & Compute Indicators
# -------------------------------
def fetch_stats(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if df.empty:
            return None

        df["EMA200"] = ta.ema(df["Close"], length=200)
        df["RSI14"] = ta.rsi(df["Close"], length=14)

        last = df.iloc[-1]
        close = float(last["Close"])
        ema200 = float(last["EMA200"])
        rsi14 = float(last["RSI14"])

        near_ema = (0.98 * ema200) < close < (1.02 * ema200)
        rsi_ok = (30 < rsi14 < 40)
        triggered = near_ema and rsi_ok

        return {
            "Symbol": symbol,
            "Price": round(close, 2),
            "EMA200": round(ema200, 2),
            "RSI14": round(rsi14, 2),
            "Near EMA": near_ema,
            "RSI 30-40": rsi_ok,
            "Triggered": triggered
        }
    except Exception as e:
        st.warning(f"Error fetching {symbol}: {e}")
        return None

# -------------------------------
# Main App
# -------------------------------
watchlist = load_watchlist()

if not watchlist:
    st.error("⚠️ No watchlist found. Please upload watchlist.xlsx or watchlist.txt.")
else:
    st.info(f"Tracking {len(watchlist)} stocks...")

    data = []
    for symbol in watchlist:
        stats = fetch_stats(symbol)
        if stats:
            data.append(stats)
            if stats["Triggered"]:
                alert_msg = (
                    f"📊 Trigger Alert!\n"
                    f"Stock: {symbol}\n"
                    f"Price: ₹{stats['Price']}\n"
                    f"EMA200: {stats['EMA200']}\n"
                    f"RSI14: {stats['RSI14']}\n"
                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                send_telegram_alert(alert_msg)

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -------------------------------
# Auto-refresh every 5 minutes
# -------------------------------
st.experimental_set_query_params(updated=str(datetime.datetime.now()))
time.sleep(300)
st.experimental_rerun()
