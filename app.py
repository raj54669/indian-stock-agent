import streamlit as st
import yfinance as yf
import pandas_ta as ta
import requests
import os
import time
import threading
import pandas as pd

# ───────────────────────────────
# TELEGRAM ALERT FUNCTION
# ───────────────────────────────
def send_telegram_alert(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ Telegram credentials not set.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        r = requests.post(url, data=payload, timeout=10)
        r.raise_for_status()
        print(f"✅ Alert sent: {message}")
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")

# ───────────────────────────────
# INDICATOR CALCULATION
# ───────────────────────────────
def get_stock_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if df.empty:
        return None
    df["EMA200"] = ta.ema(df["Close"], length=200)
    df["RSI"] = ta.rsi(df["Close"], length=14)
    return df

def check_conditions(ticker):
    df = get_stock_data(ticker)
    if df is None:
        return None
    latest = df.iloc[-1]
    price = latest["Close"]
    ema = latest["EMA200"]
    rsi = latest["RSI"]

    near_ema = ema * 0.98 < price < ema * 1.02
    rsi_ok = 30 < rsi < 40
    return near_ema, rsi_ok, price, ema, rsi

# ───────────────────────────────
# WATCHLIST LOADER
# ───────────────────────────────
def load_watchlist(path="watchlist.xlsx"):
    try:
        df = pd.read_excel(path)
        return df["Ticker"].dropna().tolist()
    except Exception as e:
        print(f"⚠️ Cannot load watchlist: {e}")
        return []

# ───────────────────────────────
# BACKGROUND TRACKER
# ───────────────────────────────
def track_stocks():
    while True:
        watchlist = load_watchlist()
        if not watchlist:
            print("⚠️ Watchlist empty.")
        for symbol in watchlist:
            res = check_conditions(symbol)
            if not res:
                continue
            near_ema, rsi_ok, price, ema, rsi = res
            if near_ema and rsi_ok:
                msg = (f"🚨 {symbol}\nPrice ₹{price:.2f}\n"
                       f"EMA200 ₹{ema:.2f}\nRSI {rsi:.2f}\n"
                       f"✅ Price near 200 EMA & RSI 30-40")
                send_telegram_alert(msg)
            else:
                print(f"{symbol}: no trigger.")
        time.sleep(60)  # repeat every minute

def start_background_thread():
    t = threading.Thread(target=track_stocks, daemon=True)
    t.start()

# ───────────────────────────────
# STREAMLIT UI
# ───────────────────────────────
st.set_page_config(page_title="📊 Indian Stock Monitor", layout="wide")
st.title("📈 Indian Stock Monitor (EMA200 + RSI 14)")

st.info("This app auto-tracks your Excel watchlist every minute "
        "and sends Telegram alerts when both EMA and RSI conditions meet.")

if st.button("▶️ Start Monitoring"):
    start_background_thread()
    st.success("Monitoring started — alerts will appear in your Telegram bot!")

uploaded = st.file_uploader("Upload new watchlist (Excel with Ticker column)",
                            type=["xlsx"])
if uploaded:
    df_up = pd.read_excel(uploaded)
    df_up.to_excel("watchlist.xlsx", index=False)
    st.success("✅ Watchlist updated.")

st.divider()
st.subheader("Sample Chart Preview")
sample = st.text_input("Enter Ticker (eg: RELIANCE.NS)", "RELIANCE.NS")
if sample:
    data = yf.download(sample, period="1mo", interval="1d", progress=False)
    st.line_chart(data["Close"])
