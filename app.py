import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import time
import os
import requests
from datetime import datetime

# -------------------------------------
# Page Configuration
# -------------------------------------
st.set_page_config(
    page_title="Indian Stock Auto Tracker",
    page_icon="📈",
    layout="wide"
)

st.title("🇮🇳 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")
st.write("""
Upload your **watchlist Excel file** (must contain a column named `Symbol` with stock tickers like `RELIANCE.NS`).  
The app will auto-check RSI (30–40) and price proximity to the 200-day EMA (±2%).  
If both conditions meet, it will automatically send a Telegram alert.
""")

# -------------------------------------
# Telegram Setup (using your secret keys)
# -------------------------------------
try:
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    CHAT_ID = st.secrets["CHAT_ID"]
    st.sidebar.success("✅ Telegram bot connected (from secrets)")
except Exception:
    st.sidebar.error("⚠️ Telegram credentials missing in secrets.toml!")

def send_telegram_message(msg):
    """Send Telegram notification"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": msg}
        requests.post(url, data=data)
    except Exception as e:
        st.sidebar.error(f"Telegram error: {e}")

# -------------------------------------
# Sidebar Settings
# -------------------------------------
st.sidebar.header("⚙️ App Settings")

default_path = "watchlist.xlsx"
df = pd.DataFrame()

# Load existing Excel if available
if os.path.exists(default_path):
    try:
        df = pd.read_excel(default_path)
        st.sidebar.success(f"Loaded default watchlist with {len(df)} stocks.")
    except Exception:
        st.sidebar.warning("Could not read default watchlist.xlsx.")

# Upload new Excel (replaces existing)
uploaded_file = st.sidebar.file_uploader("📂 Upload new watchlist (.xlsx)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.to_excel(default_path, index=False)
    st.sidebar.success(f"✅ Replaced watchlist with {len(df)} stocks.")

# -------------------------------------
# Helper function for stock data
# -------------------------------------
def fetch_stock_data(symbol):
    """Fetch EMA, RSI, and current price"""
    try:
        data = yf.download(symbol, period="1y", interval="1d", progress=False)
        data["EMA_200"] = ta.trend.EMAIndicator(data["Close"], window=200).ema_indicator()
        data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()

        latest = data.iloc[-1]
        price = latest["Close"]
        ema = latest["EMA_200"]
        rsi = latest["RSI"]
        ema_proximity = ((price - ema) / ema) * 100

        condition = (30 < rsi < 40) and (abs(ema_proximity) <= 2)

        return {
            "Symbol": symbol,
            "Close": round(price, 2),
            "EMA_200": round(ema, 2),
            "RSI": round(rsi, 2),
            "EMA_Proximity(%)": round(ema_proximity, 2),
            "Signal": "✅ Meets Criteria" if condition else "❌ No Signal"
        }
    except Exception as e:
        return {"Symbol": symbol, "Error": str(e)}

# -------------------------------------
# Dashboard
# -------------------------------------
if not df.empty and "Symbol" in df.columns:
    symbols = df["Symbol"].dropna().tolist()
    st.success(f"Loaded {len(symbols)} stocks. Tracking started.")

    run_auto = st.checkbox("🔄 Run Auto Tracking (updates every 1 minute)")
    placeholder = st.empty()

    while True:
        results = []
        for sym in symbols:
            result = fetch_stock_data(sym)
            results.append(result)
            if result.get("Signal") == "✅ Meets Criteria":
                message = f"📈 {sym} - RSI: {result['RSI']}, EMA Diff: {result['EMA_Proximity(%)']}% ✅ Meets Conditions!"
                send_telegram_message(message)

        result_df = pd.DataFrame(results)
        placeholder.dataframe(result_df, use_container_width=True)

        if not run_auto:
            break
        time.sleep(60)

else:
    st.warning("⚠️ Please upload a valid Excel file with a column named 'Symbol'.")
