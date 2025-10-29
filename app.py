import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import time
import requests

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Indian Stock Agent", page_icon="📈", layout="wide")
st.title("📊 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")

st.write("""
Upload your **watchlist Excel file** (must contain a column named `Symbol` with stock tickers like `RELIANCE.NS`).
The app will auto-check RSI (30–40) and if price is within ±2% of 200-day EMA.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Watchlist Excel", type=["xlsx"])

# --- Telegram Setup ---
st.sidebar.header("🔔 Telegram Bot Settings")
TELEGRAM_TOKEN = st.sidebar.text_input("Bot Token", type="password")
CHAT_ID = st.sidebar.text_input("Chat ID")

# --- Functions ---
def analyze_stock(symbol):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if df.empty:
            return None
        
        df.dropna(inplace=True)
        df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        latest = df.iloc[-1]
        price = latest['Close']
        ema = latest['EMA_200']
        rsi = latest['RSI']

        near_ema = (ema * 0.98) <= price <= (ema * 1.02)
        rsi_ok = 30 <= rsi <= 40

        return {
            "Symbol": symbol,
            "Price": round(price, 2),
            "EMA_200": round(ema, 2),
            "RSI": round(rsi, 2),
            "Near 200 EMA": near_ema,
            "RSI Condition": rsi_ok,
            "Alert": near_ema and rsi_ok
        }
    except Exception as e:
        return {"Symbol": symbol, "Error": str(e)}

def send_telegram_alert(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        st.sidebar.error(f"Telegram Error: {e}")

# --- Main Loop ---
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    if "Symbol" not in df.columns:
        st.error("Excel must contain a column named 'Symbol'.")
    else:
        st.success(f"Loaded {len(df)} stocks. Starting tracking...")

        run_tracking = st.checkbox("Run Auto Tracking (updates every 1 minute)")
        results_placeholder = st.empty()

        while run_tracking:
            results = []
            for symbol in df["Symbol"]:
                result = analyze_stock(symbol)
                if result:
                    results.append(result)
                    if result.get("Alert"):
                        msg = f"📢 ALERT: {symbol}\nPrice: {result['Price']}\nEMA200: {result['EMA_200']}\nRSI: {result['RSI']}"
                        if TELEGRAM_TOKEN and CHAT_ID:
                            send_telegram_alert(TELEGRAM_TOKEN, CHAT_ID, msg)
            
            results_df = pd.DataFrame(results)
            results_placeholder.dataframe(results_df, use_container_width=True)
            st.toast("Updated Stock Data ✅")
            time.sleep(60)  # wait 1 minute
else:
    st.info("Please upload your watchlist Excel file to begin.")
