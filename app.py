import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import requests
import time
from io import BytesIO

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="Indian Stock Auto Tracker", layout="wide", page_icon="📈")
st.title("🇮🇳 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")
st.markdown("""
Automatically track RSI (30–40) and 200-day EMA (±2%) for Indian stocks.  
Updates every minute and sends Telegram alerts when both conditions meet.
""")

# ------------------------------
# Load Secrets
# ------------------------------
try:
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    CHAT_ID = st.secrets["CHAT_ID"]
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["GITHUB_REPO"]
    GITHUB_FILE_PATH = st.secrets["GITHUB_FILE_PATH"]
    st.sidebar.success("✅ All credentials loaded")
except Exception as e:
    st.sidebar.error("⚠️ Missing secrets. Please check Streamlit secrets settings.")
    st.stop()

# ------------------------------
# Telegram Function
# ------------------------------
def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg}
    try:
        requests.post(url, data=data)
    except Exception as e:
        st.sidebar.error(f"Telegram error: {e}")

# ------------------------------
# GitHub Functions
# ------------------------------
def load_excel_from_github():
    """Download the Excel watchlist file from GitHub"""
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        content = res.json()["content"]
        import base64
        file_bytes = BytesIO(base64.b64decode(content))
        df = pd.read_excel(file_bytes)
        return df
    except Exception as e:
        st.error(f"Error loading from GitHub: {e}")
        return pd.DataFrame()

def upload_excel_to_github(file_data):
    """Upload (replace) Excel file to GitHub repo"""
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        get_res = requests.get(url, headers=headers)
        sha = get_res.json().get("sha")

        import base64
        content_b64 = base64.b64encode(file_data.read()).decode("utf-8")
        data = {
            "message": "Update watchlist.xlsx via Streamlit app",
            "content": content_b64,
            "sha": sha
        }
        put_res = requests.put(url, headers=headers, json=data)
        put_res.raise_for_status()
        st.sidebar.success("✅ Uploaded new watchlist to GitHub successfully.")
    except Exception as e:
        st.sidebar.error(f"GitHub upload error: {e}")

# ------------------------------
# Stock Tracking Function
# ------------------------------
def fetch_stock_data(symbol):
    try:
        data = yf.download(symbol, period="1y", interval="1d", progress=False)
        data["EMA_200"] = ta.trend.EMAIndicator(data["Close"], window=200).ema_indicator()
        data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
        latest = data.iloc[-1]

        price = latest["Close"]
        ema = latest["EMA_200"]
        rsi = latest["RSI"]
        ema_diff = ((price - ema) / ema) * 100
        meets_condition = (30 < rsi < 40) and (abs(ema_diff) <= 2)

        return {
            "Symbol": symbol,
            "Close": round(price, 2),
            "EMA_200": round(ema, 2),
            "RSI": round(rsi, 2),
            "EMA Diff (%)": round(ema_diff, 2),
            "Signal": "✅ Meets Criteria" if meets_condition else "❌ No Signal"
        }
    except Exception as e:
        return {"Symbol": symbol, "Error": str(e)}

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("📂 Upload new watchlist (.xlsx)", type=["xlsx"])
if uploaded_file:
    upload_excel_to_github(uploaded_file)
    df = pd.read_excel(uploaded_file)
else:
    df = load_excel_from_github()

# ------------------------------
# Dashboard
# ------------------------------
if not df.empty and "Symbol" in df.columns:
    symbols = df["Symbol"].dropna().tolist()
    st.success(f"Tracking {len(symbols)} stocks from GitHub watchlist")

    run_auto = st.checkbox("🔄 Run Auto Tracking (updates every 1 minute)")
    placeholder = st.empty()

    while True:
        results = []
        for sym in symbols:
            result = fetch_stock_data(sym)
            results.append(result)
            if result.get("Signal") == "✅ Meets Criteria":
                msg = (
                    f"📈 {sym}\n"
                    f"RSI: {result['RSI']}\n"
                    f"EMA Diff: {result['EMA Diff (%)']}%\n"
                    f"Condition: ✅ Meets Criteria"
                )
                send_telegram_message(msg)

        result_df = pd.DataFrame(results)
        placeholder.dataframe(result_df, use_container_width=True)

        st.sidebar.write(f"Last updated: {time.strftime('%H:%M:%S')}")
        if not run_auto:
            break
        time.sleep(60)
else:
    st.warning("⚠️ No valid Excel found in GitHub. Please upload a file with a 'Symbol' column.")
