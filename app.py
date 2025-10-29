import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
import base64
import time
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# ---------------------------
# 🔧 APP CONFIG
# ---------------------------
st.set_page_config(
    page_title="Indian Stock Auto Tracker (EMA + RSI Alert Bot)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 🔐 Load secrets
# ---------------------------
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
CHAT_ID = st.secrets["CHAT_ID"]
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
GITHUB_FILE_PATH = st.secrets["GITHUB_FILE_PATH"]

# ---------------------------
# 📦 Function: Send Telegram Alert
# ---------------------------
def send_telegram_message(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Failed to send Telegram message: {e}")

# ---------------------------
# 📁 Function: Load Excel from GitHub
# ---------------------------
def load_watchlist_from_github():
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            content = res.json().get("content")
            decoded = base64.b64decode(content)
            df = pd.read_excel(io.BytesIO(decoded))
            st.sidebar.success("✅ Watchlist loaded from GitHub.")
            return df
        else:
            st.sidebar.warning(f"⚠️ Error loading from GitHub: {res.status_code}")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading from GitHub: {e}")
        return None

# ---------------------------
# 💾 Function: Upload new file to GitHub (replace old)
# ---------------------------
def upload_watchlist_to_github(file):
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

        # Get current file SHA (if exists)
        res = requests.get(url, headers=headers)
        sha = res.json().get("sha") if res.status_code == 200 else None

        content = base64.b64encode(file.read()).decode()
        data = {
            "message": "Updated watchlist.xlsx via Streamlit app",
            "content": content,
            "sha": sha
        }

        put_res = requests.put(url, headers=headers, json=data)
        if put_res.status_code in [200, 201]:
            st.sidebar.success("✅ Watchlist updated successfully on GitHub.")
        else:
            st.sidebar.error(f"GitHub upload failed: {put_res.text}")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

# ---------------------------
# 📊 Function: Fetch stock data + Compute EMA & RSI
# ---------------------------
def analyze_stock(symbol):
    try:
        data = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if data.empty:
            return None

        data["EMA200"] = EMAIndicator(data["Close"], window=200).ema_indicator()
        data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()

        latest = data.iloc[-1]
        current_price = latest["Close"]
        ema200 = latest["EMA200"]
        rsi = latest["RSI"]

        near_ema = (ema200 * 0.98) <= current_price <= (ema200 * 1.02)
        rsi_ok = 30 <= rsi <= 40

        return {
            "Symbol": symbol,
            "Price": round(current_price, 2),
            "EMA200": round(ema200, 2),
            "RSI": round(rsi, 2),
            "Near 200 EMA": "✅" if near_ema else "❌",
            "RSI 30–40": "✅" if rsi_ok else "❌",
        }
    except Exception:
        return None

# ---------------------------
# 🧠 Function: Run Analysis on watchlist
# ---------------------------
def run_analysis(df):
    results = []
    for symbol in df["Symbol"]:
        result = analyze_stock(symbol)
        if result:
            results.append(result)
    return pd.DataFrame(results)

# ---------------------------
# 🎯 MAIN APP
# ---------------------------

st.sidebar.title("⚙️ Settings")
st.sidebar.info("All credentials loaded ✅")

uploaded_file = st.sidebar.file_uploader("Upload new watchlist (.xlsx)", type=["xlsx"])
if uploaded_file:
    upload_watchlist_to_github(uploaded_file)

# Try loading from GitHub
watchlist_df = load_watchlist_from_github()

st.title("🇮🇳 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")
st.write(
    "Automatically track RSI (30–40) and 200-day EMA (±2%) for Indian stocks. "
    "Updates every minute and sends Telegram alerts when both conditions meet."
)

# ---------------------------
# 🧾 Display results
# ---------------------------
if watchlist_df is not None and "Symbol" in watchlist_df.columns:
    results = run_analysis(watchlist_df)

    st.subheader("📈 Current Stock Analysis")
    st.dataframe(results, use_container_width=True)

    signals = results[
        (results["Near 200 EMA"] == "✅") & (results["RSI 30–40"] == "✅")
    ]

    if not signals.empty:
        message = "🚨 Stock Alerts:\n\n" + "\n".join(
            [f"{row['Symbol']} | Price: ₹{row['Price']} | RSI: {row['RSI']}" for _, row in signals.iterrows()]
        )
        st.success("✅ Conditions met! Sending Telegram alert.")
        send_telegram_message(message)
    else:
        st.info("No stock currently meeting both EMA and RSI conditions.")

    # Background auto tracking
    if st.sidebar.checkbox("🔁 Run Auto Tracking (update every 1 min)"):
        st.sidebar.write("Tracking started...")
        while True:
            results = run_analysis(watchlist_df)
            signals = results[
                (results["Near 200 EMA"] == "✅") & (results["RSI 30–40"] == "✅")
            ]
            if not signals.empty:
                message = "🚨 Stock Alerts:\n\n" + "\n".join(
                    [f"{row['Symbol']} | Price: ₹{row['Price']} | RSI: {row['RSI']}" for _, row in signals.iterrows()]
                )
                send_telegram_message(message)
            time.sleep(60)
else:
    st.warning("⚠️ No valid Excel file found. Please upload `watchlist.xlsx` to GitHub or via sidebar.")
