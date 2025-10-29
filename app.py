import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import base64
import json
from io import BytesIO
import ta

# -------------------------------
# Load secrets
# -------------------------------
TELEGRAM_TOKEN = st.secrets["telegram"]["TELEGRAM_TOKEN"]
CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
GITHUB_TOKEN = st.secrets["github"]["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["github"]["GITHUB_REPO"]
GITHUB_FILE_PATH = st.secrets["github"]["GITHUB_FILE_PATH"]

# -------------------------------
# Helper: Send Telegram Alert
# -------------------------------
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        st.error(f"Telegram Error: {e}")

# -------------------------------
# Helper: Load Excel from GitHub
# -------------------------------
@st.cache_data(ttl=300)
def load_excel_from_github():
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        content = base64.b64decode(r.json()["content"])
        df = pd.read_excel(BytesIO(content))
        return df
    else:
        st.warning(f"⚠️ Error loading from GitHub: {r.status_code} - {r.text}")
        return None

# -------------------------------
# Helper: Upload to GitHub
# -------------------------------
def upload_excel_to_github(file_data):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"

    # Check if file already exists to get SHA
    r = requests.get(url, headers=headers)
    sha = r.json().get("sha") if r.status_code == 200 else None

    data = {
        "message": "Update watchlist.xlsx via Streamlit app",
        "content": base64.b64encode(file_data).decode("utf-8"),
    }
    if sha:
        data["sha"] = sha

    res = requests.put(url, headers=headers, data=json.dumps(data))
    return res.status_code in (200, 201)

# -------------------------------
# Sidebar UI
# -------------------------------
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload new watchlist (.xlsx)", type=["xlsx"])
if uploaded_file:
    if upload_excel_to_github(uploaded_file.getvalue()):
        st.sidebar.success("✅ New file uploaded and saved to GitHub.")
    else:
        st.sidebar.error("❌ Failed to upload to GitHub.")

# -------------------------------
# Main App
# -------------------------------
st.title("🇮🇳 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")
st.write("Automatically track RSI (30–40) and 200-day EMA (±2%) for Indian stocks. Updates every minute and sends Telegram alerts when both conditions meet.")

# Load Excel
df = load_excel_from_github()

if df is None or "Symbol" not in df.columns:
    st.warning("⚠️ No valid Excel found in GitHub. Please upload a file with a 'Symbol' column.")
    st.stop()

st.success(f"✅ Loaded {len(df)} stocks from GitHub. Starting tracking...")

# -------------------------------
# Core Logic
# -------------------------------
results = []

for symbol in df["Symbol"].dropna():
    try:
        data = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if len(data) < 50:
            continue
        data["EMA200"] = ta.trend.EMAIndicator(data["Close"], window=200).ema_indicator()
        data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()

        latest = data.iloc[-1]
        price = latest["Close"]
        ema200 = latest["EMA200"]
        rsi = latest["RSI"]

        near_ema = (ema200 * 0.98) <= price <= (ema200 * 1.02)
        rsi_range = 30 <= rsi <= 40

        if near_ema and rsi_range:
            msg = f"📈 Alert: {symbol}\nPrice ₹{price:.2f} near 200 EMA ({ema200:.2f})\nRSI: {rsi:.2f}"
            send_telegram_message(msg)

        results.append({"Symbol": symbol, "Price": price, "EMA200": ema200, "RSI": rsi, "Near EMA": near_ema, "RSI 30–40": rsi_range})

    except Exception as e:
        results.append({"Symbol": symbol, "Error": str(e)})

# -------------------------------
# Display Dashboard
# -------------------------------
if results:
    st.dataframe(pd.DataFrame(results))
else:
    st.info("No valid stock data found.")
