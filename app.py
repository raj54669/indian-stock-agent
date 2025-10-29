import streamlit as st
import pandas as pd
import yfinance as yf
import time
import base64
import requests
from io import BytesIO

# --------------------------
# Load Secrets
# --------------------------
TELEGRAM_TOKEN = st.secrets["telegram"]["TELEGRAM_TOKEN"]
CHAT_ID = st.secrets["telegram"]["CHAT_ID"]

GITHUB_TOKEN = st.secrets["github"]["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["github"]["GITHUB_REPO"]
GITHUB_FILE_PATH = "watchlist.xlsx"

# --------------------------
# Streamlit Layout
# --------------------------
st.set_page_config(page_title="Indian Stock Auto Tracker", layout="wide")

st.sidebar.header("⚙️ Settings")
st.sidebar.success("All credentials loaded from app secrets.")
uploaded_file = st.sidebar.file_uploader("📁 Upload new watchlist (.xlsx)", type=["xlsx"])

st.title("🇮🇳 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")
st.write("""
Automatically track RSI (30–40) and 200-day EMA (±2%) for Indian stocks.  
Updates every minute and sends Telegram alerts when both conditions meet.
""")

# --------------------------
# Helper: Send Telegram Alert
# --------------------------
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        st.error(f"Telegram Error: {e}")

# --------------------------
# Helper: Load Excel from GitHub
# --------------------------
@st.cache_data(ttl=60)
def load_excel_from_github():
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    res = requests.get(api_url, headers=headers)

    if res.status_code == 200:
        content = base64.b64decode(res.json()["content"])
        return pd.read_excel(BytesIO(content))
    else:
        st.warning(f"⚠️ GitHub load failed ({res.status_code}). {res.text}")
        return None

# --------------------------
# Upload New File to GitHub
# --------------------------
def upload_excel_to_github(file):
    content = base64.b64encode(file.getvalue()).decode("utf-8")
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"

    # Check if file exists for update
    get_res = requests.get(api_url, headers=headers)
    sha = get_res.json().get("sha", None) if get_res.status_code == 200 else None

    data = {
        "message": "Updated watchlist.xlsx via Streamlit app",
        "content": content,
        "sha": sha
    }

    res = requests.put(api_url, headers=headers, json=data)
    if res.status_code in [200, 201]:
        st.success("✅ Watchlist updated successfully in GitHub!")
    else:
        st.error(f"GitHub upload failed: {res.text}")

# --------------------------
# Fetch and Calculate Stock Data
# --------------------------
def get_stock_data(symbol):
    try:
        df = yf.download(symbol, period="6mo", progress=False)
        df["EMA_200"] = df["Close"].ewm(span=200).mean()
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df
    except Exception:
        return None

# --------------------------
# Load Data and Display
# --------------------------
df = load_excel_from_github()

if uploaded_file:
    upload_excel_to_github(uploaded_file)
    st.cache_data.clear()
    time.sleep(3)
    df = load_excel_from_github()

if df is not None and "Symbol" in df.columns:
    st.success(f"✅ Loaded {len(df)} stocks from watchlist.")
    data = []

    for symbol in df["Symbol"]:
        stock = get_stock_data(symbol)
        if stock is not None and not stock.empty:
            current_price = stock["Close"].iloc[-1]
            ema_200 = stock["EMA_200"].iloc[-1]
            rsi = stock["RSI"].iloc[-1]

            near_ema = abs(current_price - ema_200) / ema_200 <= 0.02
            rsi_cond = 30 <= rsi <= 40

            data.append({
                "Symbol": symbol,
                "Price": round(current_price, 2),
                "200 EMA": round(ema_200, 2),
                "RSI": round(rsi, 2),
                "Near 200 EMA": near_ema,
                "RSI 30–40": rsi_cond
            })

            if near_ema and rsi_cond:
                send_telegram_message(f"📈 {symbol} | RSI={rsi:.2f}, Price near 200 EMA.")

    result_df = pd.DataFrame(data)
    st.dataframe(result_df, use_container_width=True)
else:
    st.warning("⚠️ No valid Excel found. Upload or ensure `watchlist.xlsx` exists in your GitHub repo.")
