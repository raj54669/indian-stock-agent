import streamlit as st
import pandas as pd
import yfinance as yf
import io
import requests
import ta
from github import Github

st.set_page_config(page_title="Indian Stock Auto Tracker (EMA + RSI Alert Bot)", page_icon="📈", layout="wide")

# ------------------ Load secrets safely ------------------
try:
    TELEGRAM_TOKEN = st.secrets["telegram"]["TELEGRAM_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    GITHUB_TOKEN = st.secrets["github"]["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["github"]["GITHUB_REPO"]
    GITHUB_FILE_PATH = st.secrets["github"]["GITHUB_FILE_PATH"]
except Exception as e:
    st.error("❌ Missing Streamlit secrets configuration.")
    st.stop()

# ------------------ GitHub setup ------------------
repo = None
try:
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)
except Exception as e:
    st.warning(f"⚠️ GitHub repo init failed: {e}")

# ------------------ Helper functions ------------------
def send_telegram_message(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        requests.post(url, data=payload)
    except Exception as e:
        st.warning(f"⚠️ Telegram send failed: {e}")

def load_watchlist_from_github():
    try:
        file_content = repo.get_contents(GITHUB_FILE_PATH)
        content = file_content.decoded_content
        df = pd.read_excel(io.BytesIO(content))
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load watchlist from GitHub: {e}")
        return pd.DataFrame(columns=["Symbol"])

def upload_watchlist_to_github(uploaded_file):
    try:
        file_bytes = uploaded_file.getvalue()
        existing_file = None
        try:
            existing_file = repo.get_contents(GITHUB_FILE_PATH)
            repo.update_file(GITHUB_FILE_PATH, "update watchlist", file_bytes, existing_file.sha, branch="main")
        except Exception:
            repo.create_file(GITHUB_FILE_PATH, "create watchlist", file_bytes, branch="main")
        st.success("✅ File uploaded to GitHub successfully!")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

def scan_stock(symbol):
    try:
        data = yf.download(symbol, period="6mo", progress=False)
        if data.empty:
            return None
        data["EMA_200"] = ta.trend.EMAIndicator(data["Close"], window=200).ema_indicator()
        data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
        last = data.iloc[-1]
        return {
            "Symbol": symbol,
            "Close": round(last["Close"], 2),
            "EMA_200": round(last["EMA_200"], 2),
            "RSI": round(last["RSI"], 2)
        }
    except Exception:
        return None

def run_scan(df):
    results = []
    for symbol in df["Symbol"]:
        res = scan_stock(symbol)
        if res:
            results.append(res)
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        buy_signals = results_df[(results_df["RSI"] < 30) & (results_df["Close"] > results_df["EMA_200"])]
        sell_signals = results_df[(results_df["RSI"] > 70) & (results_df["Close"] < results_df["EMA_200"])]
        if not buy_signals.empty or not sell_signals.empty:
            msg = "📊 *Stock Alert Update:*\n\n"
            for _, row in buy_signals.iterrows():
                msg += f"🟢 BUY: {row['Symbol']} | RSI={row['RSI']} | Close={row['Close']}\n"
            for _, row in sell_signals.iterrows():
                msg += f"🔴 SELL: {row['Symbol']} | RSI={row['RSI']} | Close={row['Close']}\n"
            send_telegram_message(msg)
        return results_df
    return pd.DataFrame()

# ------------------ Streamlit UI ------------------
st.title("📈 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")
st.caption("Automatically checks RSI(14) and 200-day EMA proximity for symbols in `watchlist.xlsx`")

st.sidebar.header("⚙️ Settings")

# Telegram Status
if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.warning("⚠️ Telegram secrets not set")

# GitHub Status
if repo:
    st.sidebar.success("✅ GitHub connected")
else:
    st.sidebar.error("❌ GitHub token or repo not configured")

# File Upload Section
st.sidebar.subheader("📤 Upload new watchlist (.xlsx)")
uploaded_file = st.sidebar.file_uploader("Choose a watchlist Excel file", type=["xlsx"])
if uploaded_file is not None:
    upload_watchlist_to_github(uploaded_file)
    st.experimental_rerun()

# Load Data
df = load_watchlist_from_github()
if df.empty:
    st.warning("⚠️ No valid Excel found in GitHub. Please upload a file named `watchlist.xlsx` with a 'Symbol' column.")
    st.markdown("#### Required format example:")
    st.table(pd.DataFrame({"Symbol": ["RELIANCE.NS", "TCS.NS"]}))
else:
    st.dataframe(df)

# Scan Controls
st.header("🕹️ Controls")
if st.button("Run Scan Now"):
    st.info("🔍 Running scan...")
    results = run_scan(df)
    if results.empty:
        st.warning("No data found or API issue.")
    else:
        st.success("✅ Scan completed.")
        st.dataframe(results)

# Auto refresh (optional)
auto_interval = st.number_input("Auto scan interval (seconds)", min_value=30, max_value=600, value=60)
if st.checkbox("Enable Auto Tracking (loop in session)"):
    import time
    st.info("Running auto loop... stop with rerun.")
    while True:
        results = run_scan(df)
        time.sleep(auto_interval)
