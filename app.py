import streamlit as st
import pandas as pd
import yfinance as yf
import io
from github import Github
import requests
import time

# ============================================================
# 🔐 1. Load Secrets (flat TOML style)
# ============================================================
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
CHAT_ID = st.secrets["CHAT_ID"]
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
GITHUB_FILE_PATH = st.secrets["GITHUB_FILE_PATH"]

# ============================================================
# ⚙️ 2. Initialize GitHub Connection
# ============================================================
try:
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO)
    st.sidebar.success("GitHub configured")
except Exception as e:
    repo = None
    st.sidebar.error(f"GitHub connection failed: {e}")

# ============================================================
# 💬 3. Telegram Functions
# ============================================================
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        params = {"chat_id": CHAT_ID, "text": message}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            st.sidebar.success("Telegram alert sent ✅")
        else:
            st.sidebar.warning("Failed to send Telegram message")
    except Exception as e:
        st.sidebar.error(f"Telegram error: {e}")

# ============================================================
# 📂 4. Load Watchlist from GitHub
# ============================================================
@st.cache_data(ttl=300)
def load_watchlist():
    if not repo:
        return None
    try:
        file = repo.get_contents(GITHUB_FILE_PATH)
        content = file.decoded_content
        df = pd.read_excel(io.BytesIO(content))
        return df
    except Exception as e:
        st.warning(f"Could not load watchlist from GitHub: {e}")
        return None

# ============================================================
# 📈 5. RSI and EMA Calculation
# ============================================================
def calculate_indicators(symbol):
    try:
        data = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if data.empty:
            return None

        data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        data["RSI"] = 100 - (100 / (1 + rs))
        return data
    except Exception as e:
        st.error(f"Error calculating indicators for {symbol}: {e}")
        return None

# ============================================================
# 🧠 6. Analyze Signals
# ============================================================
def analyze_symbol(symbol):
    data = calculate_indicators(symbol)
    if data is None or data.empty:
        return None

    last = data.iloc[-1]
    close, ema, rsi = last["Close"], last["EMA_200"], last["RSI"]
    status = ""

    if close > ema and rsi < 30:
        status = "🔵 BUY Signal (RSI<30 & Close>EMA200)"
    elif close < ema and rsi > 70:
        status = "🔴 SELL Signal (RSI>70 & Close<EMA200)"
    else:
        status = "⚪ Neutral"

    return {
        "Symbol": symbol,
        "Close": round(close, 2),
        "EMA_200": round(ema, 2),
        "RSI": round(rsi, 2),
        "Signal": status,
    }

# ============================================================
# 🧮 7. Streamlit UI
# ============================================================
st.set_page_config(page_title="Indian Stock Auto Tracker", layout="wide")
st.title("📈 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")

st.sidebar.header("⚙️ Settings")

telegram_ok = bool(TELEGRAM_TOKEN and CHAT_ID)
if telegram_ok:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.warning("⚠️ Telegram secrets not set. Alerts disabled.")

github_ok = bool(GITHUB_TOKEN and GITHUB_REPO and GITHUB_FILE_PATH)
if github_ok:
    st.sidebar.success("✅ GitHub configured")
else:
    st.sidebar.error("❌ GitHub token or repo not configured")

# ============================================================
# 📊 8. Load Watchlist
# ============================================================
watchlist_df = load_watchlist()

if watchlist_df is None or "Symbol" not in watchlist_df.columns:
    st.warning("No valid Excel found in GitHub. Please upload a file named `watchlist.xlsx` with a 'Symbol' column.")
    st.write("Example format:")
    st.dataframe(pd.DataFrame({"Symbol": ["RELIANCE.NS", "TCS.NS"]}))
else:
    st.dataframe(watchlist_df)

# ============================================================
# 🚀 9. Controls
# ============================================================
st.subheader("🧠 Controls")

col1, col2 = st.columns([1, 2])
with col1:
    auto_interval = st.number_input("Auto scan interval (seconds)", value=60, step=10)
with col2:
    run_once = st.button("🔄 Run Scan Now")

def run_scan():
    if watchlist_df is None or "Symbol" not in watchlist_df.columns:
        st.error("No valid symbols found.")
        return

    results = []
    alerts = []
    for symbol in watchlist_df["Symbol"]:
        res = analyze_symbol(symbol)
        if res:
            results.append(res)
            if "BUY" in res["Signal"] or "SELL" in res["Signal"]:
                alerts.append(f"{res['Symbol']}: {res['Signal']} (RSI={res['RSI']}, Close={res['Close']})")

    if results:
        st.dataframe(pd.DataFrame(results))
        if alerts and telegram_ok:
            send_telegram_message("\n".join(alerts))
        elif not alerts:
            st.info("✅ No alerts generated.")
    else:
        st.warning("No data processed.")

if run_once:
    run_scan()

# ============================================================
# 🔁 10. Auto Scan Loop (Not recommended for Streamlit Cloud)
# ============================================================
if st.checkbox("Enable auto scan (testing only, not 24/7)"):
    st.info("Running auto-scan loop...")
    while True:
        run_scan()
        time.sleep(auto_interval)
