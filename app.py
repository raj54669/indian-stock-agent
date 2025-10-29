import streamlit as st
import pandas as pd
import yfinance as yf
import time
import io
from github import Github
import requests

# ----------------------------
# 🔐 Load credentials from Streamlit secrets
# ----------------------------
try:
    TELEGRAM_TOKEN = st.secrets["telegram"]["TELEGRAM_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]

    GITHUB_TOKEN = st.secrets["github"]["GITHUB_TOKEN"]
    GITHUB_REPO_NAME = st.secrets["github"]["GITHUB_REPO"]
    GITHUB_BRANCH = st.secrets["github"].get("GITHUB_BRANCH", "main")
    GITHUB_FILE_PATH = st.secrets["github"].get("GITHUB_FILE_PATH", "watchlist.xlsx")

    all_credentials_loaded = True
except Exception as e:
    all_credentials_loaded = False
    st.error(f"Error loading credentials: {e}")

# ----------------------------
# 🎨 App layout
# ----------------------------
st.set_page_config(page_title="Indian Stock Auto Tracker", layout="wide")

st.sidebar.header("⚙️ Settings")

if all_credentials_loaded:
    st.sidebar.success("All credentials loaded from app secrets.")
else:
    st.sidebar.error("Missing credentials — please check Streamlit secrets!")

# ----------------------------
# 🧠 GitHub Helper Functions
# ----------------------------
def init_github_repo():
    """Initialize GitHub connection"""
    try:
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO_NAME)
        return repo
    except Exception as e:
        st.warning(f"⚠️ GitHub repo init failed: {e}")
        return None


def load_excel_from_github(repo):
    """Load the watchlist Excel file from GitHub"""
    try:
        file_content = repo.get_contents(GITHUB_FILE_PATH, ref=GITHUB_BRANCH)
        file_bytes = file_content.decoded_content
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df
    except Exception as e:
        st.warning(f"⚠️ Error loading from GitHub: {e}")
        return None


def upload_excel_to_github(repo, file_bytes):
    """Replace the existing Excel file in GitHub with a new one"""
    try:
        contents = repo.get_contents(GITHUB_FILE_PATH, ref=GITHUB_BRANCH)
        repo.update_file(
            path=GITHUB_FILE_PATH,
            message="Updated watchlist via Streamlit app",
            content=file_bytes,
            sha=contents.sha,
            branch=GITHUB_BRANCH
        )
        st.success("✅ Uploaded new file to GitHub successfully.")
    except Exception as e:
        st.error(f"GitHub upload failed: {e}")

# ----------------------------
# 🤖 Telegram Helper
# ----------------------------
def send_telegram_message(message: str):
    """Send a Telegram message"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, data=data)
    except Exception as e:
        st.warning(f"Telegram send failed: {e}")

# ----------------------------
# 📈 Stock Analysis Functions
# ----------------------------
def fetch_stock_data(symbol):
    """Fetch stock data and calculate RSI & EMA"""
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if df.empty:
            return None
        df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df
    except Exception as e:
        st.warning(f"⚠️ Error fetching data for {symbol}: {e}")
        return None


def check_conditions(df):
    """Check if RSI and EMA conditions are met"""
    latest = df.iloc[-1]
    price = latest["Close"]
    ema200 = latest["EMA200"]
    rsi = latest["RSI"]

    near_ema = ema200 * 0.98 <= price <= ema200 * 1.02
    rsi_ok = 30 <= rsi <= 40

    return near_ema and rsi_ok, price, ema200, rsi

# ----------------------------
# 🚀 Main App
# ----------------------------
st.title("🇮🇳 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")
st.markdown(
    """
    Automatically tracks Indian stocks using **RSI (30–40)** and **200-day EMA (±2%)**.  
    Updates every minute and sends Telegram alerts when both conditions meet.
    """
)

repo = init_github_repo()

# ---- Load from GitHub first ----
df_watchlist = load_excel_from_github(repo) if repo else None

# ---- Sidebar Upload Option ----
st.sidebar.subheader("📤 Upload new watchlist (.xlsx)")
uploaded_file = st.sidebar.file_uploader(
    "Drag and drop file here", type=["xlsx"], label_visibility="collapsed"
)

if uploaded_file and repo:
    upload_excel_to_github(repo, uploaded_file.getvalue())
    df_watchlist = pd.read_excel(uploaded_file)

# ---- Validate Excel ----
if df_watchlist is None or "Symbol" not in df_watchlist.columns:
    st.warning("⚠️ No valid Excel found. Please upload one with a 'Symbol' column.")
    st.stop()

st.success(f"Loaded {len(df_watchlist)} stocks. Tracking started...")

# ----------------------------
# 🧾 Dashboard Table
# ----------------------------
results = []
for symbol in df_watchlist["Symbol"]:
    data = fetch_stock_data(symbol)
    if data is not None:
        condition_met, price, ema, rsi = check_conditions(data)
        results.append({
            "Symbol": symbol,
            "Price": round(price, 2),
            "EMA200": round(ema, 2),
            "RSI": round(rsi, 2),
            "Alert": "✅ YES" if condition_met else "—"
        })
        if condition_met:
            send_telegram_message(f"📊 {symbol}: Price near EMA200 ({ema:.2f}), RSI={rsi:.1f}")

if results:
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
else:
    st.warning("No stock data retrieved.")

# ----------------------------
# 🔁 Auto-refresh
# ----------------------------
auto = st.sidebar.checkbox("Run Auto Tracking (updates every 1 minute)")

if auto:
    with st.spinner("Tracking live..."):
        time.sleep(60)
        st.rerun()
