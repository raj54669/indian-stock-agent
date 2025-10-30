# app.py – Indian Stock Auto Tracker with Nextbite-style GitHub connection

import streamlit as st
import pandas as pd
import yfinance as yf
import io
import requests
import time
import os
from typing import Optional
from datetime import datetime

# Optional PyGithub
try:
    from github import Github
    HAS_PYGITHUB = True
except Exception:
    HAS_PYGITHUB = False

# -----------------------
# Streamlit Config
# -----------------------
st.set_page_config(page_title="📈 Indian Stock Auto Tracker", layout="wide")

# -----------------------
# Unified GitHub Secrets Loader (Nextbite style)
# -----------------------
def get_secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)

GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_REPO_NAME = get_secret("GITHUB_REPO")
GITHUB_BRANCH = get_secret("GITHUB_BRANCH", "main")
GITHUB_FILE_PATH = get_secret("GITHUB_FILE_PATH", "watchlist.xlsx")

# -----------------------
# Debug & Connection Diagnostics
# -----------------------
st.sidebar.header("🔍 GitHub Diagnostics")

# Show token length safely (no token content exposed)
st.sidebar.write(f"Token length: {len(str(GITHUB_TOKEN)) if GITHUB_TOKEN else 'None'}")
st.sidebar.write(f"Repo: {GITHUB_REPO_NAME or '❌ Not set'}")
st.sidebar.write(f"Branch: {GITHUB_BRANCH}")
st.sidebar.write(f"File path: {GITHUB_FILE_PATH}")

# --- Direct REST test to GitHub ---
try:
    test = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}"},
        timeout=8,
    )
    st.sidebar.write(f"Token test status: {test.status_code}")
    if test.status_code != 200:
        st.sidebar.write(test.json())
    else:
        j = test.json()
        st.sidebar.success(f"Authenticated as: {j.get('login', 'Unknown')}")
except Exception as e:
    st.sidebar.error(f"Token check failed: {e}")

# --- Try PyGithub connection ---
GITHUB_REPO = None
if GITHUB_TOKEN and GITHUB_REPO_NAME and HAS_PYGITHUB:
    try:
        gh = Github(GITHUB_TOKEN)
        user = gh.get_user().login
        st.sidebar.info(f"PyGithub Authenticated as: {user}")

        GITHUB_REPO = gh.get_repo(GITHUB_REPO_NAME)
        st.sidebar.success(f"✅ Connected to repo: {GITHUB_REPO_NAME}")
    except Exception as e:
        st.sidebar.error(f"⚠️ PyGithub connection failed: {e}")
else:
    st.sidebar.warning("⚠️ GitHub token missing or PyGithub not installed")

# -----------------------
# Telegram Secrets
# -----------------------
def get_secret_section(key: str, section: Optional[str] = None, default=None):
    try:
        if section and section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        pass
    return get_secret(key, default)

TELEGRAM_TOKEN = get_secret_section("TELEGRAM_TOKEN", section="telegram")
CHAT_ID = get_secret_section("CHAT_ID", section="telegram")

# -----------------------
# Sidebar Status
# -----------------------
st.sidebar.header("Settings")
if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("Telegram configured ✅")
else:
    st.sidebar.warning("Telegram not set — alerts disabled")

if GITHUB_TOKEN and GITHUB_REPO_NAME:
    st.sidebar.info("GitHub secrets present")
else:
    st.sidebar.error("GitHub credentials missing")

# -----------------------
# GitHub Helpers
# -----------------------
def load_excel_from_github(repo, path, branch="main"):
    try:
        file = repo.get_contents(path, ref=branch)
        df = pd.read_excel(io.BytesIO(file.decoded_content))
        return df
    except Exception as e:
        st.error(f"Failed to load {path} from GitHub: {e}")
        return pd.DataFrame()

def save_excel_to_github(repo, path, df, branch="main", message="Update watchlist"):
    """Save updated Excel back to GitHub"""
    try:
        file = repo.get_contents(path, ref=branch)
        bytes_buf = io.BytesIO()
        df.to_excel(bytes_buf, index=False)
        repo.update_file(file.path, message, bytes_buf.getvalue(), file.sha, branch=branch)
        st.success("✅ Watchlist updated on GitHub!")
        st.cache_data.clear()
    except Exception as e:
        st.error(f"❌ Failed to save file: {e}")

# -----------------------
# Load Watchlist
# -----------------------
@st.cache_data(ttl=120)
def load_watchlist():
    if GITHUB_REPO:
        df = load_excel_from_github(GITHUB_REPO, GITHUB_FILE_PATH, branch=GITHUB_BRANCH)
        if not df.empty:
            return df
    # fallback to REST if PyGithub unavailable
    try:
        owner, repo = GITHUB_REPO_NAME.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}"
        auth_scheme = "Bearer" if str(GITHUB_TOKEN).startswith("github_pat_") else "token"
        headers = {"Authorization": f"{auth_scheme} {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.raw"}

        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return pd.read_excel(io.BytesIO(r.content))
    except Exception as e:
        st.error(f"Fallback load failed: {e}")
    return pd.DataFrame()

watchlist_df = load_watchlist()

# -----------------------
# UI
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

if watchlist_df.empty or "Symbol" not in [c.strip() for c in watchlist_df.columns]:
    st.warning(f"⚠️ No valid 'Symbol' column found in `{GITHUB_FILE_PATH}`.")
    st.markdown("Upload an Excel file to GitHub with a column named **Symbol**.")
    st.table(pd.DataFrame({"Symbol": ["RELIANCE.NS", "TCS.NS"]}))
else:
    st.success(f"✅ Loaded {len(watchlist_df)} symbols from GitHub")
    st.dataframe(watchlist_df.head(20))

# -----------------------
# Telegram Helper
# -----------------------
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        st.error("Telegram not configured")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=10)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Telegram error: {e}")
        return False

# -----------------------
# Stock Analysis
# -----------------------
def calc_rsi_ema(symbol: str):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
    except Exception as e:
        st.write(f"yfinance error for {symbol}: {e}")
        return None
    if df.empty:
        return None
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    delta = df["Close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = gain.rolling(14).mean(), loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def analyze(symbol):
    df = calc_rsi_ema(symbol)
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    signal = "Neutral"
    if last["Close"] > last["EMA200"] and last["RSI"] < 30:
        signal = "BUY"
    elif last["Close"] < last["EMA200"] and last["RSI"] > 70:
        signal = "SELL"
    return {
        "Symbol": symbol,
        "Close": round(last["Close"], 2),
        "EMA200": round(last["EMA200"], 2),
        "RSI": round(last["RSI"], 2),
        "Signal": signal
    }

# -----------------------
# Controls
# -----------------------
st.subheader("⚙️ Controls")
col1, col2 = st.columns([1, 2])
with col1:
    run_now = st.button("Run Scan Now")
    auto = st.checkbox("Enable Auto-scan (local only)")
    interval = st.number_input("Interval (sec)", value=60, step=10)
with col2:
    st.write("Status:")
    st.write(f"- GitHub Repo: {GITHUB_REPO_NAME or 'N/A'}")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")

def run_scan_once():
    if watchlist_df is None or "Symbol" not in watchlist_df.columns:
        st.error("No watchlist available")
        return
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()
    results, alerts = [], []
    with st.spinner(f"Scanning {len(symbols)} symbols..."):
        for s in symbols:
            r = analyze(s)
            if r:
                results.append(r)
                if r["Signal"] in ("BUY", "SELL"):
                    alerts.append(f"{r['Symbol']}: {r['Signal']} (RSI={r['RSI']}, Close={r['Close']})")

    if results:
        st.dataframe(pd.DataFrame(results))
    if alerts:
        st.warning("⚡ Alerts:\n" + "\n".join(alerts))
        send_telegram("\n".join(alerts))

    # Example: save scan summary to GitHub
    if GITHUB_REPO and results:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_path = f"scans/scan_{timestamp}.xlsx"
        try:
            bytes_buf = io.BytesIO()
            df.to_excel(bytes_buf, index=False)
            GITHUB_REPO.create_file(save_path, f"Add scan {timestamp}", bytes_buf.getvalue(), branch=GITHUB_BRANCH)
            st.success(f"📤 Saved scan results to GitHub: {save_path}")
        except Exception as e:
            st.warning(f"GitHub save skipped: {e}")

if run_now:
    run_scan_once()

if auto:
    st.warning("Auto-scan started (use cautiously)")
    while True:
        run_scan_once()
        time.sleep(max(5, int(interval)))
