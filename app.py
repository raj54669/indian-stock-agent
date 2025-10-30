# app.py – Indian Stock Agent (100% REST-based GitHub connection, Nextbite-style)

import streamlit as st
import pandas as pd
import yfinance as yf
import io
import requests
import time
import os
from datetime import datetime
from typing import Optional

# -----------------------
# Streamlit Config
# -----------------------
st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Alert Bot", layout="wide")

# -----------------------
# Unified Secrets Loader
# -----------------------
def get_secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)

GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_REPO = get_secret("GITHUB_REPO")
GITHUB_BRANCH = get_secret("GITHUB_BRANCH", "main")
GITHUB_FILE_PATH = get_secret("GITHUB_FILE_PATH", "watchlist.xlsx")

# Telegram settings
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
# GitHub REST Connection Helpers
# -----------------------
def github_headers():
    auth_scheme = "Bearer" if str(GITHUB_TOKEN).startswith("github_pat_") else "token"
    return {
        "Authorization": f"{auth_scheme} {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "streamlit-indian-stock-agent",
        "X-GitHub-Api-Version": "2022-11-28"
    }

def github_raw_headers():
    auth_scheme = "Bearer" if str(GITHUB_TOKEN).startswith("github_pat_") else "token"
    return {
        "Authorization": f"{auth_scheme} {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "streamlit-indian-stock-agent"
    }

# -----------------------
# Sidebar – Simplified
# -----------------------

st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

st.sidebar.header("Settings")

# Telegram connection indicator
if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.warning("⚠️ Telegram not set – alerts disabled")

# GitHub connection indicator
if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.error("GitHub credentials missing")


# -----------------------
# Load Excel file from GitHub (REST)
# -----------------------
@st.cache_data(ttl=120)
def load_excel_from_github():
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}?ref={GITHUB_BRANCH}"
        r = requests.get(url, headers=github_raw_headers(), timeout=10)
        if r.status_code == 200:
            df = pd.read_excel(io.BytesIO(r.content))
            return df
        else:
            st.warning(f"GitHub file fetch failed: {r.status_code} – {r.text}")
    except Exception as e:
        st.error(f"Error loading from GitHub: {e}")
    return pd.DataFrame()

# -----------------------
# Save Excel to GitHub (REST)
# -----------------------
def save_excel_to_github(df, message="Update watchlist"):
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}"
        get_resp = requests.get(get_url, headers=github_headers(), timeout=10)
        sha = None
        if get_resp.status_code == 200:
            sha = get_resp.json().get("sha")

        bytes_buf = io.BytesIO()
        df.to_excel(bytes_buf, index=False)
        encoded_content = bytes_buf.getvalue()

        import base64
        data = {
            "message": message,
            "branch": GITHUB_BRANCH,
            "content": base64.b64encode(encoded_content).decode("utf-8"),
        }
        if sha:
            data["sha"] = sha

        put_resp = requests.put(get_url, headers=github_headers(), json=data, timeout=10)
        if put_resp.status_code in (200, 201):
            st.success("✅ File successfully saved to GitHub!")
        else:
            st.error(f"GitHub save failed: {put_resp.status_code} – {put_resp.text}")
    except Exception as e:
        st.error(f"Error saving to GitHub: {e}")

# -----------------------
# Load Watchlist (with upload override)
# -----------------------

# Upload file option to override GitHub watchlist
use_uploaded = False

if uploaded_file is not None:
    try:
        uploaded_watchlist = pd.read_excel(uploaded_file)
        if "Symbol" not in uploaded_watchlist.columns:
            st.sidebar.error("Uploaded file must contain a 'Symbol' column.")
        else:
            watchlist_df = uploaded_watchlist
            use_uploaded = True
            st.sidebar.success(f"✅ Using uploaded watchlist ({len(watchlist_df)} symbols)")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")

if not use_uploaded:
    watchlist_df = load_excel_from_github()
    st.sidebar.info("Using GitHub watchlist as default source")

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
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": message},
            timeout=10,
        )
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
        if df is None or df.empty:
            st.warning(f"⚠️ No data returned for {symbol}")
            return None

        if "Close" not in df.columns:
            st.warning(f"⚠️ Missing Close column for {symbol}")
            return None

        df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
        delta = df["Close"].diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        avg_gain, avg_loss = gain.rolling(14).mean(), loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df
    except Exception as e:
        st.error(f"yfinance error for {symbol}: {e}")
        return None

def analyze(symbol):
    df = calc_rsi_ema(symbol)
    if df is None or df.empty:
        return None

    # Get the latest single row safely
    last = df.iloc[-1].to_dict()

    close = float(last.get("Close", 0))
    ema200 = float(last.get("EMA200", 0))
    rsi = float(last.get("RSI", 50))

    signal = "Neutral"
    if close > ema200 and rsi < 30:
        signal = "BUY"
    elif close < ema200 and rsi > 70:
        signal = "SELL"

    return {
        "Symbol": symbol,
        "Close": round(close, 2),
        "EMA200": round(ema200, 2),
        "RSI": round(rsi, 2),
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
    st.write(f"- GitHub Repo: {GITHUB_REPO or 'N/A'}")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")

# -----------------------
# Main Scan Logic
# -----------------------
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
    if results:
        df = pd.DataFrame(results)
        st.success("✅ Scan complete ")
        st.dataframe(df)

if run_now:
    run_scan_once()

if auto:
    st.warning("Auto-scan started (use cautiously)")
    while True:
        run_scan_once()
        time.sleep(max(5, int(interval)))
