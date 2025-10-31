# app.py — Indian Stock Agent (Final working base)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io, os, time, requests
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

def get_secret_section(key: str, section: Optional[str] = None, default=None):
    try:
        if section and section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        pass
    return get_secret(key, default)

GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_REPO = get_secret("GITHUB_REPO")
GITHUB_BRANCH = get_secret("GITHUB_BRANCH", "main")
GITHUB_FILE_PATH = get_secret("GITHUB_FILE_PATH", "watchlist.xlsx")

# Telegram secrets
TELEGRAM_TOKEN = get_secret_section("TELEGRAM_TOKEN", section="telegram")
CHAT_ID = get_secret_section("CHAT_ID", section="telegram")

# -----------------------
# GitHub REST Connection
# -----------------------
def github_raw_headers():
    auth_scheme = "Bearer" if str(GITHUB_TOKEN).startswith("github_pat_") else "token"
    return {
        "Authorization": f"{auth_scheme} {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "streamlit-indian-stock-agent"
    }

@st.cache_data(ttl=120)
def load_excel_from_github():
    if not (GITHUB_TOKEN and GITHUB_REPO):
        st.error("Missing GitHub credentials")
        return pd.DataFrame()
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}?ref={GITHUB_BRANCH}"
        r = requests.get(url, headers=github_raw_headers(), timeout=10)
        if r.status_code == 200:
            df = pd.read_excel(io.BytesIO(r.content))
            return df
        else:
            st.error(f"GitHub fetch failed: {r.status_code}")
    except Exception as e:
        st.error(f"Error loading from GitHub: {e}")
    return pd.DataFrame()

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

st.sidebar.header("Settings")

if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.warning("⚠️ Telegram not configured")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.error("❌ GitHub credentials missing")

# -----------------------
# Load Watchlist
# -----------------------
use_uploaded = False
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        if "Symbol" not in df.columns:
            st.sidebar.error("Uploaded file must have 'Symbol' column")
        else:
            watchlist_df = df
            use_uploaded = True
            st.sidebar.success(f"✅ Using uploaded watchlist ({len(df)} symbols)")
    except Exception as e:
        st.sidebar.error(f"Error reading upload: {e}")

if not use_uploaded:
    watchlist_df = load_excel_from_github()
    st.sidebar.info("Using GitHub watchlist as default source")

# -----------------------
# Telegram Helper
# -----------------------
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=10
        )
    except Exception as e:
        st.warning(f"Telegram error: {e}")

# -----------------------
# RSI & EMA
# -----------------------
def calc_rsi_ema(symbol: str, period="2y"):
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        # ensure 1-D Close
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.Series(close, name="Close").astype(float)

        df["EMA200"] = close.ewm(span=min(200, len(close)), adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = 100 - (100 / (1 + rs))

        df["52W_High"] = close.rolling(252, min_periods=1).max()
        df["52W_Low"] = close.rolling(252, min_periods=1).min()

        return df
    except Exception as e:
        st.error(f"{symbol}: {e}")
        return None

# -----------------------
# Analyzer
# -----------------------
def analyze(symbol: str):
    df = calc_rsi_ema(symbol)
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    cmp_ = float(last["Close"])
    ema = float(last["EMA200"])
    rsi = float(last["RSI14"])
    low = float(last["52W_Low"])
    high = float(last["52W_High"])

    signal = "Neutral"
    if cmp_ > ema and rsi < 30:
        signal = "BUY"
    elif cmp_ < ema and rsi > 70:
        signal = "SELL"

    return {
        "Symbol": symbol,
        "CMP": round(cmp_, 2),
        "52W_Low": round(low, 2),
        "52W_High": round(high, 2),
        "EMA200": round(ema, 2),
        "RSI14": round(rsi, 2),
        "Signal": signal
    }

# -----------------------
# Main UI
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

if watchlist_df.empty or "Symbol" not in watchlist_df.columns:
    st.warning("⚠️ No valid 'Symbol' column found")
else:
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()

    # Combined table placeholder at top
    summary_placeholder = st.empty()
    last_scan_time = st.empty()

    st.subheader("⚙️ Controls")
    run_now = st.button("Run Scan Now", key="run_now")
    auto = st.checkbox("Enable Auto-scan (local only)", key="auto_chk")
    interval = st.number_input("Interval (sec)", value=60, step=5, min_value=5, key="interval_input")

    st.write("Status:")
    st.write(f"- GitHub Repo: {GITHUB_REPO or 'N/A'}")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    st.caption(f"yfinance version: {yf.__version__}")

    # Run scan
    def run_scan():
        results, alerts = [], []
        for s in symbols:
            r = analyze(s)
            if r:
                results.append(r)
        if results:
            df = pd.DataFrame(results)
            summary_placeholder.dataframe(df, use_container_width=True, hide_index=True)
            last_scan_time.caption(f"Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            summary_placeholder.warning("No data fetched")

    if run_now:
        run_scan()

    try:
        from streamlit_autorefresh import st_autorefresh
        if auto:
            st_autorefresh(interval=int(interval)*1000, key="autorefresh")
            st.info(f"🔁 Auto-scan active — every {interval} seconds")
            run_scan()
    except Exception:
        st.info("Optional: pip install streamlit-autorefresh for background scans")
