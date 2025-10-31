# ==========================================================
# app.py — Indian Stock Agent (Final Stable Build)
# ==========================================================

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
# Sidebar
# -----------------------
st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

st.sidebar.header("Settings")
if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.warning("⚠️ Telegram not set – alerts disabled")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.error("GitHub credentials missing")

# -----------------------
# GitHub Excel Loaders
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
# Save Excel to GitHub
# -----------------------
def save_excel_to_github(df, message="Update watchlist"):
    try:
        import base64
        owner, repo = GITHUB_REPO.split("/", 1)
        get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}"
        get_resp = requests.get(get_url, headers=github_headers(), timeout=10)
        sha = None
        if get_resp.status_code == 200:
            sha = get_resp.json().get("sha")

        bytes_buf = io.BytesIO()
        df.to_excel(bytes_buf, index=False)
        encoded_content = base64.b64encode(bytes_buf.getvalue()).decode("utf-8")

        data = {"message": message, "branch": GITHUB_BRANCH, "content": encoded_content}
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
# Load Watchlist
# -----------------------
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
# UI Header
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

if watchlist_df.empty or "Symbol" not in [c.strip() for c in watchlist_df.columns]:
    st.warning("⚠️ No valid 'Symbol' column found in watchlist.")
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

# ==========================================================
# 🔍 Stock Indicators
# ==========================================================
def calc_rsi_ema(symbol: str, period="2y"):
    """Fetch daily stock data and compute EMA200, RSI14, and 52W High/Low"""
    import numpy as np
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        close = df["Close"].astype(float)

        # EMA200
        df["EMA200"] = close.ewm(span=min(200, len(close)), adjust=False).mean()

        # RSI14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

        # 52-week High/Low
        df["52W_High"] = close.rolling(window=252, min_periods=1).max()
        df["52W_Low"] = close.rolling(window=252, min_periods=1).min()

        return df
    except Exception as e:
        st.error(f"Error in calc_rsi_ema for {symbol}: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None

# ==========================================================
# 📈 Analyze One Symbol
# ==========================================================
def analyze(symbol: str):
    try:
        df = calc_rsi_ema(symbol)
        if df is None or df.empty:
            return None

        last = df.iloc[-1]
        cmp_ = float(last["Close"])
        ema200 = float(last["EMA200"])
        rsi14 = float(last["RSI14"])
        low52 = float(last["52W_Low"])
        high52 = float(last["52W_High"])

        signal = "Neutral"
        if cmp_ > ema200 and rsi14 < 30:
            signal = "BUY"
        elif cmp_ < ema200 and rsi14 > 70:
            signal = "SELL"

        return {
            "Symbol": symbol,
            "CMP": round(cmp_, 2),
            "52W_Low": round(low52, 2),
            "52W_High": round(high52, 2),
            "EMA200": round(ema200, 2),
            "RSI14": round(rsi14, 2),
            "Signal": signal,
        }

    except Exception as e:
        st.error(f"analyze() error for {symbol}: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None

# ==========================================================
# ⚙️ Controls + Placeholder for Combined Table
# ==========================================================
unified_table_placeholder = st.empty()

st.subheader("⚙️ Controls")
col1, col2 = st.columns([1, 2])
with col1:
    run_now = st.button("Run Scan Now", key="run_now_btn")
    auto = st.checkbox("Enable Auto-scan (local only)", key="auto_chk")
    interval = st.number_input("Interval (sec)", value=60, step=5, min_value=5, key="interval_input")
with col2:
    st.write("Status:")
    st.write(f"- GitHub Repo: {GITHUB_REPO or 'N/A'}")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    try:
        st.caption(f"yfinance version: {yf.__version__}")
    except Exception:
        pass

# ==========================================================
# 🚀 Unified Scan Function
# ==========================================================
def run_scan_once():
    if watchlist_df is None or "Symbol" not in watchlist_df.columns:
        st.error("No watchlist available")
        return [], []

    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()
    results, alerts, debug_lines = [], [], []

    with st.spinner(f"🔍 Scanning {len(symbols)} symbols..."):
        for s in symbols:
            debug_lines.append(f"Processing {s} ...")
            r = analyze(s)
            if r:
                results.append(r)
                debug_lines.append(f"✅ {s}: CMP={r['CMP']} EMA200={r['EMA200']} RSI={r['RSI14']}")
                if r["Signal"] in ("BUY", "SELL"):
                    alerts.append(f"{s}: {r['Signal']} (CMP={r['CMP']}, EMA200={r['EMA200']}, RSI={r['RSI14']})")
            else:
                debug_lines.append(f"⚠️ No result for {s}")
            time.sleep(0.25)

    # Render unified table at top
    unified_table_placeholder.empty()
    unified_table_placeholder.subheader("📊 Combined Summary Table")

    if results:
        df_result = pd.DataFrame(results)
        unified_table_placeholder.dataframe(
            df_result[["Symbol", "CMP", "52W_Low", "52W_High", "EMA200", "RSI14", "Signal"]],
            use_container_width=True,
            hide_index=True,
        )
        unified_table_placeholder.caption(f"Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        unified_table_placeholder.warning("⚠️ No valid results from scan")

    # Alerts
    if alerts:
        msg = "⚡ Stock Alerts:\n" + "\n".join(alerts)
        st.warning(msg)
        if TELEGRAM_TOKEN and CHAT_ID:
            send_telegram(msg)

    # Debug expander
    if debug_lines:
        with st.expander("🔍 Debug details (click to expand)"):
            for l in debug_lines:
                st.text(l)

    return results, alerts

# ==========================================================
# 🔁 Run / Auto-refresh Logic
# ==========================================================
if run_now:
    run_scan_once()

try:
    from streamlit_autorefresh import st_autorefresh
    if auto:
        st_autorefresh(interval=int(interval) * 1000, key="autorefresh")
        st.info(f"🔁 Auto-scan active — every {interval} seconds")
        run_scan_once()
except Exception:
    st.info("Optional: install streamlit-autorefresh for background scans: pip install streamlit-autorefresh")

# ==========================================================
# End of File
# ==========================================================
