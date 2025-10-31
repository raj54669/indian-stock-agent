import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
import time
import os
from datetime import datetime
import numpy as np

st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Alert Bot", layout="wide")

# --- Load secrets ---
def get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)

GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_REPO = get_secret("GITHUB_REPO")
GITHUB_FILE_PATH = get_secret("GITHUB_FILE_PATH", "watchlist.xlsx")
TELEGRAM_TOKEN = get_secret("TELEGRAM_TOKEN")
CHAT_ID = get_secret("CHAT_ID")

# --- Sidebar ---
st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.warning("⚠️ Telegram not configured")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.error("⚠️ Missing GitHub credentials")

# --- GitHub Helpers ---
def github_raw_headers():
    scheme = "Bearer" if str(GITHUB_TOKEN).startswith("github_pat_") else "token"
    return {"Authorization": f"{scheme} {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.raw"}

@st.cache_data(ttl=120)
def load_excel_from_github():
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}"
        r = requests.get(url, headers=github_raw_headers(), timeout=10)
        if r.status_code == 200:
            return pd.read_excel(io.BytesIO(r.content))
        else:
            st.warning(f"GitHub fetch failed: {r.status_code}")
    except Exception as e:
        st.error(f"GitHub load error: {e}")
    return pd.DataFrame()

# --- Load Watchlist ---
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success(f"✅ Uploaded {len(df)} symbols")
else:
    df = load_excel_from_github()
    st.sidebar.info("Using GitHub watchlist as default source")

st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

if df.empty or "Symbol" not in df.columns:
    st.warning("⚠️ No valid 'Symbol' column found in watchlist")
    st.stop()

symbols = df["Symbol"].dropna().tolist()

# --- RSI + EMA calculation ---
def calc_rsi_ema(symbol):
    try:
        data = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            return None
        close = data["Close"].astype(float)

        data["EMA200"] = close.ewm(span=min(200, len(close)), adjust=False).mean()

        delta = close.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        data["RSI14"] = 100 - (100 / (1 + rs))

        data["52W_High"] = close.rolling(window=252, min_periods=1).max()
        data["52W_Low"] = close.rolling(window=252, min_periods=1).min()

        return data
    except Exception as e:
        st.error(f"{symbol}: {e}")
        return None

def analyze(symbol):
    data = calc_rsi_ema(symbol)
    if data is None or data.empty:
        return None
    last = data.iloc[-1]
    cmp_ = last["Close"]
    ema200 = last["EMA200"]
    rsi14 = last["RSI14"]
    low52 = last["52W_Low"]
    high52 = last["52W_High"]

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
        "Signal": signal
    }

# --- Controls ---
st.subheader("⚙️ Controls")
col1, col2 = st.columns([1, 2])
with col1:
    run_now = st.button("Run Scan Now", key="scan_btn")
    auto = st.checkbox("Enable Auto-scan (local only)", key="auto_scan")
    interval = st.number_input("Interval (sec)", value=60, step=10, min_value=10)
with col2:
    st.write("Status:")
    st.write(f"- GitHub Repo: {GITHUB_REPO}")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    st.caption(f"yfinance version: {yf.__version__}")

# --- Scan Function ---
def run_scan_once():
    results, debug = [], []
    for s in symbols:
        debug.append(f"Processing {s} ...")
        r = analyze(s)
        if r:
            results.append(r)
            debug.append(f"✅ {s}: CMP={r['CMP']} EMA200={r['EMA200']} RSI={r['RSI14']}")
        else:
            debug.append(f"❌ No data for {s}")
        time.sleep(0.3)

    # Display Combined Summary Table
    st.markdown("### 📊 Combined Summary Table")
    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(
            res_df[["Symbol", "CMP", "52W_Low", "52W_High", "EMA200", "RSI14", "Signal"]],
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("No valid data found")

    # Debug section
    with st.expander("🔍 Debug details (click to expand)"):
        for line in debug:
            st.text(line)

if run_now:
    run_scan_once()

try:
    from streamlit_autorefresh import st_autorefresh
    if auto:
        st_autorefresh(interval=int(interval) * 1000, key="auto_refresh")
        st.info(f"🔁 Auto-scan active every {interval} sec")
        run_scan_once()
except Exception:
    st.info("Optional: pip install streamlit-autorefresh")
