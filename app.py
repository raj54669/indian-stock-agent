# app.py — Indian Stock Agent (final optimized version)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io, os, time, requests
from datetime import datetime, timedelta, timezone
from typing import Optional

# Optional auto-refresh support
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# -----------------------
# Streamlit Config
# -----------------------
st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Alert Bot", layout="wide")
st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# -----------------------
# Constants
# -----------------------
EMA_SPAN = 200
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
WATCH_RANGE = (30, 40)
CMP_EMA_TOLERANCE = 0.02  # ±2%

# -----------------------
# Session State
# -----------------------
if "alert_history" not in st.session_state:
    st.session_state.alert_history = pd.DataFrame(
        columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
    )
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

# -----------------------
# Secrets helpers
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
TELEGRAM_TOKEN = get_secret_section("TELEGRAM_TOKEN", section="telegram")
CHAT_ID = get_secret_section("CHAT_ID", section="telegram")

# -----------------------
# GitHub & Watchlist
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
        return pd.DataFrame()
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}?ref={GITHUB_BRANCH}"
        r = requests.get(url, headers=github_raw_headers(), timeout=10)
        if r.status_code == 200:
            return pd.read_excel(io.BytesIO(r.content))
    except Exception as e:
        st.error(f"GitHub load error: {e}")
    return pd.DataFrame()

# Sidebar
st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])
watchlist_df = pd.read_excel(uploaded_file) if uploaded_file else load_excel_from_github()

if not watchlist_df.empty and "Symbol" in watchlist_df.columns:
    watchlist_df["Symbol"] = watchlist_df["Symbol"].astype(str).str.strip()
else:
    watchlist_df = pd.DataFrame(columns=["Symbol"])

# -----------------------
# Telegram
# -----------------------
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
        return r.status_code == 200
    except Exception as e:
        st.warning(f"Telegram send error: {e}")
        return False

# -----------------------
# Indicator Calculations
# -----------------------
def calc_rsi_ema(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df = df.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    df["EMA200"] = df["Close"].ewm(span=EMA_SPAN, adjust=False, min_periods=1).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
    last_date = df.index.max()
    cutoff = last_date - timedelta(days=365)
    df1y = df[df.index >= cutoff]
    df["52W_High"] = df1y["Close"].max()
    df["52W_Low"] = df1y["Close"].min()
    return df

# -----------------------
# Analysis
# -----------------------
def analyze(symbol: str):
    try:
        st.session_state.debug_logs.append(f"Fetching {symbol}...")
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            st.session_state.debug_logs.append(f"No data for {symbol}")
            return None
        df = calc_rsi_ema(df)
        last = df.iloc[-1]
        cmp_, ema200, rsi14 = last["Close"], last["EMA200"], last["RSI14"]
        low52, high52 = last["52W_Low"], last["52W_High"]

        # Determine Signal
        signal, cond = "Neutral", ""
        if abs(cmp_ - ema200) / cmp_ <= CMP_EMA_TOLERANCE and WATCH_RANGE[0] <= rsi14 <= WATCH_RANGE[1]:
            signal, cond = "🟡 WATCH", "EMA200 within ±2% & RSI 30–40"
        elif rsi14 < RSI_OVERSOLD and cmp_ > ema200:
            signal, cond = "🟢 BUY", "RSI < 30 and CMP > EMA200"
        elif rsi14 > RSI_OVERBOUGHT and cmp_ < ema200:
            signal, cond = "🔴 SELL", "RSI > 70 and CMP < EMA200"

        # Trend Confirmation (EMA crossover)
        trend_note = ""
        if cmp_ > ema200:
            trend_note = "📈 Uptrend (CMP > EMA200)"
        elif cmp_ < ema200:
            trend_note = "📉 Downtrend (CMP < EMA200)"
        else:
            trend_note = "➖ Sideways (CMP ≈ EMA200)"

        # Telegram Message
        telegram_msg = (
            f"⚡ Alert: {signal}\n"
            f"*{symbol}*\n"
            f"CMP = {cmp_:.2f}\n"
            f"EMA200 = {ema200:.2f}\n"
            f"RSI14 = {rsi14:.2f}\n"
            f"Condition: {cond}\n"
            f"{trend_note}"
        )

        return {
            "Symbol": symbol,
            "CMP": round(cmp_, 2),
            "52W_Low": round(low52, 2),
            "52W_High": round(high52, 2),
            "EMA200": round(ema200, 2),
            "RSI14": round(rsi14, 2),
            "Signal": signal,
            "Condition": cond,
            "TelegramMessage": telegram_msg
        }
    except Exception as e:
        st.session_state.debug_logs.append(f"Error {symbol}: {e}")
        return None

# -----------------------
# Run Scan
# -----------------------
def run_scan():
    results = []
    total = len(symbols)
    progress = st.progress(0, text="Starting scan...")
    for i, symbol in enumerate(symbols, 1):
        data = analyze(symbol)
        if data:
            results.append(data)
            # send telegram if active signal
            if data["Signal"] != "Neutral":
                add_to_alert_history(symbol, data["Signal"], data["CMP"], data["EMA200"], data["RSI14"])
                send_telegram(data["TelegramMessage"])
        progress.progress(i / total, text=f"Scanning {i}/{total}: {symbol}")
    progress.empty()
    if results:
        df = pd.DataFrame(results)
        df = df[["Symbol", "CMP", "52W_Low", "52W_High", "EMA200", "RSI14", "Signal"]]
        st.session_state.summary_df = df
        summary_placeholder.dataframe(df, use_container_width=True, hide_index=True)
        ist = timezone(timedelta(hours=5, minutes=30))
        last_scan_time.caption(f"Last scan: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        summary_placeholder.warning("No valid data fetched.")

# -----------------------
# Add / Clear History
# -----------------------
def add_to_alert_history(symbol, signal, cmp_, ema200, rsi14):
    ts = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
    new = pd.DataFrame([{
        "Date & Time (IST)": ts, "Symbol": symbol, "Signal": signal,
        "CMP": cmp_, "EMA200": ema200, "RSI14": rsi14
    }])
    st.session_state.alert_history = pd.concat([st.session_state.alert_history, new], ignore_index=True)

# -----------------------
# UI
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")
symbols = watchlist_df["Symbol"].tolist()
summary_placeholder = st.empty()
last_scan_time = st.caption("Ready to scan your watchlist...")

col1, col2 = st.columns([1, 2])
with col1:
    run_now = st.button("Run Scan Now")
    interval = st.number_input("Interval (sec)", value=60, step=5, min_value=5)
    auto = st.checkbox("Enable Auto-scan")
with col2:
    st.markdown("**Status:**")
    st.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
    st.write(f"- Telegram: {'✅' if TELEGRAM_TOKEN else '❌'}")
    if auto:
        st.info(f"🔁 Auto-scan active every {interval} sec")

# -----------------------
# Alert History
# -----------------------
st.subheader("📜 Alert History")
if not st.session_state.alert_history.empty:
    st.dataframe(st.session_state.alert_history, use_container_width=True, hide_index=True)
    if st.button("🧹 Confirm Clear History"):
        st.session_state.alert_history = st.session_state.alert_history.iloc[0:0]
        st.success("✅ History cleared.")
        st.experimental_rerun()
else:
    st.info("No alerts recorded yet. Run a scan to generate new alerts.")

# -----------------------
# Debug Logs
# -----------------------
with st.expander("🪵 Debug Logs"):
    if st.session_state.debug_logs:
        st.text("\n".join(st.session_state.debug_logs[-100:]))
    else:
        st.info("No debug logs yet.")

# -----------------------
# Actions
# -----------------------
if run_now:
    run_scan()

if st.button("📨 Send Test Telegram Alert"):
    msg = "✅ Test Alert from Indian Stock Agent"
    if send_telegram(msg):
        st.success("Test Telegram sent successfully!")
    else:
        st.error("Telegram send failed. Check your credentials.")

# -----------------------
# Auto-refresh
# -----------------------
if auto and st_autorefresh:
    st_autorefresh(interval=int(interval) * 1000, key="auto_refresh")
    run_scan()
