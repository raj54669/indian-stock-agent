# app.py — Indian Stock Agent (Final Unified Version)
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
# Initialize alert history
# -----------------------
if "alert_history" not in st.session_state:
    st.session_state.alert_history = pd.DataFrame(
        columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
    )

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
# GitHub headers
# -----------------------
def github_raw_headers():
    auth_scheme = "Bearer" if str(GITHUB_TOKEN).startswith("github_pat_") else "token"
    return {
        "Authorization": f"{auth_scheme} {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "streamlit-indian-stock-agent"
    }

# -----------------------
# Load Excel from GitHub (live fetch, no cache)
# -----------------------
def load_excel_from_github():
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return pd.DataFrame()
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}?ref={GITHUB_BRANCH}"
        r = requests.get(url, headers=github_raw_headers(), timeout=10)
        if r.status_code == 200:
            df = pd.read_excel(io.BytesIO(r.content))
            return df
        else:
            st.error(f"GitHub fetch failed: {r.status_code} - {r.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading from GitHub: {e}")
        return pd.DataFrame()

# -----------------------
# Sidebar: upload + settings
# -----------------------
st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

watchlist_df = pd.DataFrame()
use_uploaded = False

if uploaded_file is not None:
    try:
        df_up = pd.read_excel(uploaded_file)
        if "Symbol" not in df_up.columns:
            st.sidebar.error("Uploaded file must contain a 'Symbol' column.")
        else:
            watchlist_df = df_up
            use_uploaded = True
            st.sidebar.success(f"✅ Using uploaded watchlist ({len(watchlist_df)} symbols)")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")

if not use_uploaded:
    watchlist_df = load_excel_from_github()
    st.sidebar.info("Using GitHub watchlist as default source")

st.sidebar.header("⚙️ Settings")

if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.info("Telegram not configured — alerts disabled")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.warning("⚠️ GitHub credentials missing")

try:
    st.sidebar.caption(f"📦 yfinance version: {yf.__version__}")
except Exception:
    pass

# -----------------------
# Telegram helper
# -----------------------
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": message},
            timeout=10
        )
        return r.status_code == 200
    except Exception as e:
        st.warning(f"Telegram send error: {e}")
        return False

# -----------------------
# Indicator Calculation (unchanged original logic)
# -----------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in c if x]).strip("_") for c in df.columns]
    return df

def _find_close_column(df: pd.DataFrame):
    for prefer in ("close", "adjclose", "adj_close", "adjusted_close"):
        for c in df.columns:
            if c.replace(" ", "").replace("_", "").lower() == prefer:
                return c
    for c in df.columns:
        if "close" in c.lower():
            return c
    return None

def calc_rsi_ema(df: pd.DataFrame):
    try:
        if df is None or df.empty:
            return None
        df = _flatten_columns(df)
        close_col = _find_close_column(df)
        if close_col is None:
            return None
        df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=["Close"])
        df.index = pd.to_datetime(df.index)
        df["EMA200"] = df["Close"].ewm(span=200, adjust=False, min_periods=1).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = 100 - (100 / (1 + rs))
        cutoff = df.index.max() - timedelta(days=365)
        df_1y = df[df.index >= cutoff]
        df["52W_High"] = df_1y["Close"].max() if not df_1y.empty else df["Close"].max()
        df["52W_Low"] = df_1y["Close"].min() if not df_1y.empty else df["Close"].min()
        return df
    except Exception:
        return None

# -----------------------
# Analyzer (unchanged original logic)
# -----------------------
def analyze(symbol: str):
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df_ind = calc_rsi_ema(df)
        if df_ind is None or df_ind.empty:
            return None
        last = df_ind.iloc[-1]
        cmp_ = float(last["Close"])
        ema200 = float(last["EMA200"])
        rsi14 = float(last["RSI14"])
        low52 = float(last["52W_Low"])
        high52 = float(last["52W_High"])
        signal = "Neutral"
        if (cmp_ * 0.98 <= ema200 <= cmp_ * 1.02) and (30 <= rsi14 <= 40):
            signal = "🟡 WATCH"
        elif rsi14 < 30:
            signal = "🟢 BUY"
        elif rsi14 > 70:
            signal = "🔴 SELL"
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
        st.error(f"{symbol}: {e}")
        return None

# -----------------------
# Add to Alert History
# -----------------------
IST = timezone(timedelta(hours=5, minutes=30))
def add_to_alert_history(symbol, signal, cmp_, ema200, rsi14):
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([{
        "Date & Time (IST)": ts,
        "Symbol": symbol,
        "Signal": signal,
        "CMP": round(float(cmp_), 2),
        "EMA200": round(float(ema200), 2),
        "RSI14": round(float(rsi14), 2),
    }])
    st.session_state.alert_history = pd.concat(
        [st.session_state.alert_history, new_row],
        ignore_index=True
    )

# -----------------------
# Main UI
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

if watchlist_df.empty or "Symbol" not in watchlist_df.columns:
    st.warning("⚠️ No valid 'Symbol' column found.")
    symbols = []
else:
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()

# Combined summary placeholder
st.subheader("📋 Combined Summary Table")
summary_placeholder = st.empty()
last_scan_time = st.caption("Waiting for scan...")

# -----------------------
# Controls
# -----------------------
st.subheader("⚙️ Controls")

col1, col2 = st.columns([1, 2])
with col1:
    run_now = st.button("Run Scan Now")
    interval = st.number_input("Interval (sec)", value=60, step=5, min_value=10)
    auto = st.checkbox("Enable Auto-scan", value=True)

with col2:
    st.markdown("**Status:**")
    st.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    if auto:
        st.write(f"🔁 Auto-scan active every {interval} sec")

# -----------------------
# Run Scan
# -----------------------
def run_scan():
    if not symbols:
        st.warning("⚠️ No symbols found.")
        return
    results = []
    with st.spinner("🔍 Scanning symbols..."):
        for symbol in symbols:
            data = analyze(symbol)
            if data:
                results.append(data)
                if data["Signal"] in ("🟢 BUY", "🔴 SELL", "🟡 WATCH"):
                    add_to_alert_history(
                        data["Symbol"], data["Signal"], data["CMP"], data["EMA200"], data["RSI14"]
                    )
    if results:
        df = pd.DataFrame(results)
        summary_placeholder.dataframe(df, use_container_width=True, hide_index=True)
        last_scan_time.caption(f"✅ Last scan: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        summary_placeholder.warning("⚠️ No valid data fetched.")

# -----------------------
# Alert History
# -----------------------
st.subheader("📜 Alert History")

if not st.session_state.alert_history.empty:
    st.dataframe(st.session_state.alert_history, use_container_width=True, hide_index=True)
    if st.button("🧹 Clear History"):
        st.session_state.alert_history = pd.DataFrame(
            columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
        )
        st.success("✅ Alert history cleared!")
else:
    st.info("No alerts yet. Run a scan to generate alerts.")

# -----------------------
# Manual Actions
# -----------------------
if run_now:
    run_scan()

if st.button("📨 Send Test Telegram Alert"):
    if send_telegram("✅ Test alert from Indian Stock Agent Bot!"):
        st.success("✅ Telegram test alert sent successfully!")
    else:
        st.error("❌ Telegram send failed.")

# -----------------------
# Auto Scan
# -----------------------
if auto:
    if st_autorefresh:
        refresh_interval_ms = int(interval) * 1000
        st_autorefresh(interval=refresh_interval_ms, key="auto_refresh")
        run_scan()
    else:
        st.warning("⚠️ Auto-refresh package missing. Run: pip install streamlit-autorefresh")

# end of file
