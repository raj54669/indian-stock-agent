# app.py — Indian Stock Agent (Improved: Only 8 Enhancements)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io, os, requests, time
from datetime import datetime, timedelta, timezone
from typing import Optional

# --------------------------------
# Streamlit Config
# --------------------------------
st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Alert Bot", layout="wide")
st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# --------------------------------
# Initialize session state
# --------------------------------
if "alert_history" not in st.session_state:
    st.session_state.alert_history = pd.DataFrame(
        columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
    )

# --------------------------------
# Constants (Improvement #1)
# --------------------------------
EMA_SPAN = 200
RSI_PERIOD = 14
WATCH_RSI_MIN, WATCH_RSI_MAX = 30, 40
RSI_OVERSOLD, RSI_OVERBOUGHT = 30, 70
EMA_TOLERANCE = 0.02  # ±2%

# --------------------------------
# Optional Auto-refresh
# --------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# --------------------------------
# Secrets helpers
# --------------------------------
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

# GitHub + Telegram secrets
GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_REPO = get_secret("GITHUB_REPO")
GITHUB_FILE_PATH = get_secret("GITHUB_FILE_PATH", "watchlist.xlsx")
GITHUB_BRANCH = get_secret("GITHUB_BRANCH", "main")

TELEGRAM_TOKEN = get_secret_section("TELEGRAM_TOKEN", section="telegram")
CHAT_ID = get_secret_section("CHAT_ID", section="telegram")

# --------------------------------
# GitHub + Watchlist
# --------------------------------
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
    except Exception:
        pass
    return pd.DataFrame()

# Sidebar
st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

use_uploaded = False
watchlist_df = pd.DataFrame()
if uploaded_file is not None:
    try:
        df_up = pd.read_excel(uploaded_file)
        if "Symbol" in df_up.columns:
            watchlist_df = df_up
            use_uploaded = True
            st.sidebar.success(f"✅ Using uploaded watchlist ({len(df_up)} symbols)")
        else:
            st.sidebar.error("Uploaded file must contain 'Symbol' column.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if not use_uploaded:
    watchlist_df = load_excel_from_github()
    st.sidebar.info("Using GitHub watchlist")

st.sidebar.header("Status")
st.sidebar.write(f"GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
st.sidebar.write(f"GitHub Token: {'✅' if GITHUB_TOKEN else '❌'}")
st.sidebar.write(f"Telegram Token: {'✅' if TELEGRAM_TOKEN else '❌'}")
st.sidebar.caption(f"📦 yfinance version: {yf.__version__}")

# --------------------------------
# Telegram Helper
# --------------------------------
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, data=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        st.warning(f"Telegram send error: {e}")
        return False

# --------------------------------
# Indicator Calculations
# --------------------------------
def calc_rsi_ema(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)
    df["EMA200"] = df["Close"].ewm(span=EMA_SPAN, adjust=False).mean()

    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))
    cutoff = df.index.max() - timedelta(days=365)
    df_1y = df[df.index >= cutoff]
    df["52W_High"] = df_1y["Close"].max()
    df["52W_Low"] = df_1y["Close"].min()
    return df

# --------------------------------
# Main UI
# --------------------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

symbols = watchlist_df["Symbol"].dropna().astype(str).tolist() if not watchlist_df.empty else []
st.subheader("📋 Combined Summary Table")

initial_df = pd.DataFrame({
    "Symbol": symbols,
    "CMP": ["" for _ in symbols],
    "52W_Low": ["" for _ in symbols],
    "52W_High": ["" for _ in symbols],
    "EMA200": ["" for _ in symbols],
    "RSI14": ["" for _ in symbols],
    "Signal": ["" for _ in symbols],
})
summary_placeholder = st.empty()
summary_placeholder.dataframe(initial_df, use_container_width=True, hide_index=True)
last_scan_time = st.caption("Will auto-update after scanning.")

# --------------------------------
# Controls
# --------------------------------
st.subheader("⚙️ Controls")
col1, col2 = st.columns([1, 2])

with col1:
    run_now = st.button("Run Scan Now", key="run_now_btn")
    interval = st.number_input("Interval (sec)", value=60, step=5, min_value=5)
    auto = st.checkbox("Enable Auto-scan", key="auto_chk")

with col2:
    st.markdown("**Status:**")
    st.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    if auto:
        st.markdown(f"<span style='margin-left:10px;'>🔁 Auto-scan active — every {interval} sec</span>", unsafe_allow_html=True)

# --------------------------------
# Run Scan (Improvements #2, #4, #5, #6, #7)
# --------------------------------
def run_scan():
    results = []
    debug_logs = []
    progress_bar = st.progress(0)

    for i, symbol in enumerate(symbols):
        try:
            debug_logs.append(f"Fetching {symbol} ...")
            df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                debug_logs.append(f"⚠️ No data for {symbol}")
                continue
            df = calc_rsi_ema(df)
            last = df.iloc[-1]

            cmp_ = float(last["Close"])
            ema = float(last["EMA200"])
            rsi = float(last["RSI14"])
            h52, l52 = float(last["52W_High"]), float(last["52W_Low"])

            signal, cond = "Neutral", ""
            if abs(cmp_ - ema) / cmp_ <= EMA_TOLERANCE and WATCH_RSI_MIN <= rsi <= WATCH_RSI_MAX:
                signal, cond = "🟡 WATCH", "EMA200 within ±2% of CMP & RSI 30–40"
            elif cmp_ > ema and rsi < RSI_OVERSOLD:
                signal, cond = "🟢 BUY", "RSI < 30 and CMP above EMA200"
            elif cmp_ < ema and rsi > RSI_OVERBOUGHT:
                signal, cond = "🔴 SELL", "RSI > 70 and CMP below EMA200"

            if signal != "Neutral":
                trend = "Uptrend" if cmp_ > ema else "Downtrend"
                msg = (
                    f"⚡ Alert: {signal}\n"
                    f"**{symbol}**\n"
                    f"CMP = {cmp_:.2f}\n"
                    f"EMA200 = {ema:.2f}\n"
                    f"RSI14 = {rsi:.2f}\n"
                    f"Condition: {cond}\n"
                    f"Trend: {trend}"
                )
                send_telegram(msg)
                add_to_alert_history(symbol, signal, cmp_, ema, rsi)

            results.append({
                "Symbol": symbol,
                "CMP": round(cmp_, 2),
                "52W_Low": round(l52, 2),
                "52W_High": round(h52, 2),
                "EMA200": round(ema, 2),
                "RSI14": round(rsi, 2),
                "Signal": signal,
            })
            debug_logs.append(f"✅ {symbol} — CMP={cmp_:.2f}, EMA200={ema:.2f}, RSI={rsi:.2f}")
        except Exception as e:
            debug_logs.append(f"❌ {symbol}: {e}")
        progress_bar.progress(int(((i + 1) / len(symbols)) * 100))

    if results:
        df = pd.DataFrame(results)
        summary_placeholder.dataframe(df, use_container_width=True, hide_index=True)
        ist = timezone(timedelta(hours=5, minutes=30))
        last_scan_time.caption(f"Last scan: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        summary_placeholder.warning("No valid data fetched.")

    with st.expander("🔍 Debug Logs"):
        for line in debug_logs:
            st.text(line)

# --------------------------------
# Alert History (Improvement #3)
# --------------------------------
st.subheader("📜 Alert History")

def add_to_alert_history(symbol, signal, cmp_, ema, rsi):
    ts = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([{
        "Date & Time (IST)": ts,
        "Symbol": symbol,
        "Signal": signal,
        "CMP": round(cmp_, 2),
        "EMA200": round(ema, 2),
        "RSI14": round(rsi, 2),
    }])
    st.session_state.alert_history = pd.concat(
        [st.session_state.alert_history, new_row], ignore_index=True
    )

if not st.session_state.alert_history.empty:
    st.dataframe(st.session_state.alert_history, use_container_width=True, hide_index=True)
    if st.button("🧹 Clear History"):
        if st.button("✅ Confirm Clear"):
            st.session_state.alert_history = pd.DataFrame(
                columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
            )
            st.success("✅ Alert history cleared!")
            st.experimental_rerun()
else:
    st.info("No alerts recorded yet. Run a scan to generate new alerts.")

# --------------------------------
# Buttons & Auto-scan
# --------------------------------
if run_now:
    run_scan()

if auto and st_autorefresh:
    st_autorefresh(interval=int(interval) * 1000, key="auto_refresh")
    run_scan()

# end of file
