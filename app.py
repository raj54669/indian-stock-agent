# app.py — Indian Stock Agent (ready-to-paste)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io, os, time, requests
from datetime import datetime
from typing import Optional

# Optional auto-refresh support (safe import)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# -----------------------
# Streamlit Config
# -----------------------
st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Alert Bot", layout="wide")

st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

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

# Telegram secrets
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
# Load Excel from GitHub
# -----------------------
@st.cache_data(ttl=120)
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
# Sidebar: upload + status
# -----------------------
st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

# -----------------------
# Load watchlist (uploaded override or GitHub)
# -----------------------
use_uploaded = False
watchlist_df = pd.DataFrame()
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
    
# -----------------------
st.sidebar.header("Settings")

if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.info("Telegram not configured — alerts disabled")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.warning("GitHub credentials missing (set GITHUB_TOKEN and GITHUB_REPO)")


# sanitize watchlist: ensure Symbol column exists
if not watchlist_df.empty:
    if "Symbol" in watchlist_df.columns:
        watchlist_df["Symbol"] = watchlist_df["Symbol"].astype(str).str.strip()
        watchlist_df = watchlist_df[watchlist_df["Symbol"] != ""]
    else:
        watchlist_df = pd.DataFrame()

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
# Robust indicators calculator + analyzer (replace existing functions)
# -----------------------
from datetime import timedelta

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns into single-level strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(x) for x in col if x is not None and str(x) != ""])
            .strip("_")
            for col in df.columns.values
        ]
    return df

def _find_close_column(df: pd.DataFrame):
    """Find the best candidate for Close column (case-insensitive)."""
    cols = [c for c in df.columns]
    # Prefer exact 'Close' or 'Adj Close' endings; fallback to any column containing 'close'
    for prefer in ("close", "adjclose", "adj_close", "adjusted_close"):
        for c in cols:
            if c.replace(" ", "").replace("_", "").lower() == prefer:
                return c
    for c in cols:
        if "close" in c.lower():
            return c
    return None

def calc_rsi_ema(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Input: raw dataframe from yfinance (may be MultiIndex).
    Output: df with EMA200, RSI14, 52W_High, 52W_Low (or None on failure).
    """
    try:
        if df is None or df.empty:
            return None

        df = _flatten_columns(df)

        close_col = _find_close_column(df)
        if close_col is None:
            # can't find any close-like column
            return None

        # Coerce to numeric and drop missing closes
        df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)

        # EMA200: use full available data, min_periods=1 so we always get a number
        df["EMA200"] = df["Close"].ewm(span=200, adjust=False, min_periods=1).mean()

        # RSI14 (Wilder smoothing via ewm alpha = 1/14)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

        # 52-week high/low using exactly 365 calendar days from last available date
        last_date = df.index.max()
        cutoff = last_date - timedelta(days=365)
        df_1y = df[df.index >= cutoff]
        if not df_1y.empty:
            h52 = df_1y["Close"].max()
            l52 = df_1y["Close"].min()
        else:
            # fallback to full available history if 1y slice empty
            h52 = df["Close"].max()
            l52 = df["Close"].min()

        # broadcast scalar 52W values so last row can display them easily
        df["52W_High"] = h52
        df["52W_Low"] = l52

        return df

    except Exception:
        return None


def analyze(symbol: str):
    """
    Download data and return a single-row dict with indicators.
    Returns None on failure (caller should log/display debug).
    """
    try:
        # fetch 2 years to stabilize EMA200
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        df_ind = calc_rsi_ema(df)
        if df_ind is None or df_ind.empty:
            return None

        last = df_ind.iloc[-1]

        # safe extraction (guard against NaN)
        cmp_ = float(last["Close"]) if not pd.isna(last["Close"]) else None
        ema200 = float(last["EMA200"]) if "EMA200" in last.index and not pd.isna(last["EMA200"]) else None
        rsi14 = float(last["RSI14"]) if "RSI14" in last.index and not pd.isna(last["RSI14"]) else None
        low52 = float(last["52W_Low"]) if "52W_Low" in last.index and not pd.isna(last["52W_Low"]) else None
        high52 = float(last["52W_High"]) if "52W_High" in last.index and not pd.isna(last["52W_High"]) else None

        if cmp_ is None:
            return None

        signal = "Neutral"
        if ema200 is not None and rsi14 is not None:
            if cmp_ > ema200 and rsi14 < 30:
                signal = "BUY"
            elif cmp_ < ema200 and rsi14 > 70:
                signal = "SELL"

        return {
            "Symbol": symbol,
            "CMP": round(cmp_, 2),
            "52W_Low": round(low52, 2) if low52 is not None else None,
            "52W_High": round(high52, 2) if high52 is not None else None,
            "EMA200": round(ema200, 2) if ema200 is not None else None,
            "RSI14": round(rsi14, 2) if rsi14 is not None else None,
            "Signal": signal
        }

    except Exception:
        return None


# -----------------------
# Main UI
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

if watchlist_df.empty or "Symbol" not in watchlist_df.columns:
    st.warning("⚠️ No valid 'Symbol' column found in your watchlist Excel file.")
    symbols = []
else:
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()

# Combined summary placeholder (only one)
st.subheader("📋 Combined Summary Table")
initial_df = pd.DataFrame({
    "Symbol": symbols if symbols else [],
    "CMP": ["" for _ in symbols] if symbols else [],
    "52W_Low": ["" for _ in symbols] if symbols else [],
    "52W_High": ["" for _ in symbols] if symbols else [],
    "EMA200": ["" for _ in symbols] if symbols else [],
    "RSI14": ["" for _ in symbols] if symbols else [],
    "Signal": ["" for _ in symbols] if symbols else [],
})
summary_placeholder = st.empty()
summary_placeholder.dataframe(initial_df, use_container_width=True, hide_index=True)
last_scan_time = st.caption("Will auto-update after scanning.")

# -----------------------
# Controls
# -----------------------
st.subheader("⚙️ Controls")

col1, col2 = st.columns([1, 2])

with col1:
    run_now = st.button("Run Scan Now", key="run_now_btn")

    interval = st.number_input("Interval (sec)", value=60, step=5, min_value=5, key="interval_input")

    auto_col1, auto_col2 = st.columns([1, 3])
    with auto_col1:
        auto = st.checkbox("Enable Auto-scan", key="auto_chk")

with col2:
    st.markdown("**Status:**")
    st.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    if auto:
        st.write(f"🔁 Auto-scan active — every {interval} seconds")

# Move yfinance version info into sidebar
try:
    st.sidebar.caption(f"📦 yfinance version: {yf.__version__}")
except Exception:
    pass

# -----------------------
# Maintain Alert History
# -----------------------
if "alert_history" not in st.session_state:
    st.session_state["alert_history"] = []

# -----------------------
# Run Scan (with conditional debug output)
# -----------------------
def run_scan():
    results = []
    debug_logs = []
    errors_found = False

    for symbol in symbols:
        try:
            debug_logs.append(f"Processing {symbol} ...")
            df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)

            if df is None or df.empty:
                msg = f"⚠️ No data for {symbol}"
                debug_logs.append(msg)
                errors_found = True
                continue

            df = calc_rsi_ema(df)
            last = df.iloc[-1]

            cmp_ = float(last["Close"])
            ema200 = float(last["EMA200"])
            rsi14 = float(last["RSI14"])
            high_52w = float(last["52W_High"])
            low_52w = float(last["52W_Low"])

            # --- Determine signal and log alerts ---
            if cmp_ > ema200 and rsi14 < 30:
                signal = "🔼 Oversold + Above EMA200"
                st.session_state["alert_history"].append(
                    f"BUY ALERT: {symbol} | CMP={cmp_} | EMA200={ema200} | RSI={rsi14}"
                )
                send_telegram(f"📈 BUY Triggered for {symbol}\nCMP={cmp_}\nEMA200={ema200}\nRSI={rsi14}")
            elif cmp_ < ema200 and rsi14 > 70:
                signal = "🔻 Overbought + Below EMA200"
                st.session_state["alert_history"].append(
                    f"SELL ALERT: {symbol} | CMP={cmp_} | EMA200={ema200} | RSI={rsi14}"
                )
                send_telegram(f"📉 SELL Triggered for {symbol}\nCMP={cmp_}\nEMA200={ema200}\nRSI={rsi14}")
            else:
                signal = "Neutral"

            results.append({
                "Symbol": symbol,
                "CMP": round(cmp_, 2),
                "52W_Low": round(low_52w, 2),
                "52W_High": round(high_52w, 2),
                "EMA200": round(ema200, 2),
                "RSI14": round(rsi14, 2),
                "Signal": signal
            })

            debug_logs.append(f"✅ OK {symbol}: CMP={cmp_} EMA200={ema200} RSI={rsi14}")

        except Exception as e:
            msg = f"❌ {symbol}: {e}"
            debug_logs.append(msg)
            st.error(msg)
            errors_found = True

    # --- Update UI ---
    if results:
        df = pd.DataFrame(results)
        summary_placeholder.dataframe(df, use_container_width=True, hide_index=True)
        from datetime import timezone, timedelta 
        ist = timezone(timedelta(hours=5, minutes=30)) 
        last_scan_time.caption(f"Last scan: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        summary_placeholder.warning("No valid data fetched.")


# -----------------------
# Buttons and Actions
# -----------------------
if run_now:
    run_scan()

test_telegram = st.button("📨 Send Test Telegram Alert")
if test_telegram:
    success = send_telegram("✅ Test alert from Indian Stock Agent Bot!")
    if success:
        st.success("✅ Telegram test alert sent successfully!")
    else:
        st.error("❌ Telegram send failed. Check your token or chat_id.")

# -----------------------
# Alert History Section
# -----------------------
with st.expander("📜 View Alert History"):
    if st.session_state["alert_history"]:
        for msg in reversed(st.session_state["alert_history"][-20:]):  # last 20 alerts
            st.write(msg)
    else:
        st.info("No alerts triggered yet.")

# -----------------------
# Auto-scan via streamlit-autorefresh
# -----------------------
if auto:
    if st_autorefresh:
        refresh_interval_ms = int(interval) * 1000
        st_autorefresh(interval=refresh_interval_ms, key="auto_refresh")
        run_scan()
    else:
        st.warning("⚠️ Auto-refresh package missing. Run: pip install streamlit-autorefresh")

# end of file
