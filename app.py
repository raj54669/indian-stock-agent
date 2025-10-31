# app.py — Indian Stock Agent (ready-to-paste)
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
st.sidebar.header("Settings")

if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.info("Telegram not configured — alerts disabled")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.warning("GitHub credentials missing (set GITHUB_TOKEN and GITHUB_REPO)")

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
# Indicator calc (expects DataFrame)
# -----------------------
def calc_rsi_ema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a DataFrame with a 'Close' column (numeric). Returns df with EMA200, RSI14, 52W_High, 52W_Low.
    """
    df = df.copy()
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")

    # ensure numeric close prices
    close = pd.to_numeric(df["Close"], errors="coerce")

    # EMA200
    span_val = 200 if len(close) >= 200 else max(2, len(close))
    df["EMA200"] = close.ewm(span=span_val, adjust=False).mean()

    # RSI14 calculation
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # 52-week high/low
    df["52W_High"] = close.rolling(window=252, min_periods=1).max()
    df["52W_Low"] = close.rolling(window=252, min_periods=1).min()

    return df


# -----------------------
# Analyzer: download & return single-row dict
# -----------------------
# -----------------------
# Analyzer
# -----------------------
def analyze(symbol: str):
    try:
        # Download 1 year of daily data for each symbol
        df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError(f"No data fetched for {symbol}")

        # Calculate EMA and RSI using the existing helper
        df = calc_rsi_ema(df)

        # Compute 52-week high/low
        df["52W_High"] = df["Close"].rolling(252, min_periods=1).max()
        df["52W_Low"] = df["Close"].rolling(252, min_periods=1).min()

        # Get latest row
        last = df.iloc[-1]
        cmp_ = float(last["Close"])
        ema200 = float(last["EMA200"])
        rsi14 = float(last["RSI14"])
        high_52w = float(last["52W_High"])
        low_52w = float(last["52W_Low"])

        # Signal logic
        signal = "Neutral"
        if cmp_ > ema200 and rsi14 < 30:
            signal = "🔼 Oversold + Above EMA200"
        elif cmp_ < ema200 and rsi14 > 70:
            signal = "🔻 Overbought + Below EMA200"

        return {
            "Symbol": symbol,
            "CMP": round(cmp_, 2),
            "52W_Low": round(low_52w, 2),
            "52W_High": round(high_52w, 2),
            "EMA200": round(ema200, 2),
            "RSI14": round(rsi14, 2),
            "Signal": signal
        }

    except Exception as e:
        st.error(f"analyze() error for {symbol}: {e}")
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

# Controls
st.subheader("⚙️ Controls")
col1, col2 = st.columns([1, 2])

with col1:
    run_now = st.button("Run Scan Now", key="run_now_btn")
    auto = st.checkbox("Enable Auto-scan (local only)", key="auto_chk")
    interval = st.number_input("Interval (sec)", value=60, step=5, min_value=5, key="interval_input")

with col2:
    st.markdown("**Status:**")
    st.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    try:
        st.caption(f"yfinance version: {yf.__version__}")
    except Exception:
        pass

# -----------------------
# Run scan
# -----------------------
def run_scan():
    if not symbols:
        st.warning("No symbols to scan.")
        return

    results = []
    debug_lines = []

    for s in symbols:
        debug_lines.append(f"Processing {s} ...")
        r = analyze(s)
        if r:
            results.append(r)
            debug_lines.append(f"OK {s}: CMP={r['CMP']} EMA200={r['EMA200']} RSI={r['RSI14']}")
        else:
            debug_lines.append(f"No result for {s}")
        # small delay to be polite to Yahoo
        time.sleep(0.25)

    if results:
        df_result = pd.DataFrame(results)
        # ensure consistent column order
        cols = ["Symbol", "CMP", "52W_Low", "52W_High", "EMA200", "RSI14", "Signal"]
        df_result = df_result[cols]
        summary_placeholder.dataframe(df_result, use_container_width=True, hide_index=True)
        last_scan_time.caption(f"Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        summary_placeholder.warning("No valid data fetched.")

    # show debug in expander
    if debug_lines:
        with st.expander("🔍 Debug details (click to expand)"):
            for l in debug_lines:
                st.text(l)

# Run button
if run_now:
    run_scan()

# Optional auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    if auto:
        # trigger rerun using autorefresh; the app will rerun and re-evaluate run_now/auto
        st_autorefresh(interval=int(interval) * 1000, key="autorefresh")
        st.info(f"🔁 Auto-scan active — every {interval} seconds")
        # run on this render too
        run_scan()
except Exception:
    st.info("Optional: pip install streamlit-autorefresh for background scans")

# end of file
