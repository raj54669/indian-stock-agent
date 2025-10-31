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
# -----------------------
# RSI & EMA Calculation (fixed)
# -----------------------
# -----------------------
# RSI & EMA Calculation (365-day window)
# -----------------------
# -----------------------
# RSI & EMA Calculation (exact trailing 365 days)
# -----------------------
def calc_rsi_ema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure valid numeric Close
    close = pd.to_numeric(df["Close"], errors="coerce")
    df = df.loc[close.dropna().index]

    if df.empty:
        return df

    # --- EMA200 ---
    df["EMA200"] = close.ewm(span=min(200, len(close)), adjust=False).mean()

    # --- RSI14 ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # --- 52-week High/Low based on actual 365-day window ---
    cutoff_date = datetime.now() - pd.Timedelta(days=365)
    df_1y = df[df.index >= cutoff_date]

    if not df_1y.empty:
        high_52w = df_1y["Close"].max()
        low_52w = df_1y["Close"].min()
    else:
        # fallback to full data if less than 1y available
        high_52w = df["Close"].max()
        low_52w = df["Close"].min()

    # assign same values to all rows so last row can access
    df["52W_High"] = high_52w
    df["52W_Low"] = low_52w

    return df



# -----------------------
# Analyzer (fixed)
# -----------------------
def analyze(symbol: str):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError(f"No data for {symbol}")

        df = calc_rsi_ema(df)
        last = df.iloc[-1]

        cmp_ = float(last["Close"])
        ema200 = float(last["EMA200"])
        rsi14 = float(last["RSI14"])
        high_52w = float(last["52W_High"])
        low_52w = float(last["52W_Low"])

        # Determine signal
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

            if cmp_ > ema200 and rsi14 < 30:
                signal = "🔼 Oversold + Above EMA200"
            elif cmp_ < ema200 and rsi14 > 70:
                signal = "🔻 Overbought + Below EMA200"
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
