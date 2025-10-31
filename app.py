# app.py — Indian Stock Agent (fixed & ready to paste)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io, os, requests, time
from datetime import datetime, timedelta, timezone
from typing import Optional

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Alert Bot", layout="wide")

# -----------------------
# Secrets helper
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

# GitHub headers (for raw file)
def github_raw_headers():
    auth_scheme = "Bearer" if str(GITHUB_TOKEN or "").startswith("github_pat_") else "token"
    return {
        "Authorization": f"{auth_scheme} {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "streamlit-indian-stock-agent"
    }

# -----------------------
# Load watchlist (GitHub or upload)
# -----------------------
st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

st.sidebar.header("Settings")
if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.info("⚠️ Telegram not configured — alerts disabled")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.warning("GitHub credentials missing (watchlist fetch disabled)")

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
        else:
            st.sidebar.error(f"GitHub fetch failed: {r.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.sidebar.error(f"Error loading from GitHub: {e}")
        return pd.DataFrame()

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
# Page title & initial combined table (static)
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

cols = ["Symbol", "CMP", "52W_Low", "52W_High", "EMA200", "RSI14", "Signal"]

if watchlist_df is None or watchlist_df.empty or "Symbol" not in watchlist_df.columns:
    st.warning("⚠️ No valid watchlist found. Upload a watchlist with a column named `Symbol` or configure GitHub.")
    # show an empty example so UI layout is consistent
    example = pd.DataFrame({"Symbol": ["RELIANCE.NS", "INFY.NS"], "CMP": ["", ""], "52W_Low": ["", ""], "52W_High": ["", ""], "EMA200": ["", ""], "RSI14": ["", ""], "Signal": ["", ""]})
    summary_placeholder = st.empty()
    summary_placeholder.dataframe(example, use_container_width=True, hide_index=True)
    symbols = []
else:
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()
    initial_df = pd.DataFrame({c: ["" if c != "Symbol" else s for s in symbols] for c in cols})
    summary_placeholder = st.empty()
    summary_placeholder.dataframe(initial_df, use_container_width=True, hide_index=True)
    last_scan_time = st.empty()
    last_scan_time.caption("Will auto-update after scanning.")

st.caption("")  # small spacing

# -----------------------
# Helpers
# -----------------------
def send_telegram(message: str):
    if not (TELEGRAM_TOKEN and CHAT_ID):
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data={"chat_id": CHAT_ID, "text": message}, timeout=10)
        return r.status_code == 200
    except Exception as e:
        st.warning(f"Telegram error: {e}")
        return False

# -----------------------
# Indicators calculation (robust)
# -----------------------
def calc_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Input: a daily OHLC DataFrame with Date index and 'Close' column
    Output: same DataFrame with EMA200, RSI14, 52W_High, 52W_Low columns added
    """
    try:
        if df is None or df.empty:
            return None

        df = df.copy()
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        # Ensure numeric Close and drop NA
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None

        # EMA200 on full available data (more stable); min_periods=1 so prints even if <200 rows
        df["EMA200"] = df["Close"].ewm(span=200, adjust=False, min_periods=1).mean()

        # RSI14 using Wilder smoothing (alpha=1/14)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

        # 52-week high/low based on exact last_date - 365 days (calendar days)
        last_date = df.index.max()
        cutoff = last_date - timedelta(days=365)
        df_1y = df[df.index >= cutoff]
        if not df_1y.empty:
            h52 = df_1y["Close"].max()
            l52 = df_1y["Close"].min()
        else:
            # fallback to full-history
            h52 = df["Close"].max()
            l52 = df["Close"].min()
        # Put scalar values into columns so easier to display from last row
        df["52W_High"] = h52
        df["52W_Low"] = l52

        return df

    except Exception as e:
        # Do not raise — return None and let caller handle debug
        return None

# -----------------------
# analyze(symbol) - downloads & returns single-row dict or None
# -----------------------
def analyze(symbol: str):
    """
    Download daily data for symbol (2y to ensure enough history), compute indicators,
    and return a dict with required fields or None on failure.
    """
    try:
        # Download using yfinance; use 2y so EMA200 stable
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError("No data returned from yfinance")

        df_ind = calc_indicators(df)
        if df_ind is None or df_ind.empty:
            raise RuntimeError("Indicator calc failed or no numeric Close")

        last = df_ind.iloc[-1]

        cmp_ = float(last["Close"])
        ema200 = float(last["EMA200"]) if not pd.isna(last["EMA200"]) else None
        rsi14 = float(last["RSI14"]) if not pd.isna(last["RSI14"]) else None
        low52 = float(last["52W_Low"]) if not pd.isna(last["52W_Low"]) else None
        high52 = float(last["52W_High"]) if not pd.isna(last["52W_High"]) else None

        # Signal rule (example)
        signal = "Neutral"
        if (ema200 is not None) and (rsi14 is not None):
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

    except Exception as e:
        # Return exception message so caller can show in debug
        return {"__error__": str(e), "Symbol": symbol}

# -----------------------
# Controls UI
# -----------------------
st.subheader("⚙️ Controls")
col_left, col_right = st.columns([1, 1])  # balanced so Interval input not too wide

with col_left:
    run_now = st.button("Run Scan Now", key="run_now_btn")
    auto = st.checkbox("Enable Auto-scan (local only)", key="auto_chk")
    interval = st.number_input("Interval (sec)", value=60, step=5, min_value=5, key="interval_input")

with col_right:
    st.write("Status:")
    st.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    try:
        st.caption(f"yfinance version: {yf.__version__}")
    except Exception:
        pass

# -----------------------
# Main scanning function
# -----------------------
def run_scan_once():
    results = []
    debug_lines = []
    alerts = []

    if not symbols:
        st.warning("No symbols to scan.")
        return results, debug_lines, alerts

    with st.spinner(f"Scanning {len(symbols)} symbols..."):
        for s in symbols:
            debug_lines.append(f"Processing {s} ...")
            r = analyze(s)
            # r may be an error-dict or valid dict
            if r is None:
                debug_lines.append(f"❌ {s}: No result (None returned)")
            elif "__error__" in r:
                debug_lines.append(f"❌ {s}: {r['__error__']}")
            else:
                results.append(r)
                debug_lines.append(f"✅ {s}: CMP={r['CMP']} EMA200={r['EMA200']} RSI={r['RSI14']}")
                if r["Signal"] in ("BUY", "SELL"):
                    alerts.append(f"{s}: {r['Signal']} (CMP={r['CMP']}, EMA200={r['EMA200']}, RSI={r['RSI14']})")
            # small delay to avoid heavy hammering
            time.sleep(0.25)

    # Update combined summary table (in-place)
    if results:
        df_result = pd.DataFrame(results)
        # Ensure order is same as watchlist (map)
        order = symbols
        df_result = df_result.set_index("Symbol").reindex(order).reset_index()
        summary_placeholder.dataframe(df_result[cols], use_container_width=True, hide_index=True)
        # Indian timezone timestamp
        IST = timezone(timedelta(hours=5, minutes=30))
        last_scan_time.caption(f"Last scan: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        # show warning inside placeholder area
        summary_placeholder.warning("No valid data fetched.")
        last_scan_time.caption("")

    # Alerts via Telegram
    if alerts:
        msg = "⚠️ Stock Alerts:\n" + "\n".join(alerts)
        st.warning(msg)
        send_telegram(msg)

    return results, debug_lines, alerts

# -----------------------
# Run / Auto-refresh wiring
# -----------------------
# Only run when user clicks Run Scan Now (or auto)
if run_now:
    results, debug_lines, alerts = run_scan_once()
    if debug_lines:
        with st.expander("🔍 Debug details (click to expand)"):
            for l in debug_lines:
                st.write(l)

# Optional auto refresh
try:
    from streamlit_autorefresh import st_autorefresh
    if auto:
        # trigger rerun periodically and call run_scan() again
        st_autorefresh(interval=int(interval) * 1000, key="autorefresh")
        st.info(f"🔁 Auto-scan active — every {interval} seconds")
        results, debug_lines, alerts = run_scan_once()
        if debug_lines:
            with st.expander("🔍 Debug details (click to expand)"):
                for l in debug_lines:
                    st.write(l)
except Exception:
    # package optional — ignore if not installed
    pass
