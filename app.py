# app.py – Indian Stock Agent (100% REST-based GitHub connection, Nextbite-style)

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

# Telegram settings
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
# Sidebar – Simplified
# -----------------------

st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

st.sidebar.header("Settings")

# Telegram connection indicator
if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.warning("⚠️ Telegram not set – alerts disabled")

# GitHub connection indicator
if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.error("GitHub credentials missing")


# -----------------------
# Load Excel file from GitHub (REST)
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
# Save Excel to GitHub (REST)
# -----------------------
def save_excel_to_github(df, message="Update watchlist"):
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}"
        get_resp = requests.get(get_url, headers=github_headers(), timeout=10)
        sha = None
        if get_resp.status_code == 200:
            sha = get_resp.json().get("sha")

        bytes_buf = io.BytesIO()
        df.to_excel(bytes_buf, index=False)
        encoded_content = bytes_buf.getvalue()

        import base64
        data = {
            "message": message,
            "branch": GITHUB_BRANCH,
            "content": base64.b64encode(encoded_content).decode("utf-8"),
        }
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
# Load Watchlist (with upload override)
# -----------------------

# Upload file option to override GitHub watchlist
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
# UI
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

if watchlist_df.empty or "Symbol" not in [c.strip() for c in watchlist_df.columns]:
    st.warning(f"⚠️ No valid 'Symbol' column found in `{GITHUB_FILE_PATH}`.")
    st.markdown("Upload an Excel file to GitHub with a column named **Symbol**.")
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

# -----------------------
# Stock Analysis
# -----------------------
def calc_rsi_ema(symbol: str):
    """
    Fetches 1-day interval historical data, computes EMA200 and RSI14,
    and returns a dataframe plus the last computed row (latest indicators).
    """

    import pandas as pd
    import numpy as np
    import yfinance as yf
    from datetime import datetime, timedelta

    try:
        # --- Fetch last 400 trading days for context ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)

        data = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False)

        if data is None or data.empty:
            st.error(f"No valid data returned for {symbol}")
            return None

        # --- Compute EMA200 ---
        data["EMA200"] = data["Close"].ewm(span=200, adjust=False).mean()

        # --- Compute RSI14 ---
        delta = data["Close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        roll_up = pd.Series(gain).rolling(window=14).mean()
        roll_down = pd.Series(loss).rolling(window=14).mean()

        RS = roll_up / roll_down
        RSI = 100.0 - (100.0 / (1.0 + RS))
        data["RSI14"] = RSI.values

        # --- Compute 52-week high and low ---
        data["52W_High"] = data["High"].rolling(window=252, min_periods=1).max()
        data["52W_Low"] = data["Low"].rolling(window=252, min_periods=1).min()

        # --- Get latest row safely ---
        last_row = data.iloc[-1].copy()
        cmp_price = float(last_row["Close"])
        ema200 = float(last_row["EMA200"])
        rsi14 = float(last_row["RSI14"]) if not np.isnan(last_row["RSI14"]) else None
        high_52w = float(last_row["52W_High"])
        low_52w = float(last_row["52W_Low"])

        # --- Determine Signal ---
        if rsi14 is None:
            signal = "Neutral"
        elif rsi14 < 30 and cmp_price > ema200:
            signal = "BUY"
        elif rsi14 > 70 and cmp_price < ema200:
            signal = "SELL"
        else:
            signal = "Neutral"

        # --- Build summary result ---
        result = {
            "Symbol": symbol,
            "CMP": round(cmp_price, 2),
            "52W_High": round(high_52w, 2),
            "52W_Low": round(low_52w, 2),
            "EMA200": round(ema200, 2),
            "RSI14": round(rsi14, 2) if rsi14 else None,
            "Signal": signal,
        }

        # --- Optional: Debug mini preview ---
        with st.expander(f"🔍 Debug {symbol}", expanded=False):
            st.write(data.tail(3)[["Close", "EMA200", "RSI14"]])

        return result

    except Exception as e:
        st.error(f"Error in calc_rsi_ema for {symbol}: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None

        
def analyze(symbol):
    """
    Use calc_rsi_ema to get the full DataFrame, then extract the latest indicators as a single dict.
    """
    try:
        df = calc_rsi_ema(symbol)
        if df is None or df.empty:
            st.warning(f"No valid DataFrame returned from calc_rsi_ema for {symbol}")
            return None

        last = df.iloc[-1]
        cmp_ = float(last["Close"])
        ema200 = float(last["EMA200"]) if not pd.isna(last["EMA200"]) else None
        rsi14 = float(last["RSI14"]) if not pd.isna(last["RSI14"]) else None
        low52 = float(last.get("52W_Low", pd.NA)) if "52W_Low" in last.index else None
        high52 = float(last.get("52W_High", pd.NA)) if "52W_High" in last.index else None

        # Signal logic per your spec
        signal = "Neutral"
        if ema200 and rsi14 is not None:
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
        st.error(f"analyze() error for {symbol}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# =============================
# ⚙️ Controls and Unified Scanning
# =============================

st.subheader("⚙️ Controls")
col1, col2 = st.columns([1, 2])

with col1:
    # Explicit keys prevent StreamlitDuplicateElementId
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


# =============================
# 🧠 Main Scan Function
# =============================
def run_scan_once():
    """Fetch, analyze all stocks, and display unified table first."""
    if watchlist_df is None or "Symbol" not in watchlist_df.columns:
        st.error("No watchlist available")
        return [], []

    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()
    results, alerts = [], []
    debug_info = []  # store debug notes instead of printing directly

    with st.spinner(f"Scanning {len(symbols)} symbols..."):
        for s in symbols:
            try:
                debug_info.append(f"Processing {s} ...")
                r = analyze(s)
                if r:
                    results.append(r)
                    debug_info.append(f"✔️ Completed {s}: RSI={r['RSI14']} EMA200={r['EMA200']} CMP={r['CMP']}")
                    if r["Signal"] in ("BUY", "SELL"):
                        alerts.append(
                            f"{s}: {r['Signal']} (CMP={r['CMP']}, EMA200={r['EMA200']}, RSI={r['RSI14']})"
                        )
                else:
                    debug_info.append(f"⚠️ No result for {s}")
            except Exception as e:
                debug_info.append(f"❌ Error for {s}: {e}")
            time.sleep(0.25)

    # --- Unified summary table at top ---
    st.markdown("---")
    st.subheader("📊 Combined Summary Table")

    if results:
        df_result = pd.DataFrame(results)
        st.dataframe(
            df_result[["Symbol", "CMP", "52W_Low", "52W_High", "EMA200", "RSI14", "Signal"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Last scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("⚠️ No valid results from scan")

    # --- Alerts (Telegram + UI) ---
    if alerts:
        msg = "⚠️ Stock Alerts:\n" + "\n".join(alerts)
        st.warning(msg)
        try:
            if TELEGRAM_TOKEN and CHAT_ID:
                send_telegram(msg)
        except Exception as e:
            st.error(f"Telegram send failed: {e}")

    # --- Collapsible debug info (no clutter) ---
    if debug_info:
        with st.expander("🔍 Debug details (click to expand)"):
            for line in debug_info:
                st.text(line)

    return results, alerts


# =============================
# 🚀 Run / Auto-refresh Logic
# =============================
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
