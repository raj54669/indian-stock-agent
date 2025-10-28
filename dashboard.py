# dashboard.py
import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime
import time
import math

# -------------------------------
# Config
# -------------------------------
AUTO_REFRESH_SECONDS = 60
WATCHLIST_FILE = "watchlist.txt"

st.set_page_config(page_title="Indian Stock Live Monitor", page_icon="📊", layout="wide")
st.title("📊 Indian Stock Live Monitor (EMA200 & RSI14)")
st.caption(f"Auto-refresh every {AUTO_REFRESH_SECONDS} seconds | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Use an HTML meta refresh to reload the page (works on Render + many Streamlit versions)
refresh_meta = f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SECONDS}">'
st.markdown(refresh_meta, unsafe_allow_html=True)


# -------------------------------
# Utilities
# -------------------------------
def load_watchlist(path=WATCHLIST_FILE):
    try:
        with open(path, "r") as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]
        return symbols
    except FileNotFoundError:
        st.error(f"⚠️ {WATCHLIST_FILE} not found in repository root. Add it and redeploy.")
        return []
    except Exception as e:
        st.error(f"Error reading {WATCHLIST_FILE}: {e}")
        return []


def safe_float(val):
    """Return Python float for scalars, otherwise None."""
    try:
        if val is None:
            return None
        # If it's a pandas scalar/np.nan, handle via pandas
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


# -------------------------------
# Core: fetch single symbol stats
# -------------------------------
def fetch_stats(symbol):
    """
    Returns a dict with:
    - Symbol, Price, Price Time, EMA200, RSI14, Near EMA?, RSI 30-40?, Triggered, error
    """
    try:
        # 1) daily data (1 year gives enough history for EMA200)
        df = yf.download(symbol, period="1y", interval="1d", progress=False, threads=False)
        if df is None or df.empty:
            return {"Symbol": symbol, "error": "No historical data"}

        # 2) indicators (pandas_ta returns Series; ensure last row scalar extraction)
        df["EMA200"] = ta.ema(df["Close"], length=200)
        df["RSI14"] = ta.rsi(df["Close"], length=14)

        # get the last row as a Series, then extract scalars
        last = df.iloc[-1]

        ema200 = safe_float(last.get("EMA200"))
        rsi14 = safe_float(last.get("RSI14"))
        close_price = safe_float(last.get("Close"))

        # validate
        if ema200 is None or rsi14 is None or close_price is None:
            return {"Symbol": symbol, "error": "Insufficient EMA/RSI/Close data"}

        # 3) try quick intraday snapshot (1-min) for freshest price; fallback to daily close
        latest_close = close_price
        price_time = df.index[-1].strftime("%Y-%m-%d")
        try:
            intr = yf.download(symbol, period="2d", interval="1m", progress=False, threads=False)
            if intr is not None and not intr.empty:
                latest_close = safe_float(intr["Close"].iloc[-1])
                # ensure latest_close is valid
                if latest_close is None:
                    latest_close = close_price
                # timestamp
                try:
                    price_time = intr.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    price_time = df.index[-1].strftime("%Y-%m-%d")
        except Exception:
            latest_close = close_price
            price_time = df.index[-1].strftime("%Y-%m-%d")

        # 4) compute boolean triggers using pure Python floats (no Series)
        # guard against zero division or NaNs (ema200 already validated)
        near_ema = False
        try:
            lower = 0.98 * ema200
            upper = 1.02 * ema200
            # ensure latest_close is a float
            latest_close = float(latest_close)
            near_ema = (lower <= latest_close <= upper)
        except Exception:
            near_ema = False

        rsi_ok = (30 <= rsi14 <= 40)

        triggered = (near_ema and rsi_ok)

        return {
            "Symbol": symbol,
            "Price": round(latest_close, 2),
            "Price Time": price_time,
            "EMA200": round(ema200, 2),
            "RSI14": round(rsi14, 2),
            "Near EMA?": "✅" if near_ema else "❌",
            "RSI 30-40?": "✅" if rsi_ok else "❌",
            "Triggered": "✅" if triggered else "❌",
            "error": ""
        }
    except Exception as e:
        return {"Symbol": symbol, "error": str(e)}


# -------------------------------
# UI: fetch all and render
# -------------------------------
watchlist = load_watchlist()
if not watchlist:
    st.info("Add symbol lines to watchlist.txt (e.g. INFY.NS, TCS.NS) and redeploy.")
else:
    # Fetch and present results
    results = []
    total = len(watchlist)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # keep a short pause between calls (helps avoid rate limits)
    for idx, sym in enumerate(watchlist):
        status_text.text(f"Fetching {sym} ({idx+1}/{total})...")
        stats = fetch_stats(sym)
        results.append(stats)
        # update progress as integer 0-100
        progress_value = int(((idx + 1) / total) * 100)
        progress_bar.progress(min(max(progress_value, 0), 100))
        # small delay (short) to be polite to yfinance; reduce if you want faster
        time.sleep(0.4)

    status_text.empty()
    progress_bar.empty()

    df = pd.DataFrame(results)

    # Split valid vs error rows
    if "error" in df.columns:
        valid_df = df[df["error"] == ""].copy()
        error_df = df[df["error"] != ""].copy()
    else:
        valid_df = df.copy()
        error_df = pd.DataFrame(columns=df.columns)

    if not valid_df.empty:
        st.subheader("✅ Live Stock Status")
        st.dataframe(valid_df.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No valid stock data currently available.")

    if not error_df.empty:
        with st.expander("⚠️ View Stocks with Errors"):
            st.dataframe(error_df.reset_index(drop=True), use_container_width=True)

# Footer (timestamp)
st.markdown("---")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
