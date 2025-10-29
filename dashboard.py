# dashboard.py
import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime
import time
from pathlib import Path

AUTO_REFRESH_SECONDS = 60
WATCHLIST_FILE = "watchlist.txt"

st.set_page_config(page_title="Indian Stock Live Monitor", page_icon="📊", layout="wide")
st.title("📊 Indian Stock Live Monitor (EMA200 & RSI14)")
st.caption(f"Auto-refresh every {AUTO_REFRESH_SECONDS} seconds | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Meta refresh (works across Streamlit versions)
st.markdown(f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SECONDS}">', unsafe_allow_html=True)

def load_watchlist(path=WATCHLIST_FILE):
    p = Path(path)
    if not p.exists():
        st.error(f"⚠️ '{path}' not found in repository root. Add NSE tickers (e.g. INFY.NS) one per line and redeploy.")
        return []
    try:
        lines = [ln.strip().upper() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
        return lines
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return []

def safe_float(v):
    try:
        if v is None:
            return None
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

def fetch_stats(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False, threads=False)
        if df is None or df.empty:
            return {"Symbol": symbol, "error": "No historical data"}

        df["EMA200"] = ta.ema(df["Close"], length=200)
        df["RSI14"] = ta.rsi(df["Close"], length=14)

        last = df.iloc[-1]
        ema200 = safe_float(last.get("EMA200"))
        rsi14 = safe_float(last.get("RSI14"))
        close_price = safe_float(last.get("Close"))

        if ema200 is None or rsi14 is None or close_price is None:
            return {"Symbol": symbol, "error": "Insufficient EMA/RSI/Close data"}

        latest_close = close_price
        price_time = df.index[-1].strftime("%Y-%m-%d")
        try:
            intr = yf.download(symbol, period="2d", interval="1m", progress=False, threads=False)
            if intr is not None and not intr.empty:
                val = safe_float(intr["Close"].iloc[-1])
                if val is not None:
                    latest_close = val
                try:
                    price_time = intr.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    price_time = df.index[-1].strftime("%Y-%m-%d")
        except Exception:
            latest_close = close_price
            price_time = df.index[-1].strftime("%Y-%m-%d")

        try:
            lower = 0.98 * ema200
            upper = 1.02 * ema200
            latest_close = float(latest_close)
            near_ema = (lower <= latest_close <= upper)
        except Exception:
            near_ema = False

        rsi_ok = (30 <= rsi14 <= 40)
        triggered = near_ema and rsi_ok

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

# Main UI
watchlist = load_watchlist()
if not watchlist:
    st.info("Add NSE tickers (e.g. INFY.NS) to watchlist.txt and redeploy.")
else:
    results = []
    total = len(watchlist)
    progress = st.progress(0)
    status = st.empty()

    for idx, sym in enumerate(watchlist):
        status.text(f"Fetching {sym} ({idx+1}/{total})")
        stats = fetch_stats(sym)
        results.append(stats)
        progress.progress(int(((idx+1)/total) * 100))
        time.sleep(0.35)

    status.empty()
    progress.empty()

    df = pd.DataFrame(results)
    valid_df = df[df.get("error", "") == ""].reset_index(drop=True) if "error" in df.columns else df.copy()
    error_df = df[df.get("error", "") != ""].reset_index(drop=True) if "error" in df.columns else pd.DataFrame()

    if not valid_df.empty:
        st.subheader("✅ Live Stock Status")
        st.dataframe(valid_df, use_container_width=True)
    else:
        st.info("No valid stock data available (see Errors).")

    if not error_df.empty:
        with st.expander("⚠️ Stocks with errors"):
            st.dataframe(error_df, use_container_width=True)

st.markdown("---")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
