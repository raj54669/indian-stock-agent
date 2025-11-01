"""
app.py — Indian Stock Agent (Refactored)
----------------------------------------
A robust Streamlit app that:
    ✅ Loads stock watchlist from GitHub (with upload option)
    ✅ Scans using EMA200 + RSI14 indicators
    ✅ Sends Telegram alerts for BUY / SELL / WATCH signals
    ✅ Maintains in-session alert history
    ✅ Supports auto-scan (default ON)
"""

import os
import io
import time
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta, timezone

# Internal imports
from indicators import calc_rsi_ema, analyze
from utils import (
    send_telegram,
    load_excel_from_github,
    upload_to_github,
    get_secret,
    GITHUB_REPO,
    GITHUB_TOKEN,
    GITHUB_BRANCH,
    GITHUB_FILE_PATH,
    TELEGRAM_TOKEN,
    CHAT_ID,
)
from ui_helpers import show_alert_history, display_stock_summary, handle_file_upload


# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Bot", layout="wide")
st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# -------------------------------
# Initialize session state
# -------------------------------
if "alert_history" not in st.session_state:
    st.session_state.alert_history = pd.DataFrame(
        columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
    )

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("📂 Watchlist Management")

# Allow GitHub watchlist loading
watchlist_df = load_excel_from_github()
if watchlist_df.empty:
    st.sidebar.warning("⚠️ No watchlist found on GitHub.")
else:
    st.sidebar.success(f"✅ Loaded {len(watchlist_df)} symbols from GitHub.")

# Allow file upload (replaces GitHub file)
handle_file_upload(upload_to_github)

st.sidebar.header("⚙️ Settings")

if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.warning("⚠️ Telegram not configured (alerts disabled)")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub connected")
else:
    st.sidebar.warning("⚠️ GitHub secrets missing")

st.sidebar.caption(f"📦 yfinance version: {yf.__version__}")

# -------------------------------
# Main UI
# -------------------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

symbols = []
if not watchlist_df.empty and "Symbol" in watchlist_df.columns:
    symbols = watchlist_df["Symbol"].dropna().astype(str).str.strip().tolist()

if not symbols:
    st.warning("⚠️ No valid 'Symbol' column found in your watchlist Excel file.")
    st.stop()

# Placeholder for table
summary_placeholder = st.empty()
summary_placeholder.dataframe(
    pd.DataFrame({"Symbol": symbols, "CMP": "", "EMA200": "", "RSI14": "", "Signal": ""}),
    use_container_width=True,
    hide_index=True,
)
last_scan_time = st.caption("Waiting for scan...")

# -------------------------------
# Controls
# -------------------------------
st.subheader("⚙️ Controls")
col1, col2 = st.columns([1, 2])

with col1:
    run_now = st.button("🚀 Run Scan Now", key="run_now_btn")
    interval = st.number_input("Interval (sec)", value=60, step=5, min_value=5, key="interval_input")
    auto = st.checkbox("Enable Auto-scan", value=True, key="auto_chk")

with col2:
    st.markdown("**Status:**")
    st.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    if auto:
        st.markdown(
            f"<span style='margin-left:10px;'>🔁 Auto-scan active — every {interval} seconds</span>",
            unsafe_allow_html=True,
        )


# -------------------------------
# Run Scan
# -------------------------------
IST = timezone(timedelta(hours=5, minutes=30))

def add_to_alert_history(symbol: str, signal: str, cmp_: float, ema200: float, rsi14: float):
    """Adds a new alert record to session-state history."""
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

def run_scan():
    """Runs scan for all symbols and updates UI."""
    results = []

    for symbol in symbols:
        try:
            data = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
            if data is None or data.empty:
                st.warning(f"⚠️ No data for {symbol}")
                continue

            df = calc_rsi_ema(data)
            if df is None or df.empty:
                continue

            last = df.iloc[-1]
            cmp_, ema200, rsi14 = last["Close"], last["EMA200"], last["RSI14"]

            # Signal logic (unchanged)
            signal = "Neutral"
            if cmp_ > ema200 and rsi14 < 30:
                signal = "🟢 BUY"
            elif cmp_ < ema200 and rsi14 > 70:
                signal = "🔴 SELL"
            elif abs(cmp_ - ema200) / cmp_ <= 0.02 and 30 <= rsi14 <= 40:
                signal = "🟡 WATCH"

            # Add alert if triggered
            if signal != "Neutral":
                add_to_alert_history(symbol, signal, cmp_, ema200, rsi14)
                emoji = "⚡" if "WATCH" in signal else ("📈" if "BUY" in signal else "📉")
                alert_msg = (
                    f"{emoji} *Alert:* {symbol}\n"
                    f"CMP = {cmp_:.2f}\n"
                    f"EMA200 = {ema200:.2f}\n"
                    f"RSI14 = {rsi14:.2f}\n"
                    f"Signal: {signal}"
                )
                send_telegram(alert_msg)

            results.append({
                "Symbol": symbol,
                "CMP": round(cmp_, 2),
                "EMA200": round(ema200, 2),
                "RSI14": round(rsi14, 2),
                "Signal": signal
            })

        except Exception as e:
            st.error(f"❌ {symbol}: {e}")

    if results:
        df = pd.DataFrame(results)
        display_stock_summary(df)
        last_scan_time.caption(f"Last scan: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        st.warning("⚠️ No valid results found.")

# -------------------------------
# Buttons & Auto Scan
# -------------------------------
if run_now:
    run_scan()

if auto:
    try:
        from streamlit_autorefresh import st_autorefresh
        refresh_interval_ms = int(interval) * 1000
        st_autorefresh(interval=refresh_interval_ms, key="auto_refresh")
        run_scan()
    except Exception:
        st.warning("⚠️ Auto-refresh unavailable (install: pip install streamlit-autorefresh)")

# -------------------------------
# Alert History Display
# -------------------------------
st.divider()
show_alert_history()
