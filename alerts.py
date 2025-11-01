"""
alerts.py
---------------------------------
Handles alert management and Telegram messaging for the Indian Stock Agent app.

Responsibilities:
    • Manage alert history in Streamlit session state.
    • Send formatted alerts to Telegram.
    • Expose safe helper functions for the UI layer.
"""

import pandas as pd
import requests
import streamlit as st
from datetime import datetime
from pytz import timezone

# Define timezone for India
IST = timezone("Asia/Kolkata")


# ---------------------------
# 📘 Initialize Alert History
# ---------------------------
def init_alert_history():
    """
    Initializes an empty alert history DataFrame in Streamlit session state
    if not already created or corrupted.
    """
    if "alert_history" not in st.session_state or not isinstance(st.session_state["alert_history"], pd.DataFrame):
        st.session_state["alert_history"] = pd.DataFrame(
            columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
        )


# ---------------------------
# 📘 Add Alert Entry
# ---------------------------
def add_to_alert_history(symbol: str, signal: str, cmp_: float, ema200: float, rsi14: float):
    """
    Appends a new alert to the session's alert history.

    Parameters
    ----------
    symbol : str
        Stock symbol or name.
    signal : str
        Generated signal ('BUY', 'SELL', 'WATCH').
    cmp_ : float
        Current market price.
    ema200 : float
        EMA200 value.
    rsi14 : float
        RSI14 value.
    """
    init_alert_history()  # Ensure structure exists

    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame([{
        "Date & Time (IST)": ts,
        "Symbol": symbol,
        "Signal": signal,
        "CMP": round(float(cmp_), 2),
        "EMA200": round(float(ema200), 2),
        "RSI14": round(float(rsi14), 2),
    }])

    # Safely append to DataFrame
    st.session_state["alert_history"] = pd.concat(
        [st.session_state["alert_history"], new_row],
        ignore_index=True
    )


# ---------------------------
# 📘 Clear Alert History
# ---------------------------
def clear_alert_history():
    """Resets alert history in Streamlit session state."""
    st.session_state["alert_history"] = pd.DataFrame(
        columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
    )


# ---------------------------
# 📘 Telegram Alert Sender
# ---------------------------
def send_telegram_alert(symbol: str, signal: str, cmp_: float, ema200: float, rsi14: float):
    """
    Sends a formatted alert message to Telegram using configured secrets.

    Secrets expected in .streamlit/secrets.toml:
        [telegram]
        TELEGRAM_TOKEN = "<your_bot_token>"
        CHAT_ID = "<your_chat_id>"
    """
    try:
        token = st.secrets["telegram"]["TELEGRAM_TOKEN"]
        chat_id = st.secrets["telegram"]["CHAT_ID"]
    except Exception:
        st.warning("⚠️ Telegram credentials not found in secrets.toml")
        return

    message = (
        f"📈 *{symbol}* generated a *{signal}* signal!\n\n"
        f"💰 CMP: ₹{cmp_}\n"
        f"📊 EMA200: ₹{ema200}\n"
        f"📉 RSI14: {rsi14}\n"
        f"🕒 {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

    try:
        requests.post(url, data=payload, timeout=10)
    except requests.RequestException as e:
        st.error(f"🚫 Telegram alert failed: {e}")
