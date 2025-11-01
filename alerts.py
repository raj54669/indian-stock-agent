import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import pytz

IST = pytz.timezone('Asia/Kolkata')

def init_alert_history():
    if 'alert_history' not in st.session_state or not isinstance(st.session_state['alert_history'], pd.DataFrame):
        st.session_state['alert_history'] = pd.DataFrame(columns=['Date & Time (IST)','Symbol','Signal','CMP','EMA200','RSI14'])

def add_to_alert_history(symbol, signal, cmp_, ema200, rsi14):
    init_alert_history()
    ts = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
    new = pd.DataFrame([{
        'Date & Time (IST)': ts,
        'Symbol': symbol,
        'Signal': signal,
        'CMP': round(float(cmp_),2),
        'EMA200': round(float(ema200),2),
        'RSI14': round(float(rsi14),2)
    }])
    st.session_state['alert_history'] = pd.concat([st.session_state['alert_history'], new], ignore_index=True)

def send_telegram(message: str) -> bool:
    # Try secrets first
    try:
        token = st.secrets['telegram']['TELEGRAM_TOKEN']
        chat_id = st.secrets['telegram']['CHAT_ID']
    except Exception:
        return False
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = {'chat_id': chat_id, 'text': message}
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False
