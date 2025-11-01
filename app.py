import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from indicators import calc_rsi_ema, analyze
from alerts import send_telegram, add_to_alert_history
from github_utils import load_watchlist_from_github, upload_watchlist_to_github
from ui_helpers import show_alert_history

st.set_page_config(page_title='📈 Indian Stock Agent – EMA + RSI Alert Bot', layout='wide')
st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# init session state
if 'alert_history' not in st.session_state:
    st.session_state['alert_history'] = pd.DataFrame(columns=['Date & Time (IST)','Symbol','Signal','CMP','EMA200','RSI14'])

# Sidebar
st.sidebar.header('📂 Watchlist Management')
uploaded_file = st.sidebar.file_uploader('Upload new watchlist (Excel)', type=['xlsx'])
use_uploaded = False
watchlist_df = pd.DataFrame()
if uploaded_file is not None:
    try:
        df_up = pd.read_excel(uploaded_file)
        if 'Symbol' not in df_up.columns:
            st.sidebar.error("Uploaded file must contain a 'Symbol' column.")
        else:
            watchlist_df = df_up
            use_uploaded = True
            st.sidebar.success(f"✅ Using uploaded watchlist ({len(watchlist_df)} symbols)")
            # push to github
            try:
                upload_watchlist_to_github(uploaded_file)
            except Exception:
                pass
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")

if not use_uploaded:
    watchlist_df = load_watchlist_from_github()
    st.sidebar.info('Using GitHub watchlist as default source')

st.sidebar.header('Settings')

if 'telegram' in st.secrets and 'TELEGRAM_TOKEN' in st.secrets['telegram']:
    st.sidebar.success('✅ Telegram configured')
else:
    st.sidebar.info('Telegram not configured — alerts disabled')

if 'GITHUB_TOKEN' in st.secrets and 'GITHUB_REPO' in st.secrets:
    st.sidebar.success('✅ GitHub secrets present')
else:
    st.sidebar.warning('GitHub credentials missing (set GITHUB_TOKEN and GITHUB_REPO)')

# sanitize watchlist
if not watchlist_df.empty and 'Symbol' in watchlist_df.columns:
    watchlist_df['Symbol'] = watchlist_df['Symbol'].astype(str).str.strip()
    watchlist_df = watchlist_df[watchlist_df['Symbol']!='']
else:
    watchlist_df = pd.DataFrame()

# Main UI
st.title('📊 Indian Stock Agent – EMA + RSI Alert Bot')
if watchlist_df.empty or 'Symbol' not in watchlist_df.columns:
    st.warning("⚠️ No valid 'Symbol' column found in your watchlist Excel file.")
    symbols = []
else:
    symbols = watchlist_df['Symbol'].dropna().astype(str).tolist()

# summary placeholder
st.subheader('📋 Combined Summary Table')
initial_df = pd.DataFrame({
    'Symbol': symbols if symbols else [],
    'CMP': ['' for _ in symbols] if symbols else [],
    '52W_Low': ['' for _ in symbols] if symbols else [],
    '52W_High': ['' for _ in symbols] if symbols else [],
    'EMA200': ['' for _ in symbols] if symbols else [],
    'RSI14': ['' for _ in symbols] if symbols else [],
    'Signal': ['' for _ in symbols] if symbols else [],
})
summary_placeholder = st.empty()
summary_placeholder.dataframe(initial_df, use_container_width=True, hide_index=True)
last_scan_time = st.caption('Will auto-update after scanning.')

# Controls
st.subheader('⚙️ Controls')
col1, col2 = st.columns([1,2])
with col1:
    run_now = st.button('Run Scan Now', key='run_now_btn')
    interval = st.number_input('Interval (sec)', value=60, step=5, min_value=5, key='interval_input')
    auto = st.checkbox('Enable Auto-scan', key='auto_chk')
with col2:
    st.markdown('**Status:**')
    st.write(f"- GitHub Repo: `{st.secrets.get('GITHUB_REPO','N/A')}`")
    st.write(f"- Token: {'✅' if st.secrets.get('GITHUB_TOKEN') else '❌'}")
    if auto:
        st.markdown(f"<span style='margin-left:10px;'>🔁 Auto-scan active — every {interval} seconds</span>", unsafe_allow_html=True)

# run_scan function - uses calc_rsi_ema from indicators
def run_scan():
    results = []
    for symbol in symbols:
        try:
            df = yf.download(symbol, period='1y', interval='1d', progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            df_ind = calc_rsi_ema(df)
            last = df_ind.iloc[-1]
            cmp_ = float(last['Close'])
            ema200 = float(last['EMA200'])
            rsi14 = float(last['RSI14'])
            high_52w = float(last['52W_High'])
            low_52w = float(last['52W_Low'])

            signal = 'Neutral'
            if cmp_ > ema200 and rsi14 < 30:
                signal = '🟢 BUY'
            elif cmp_ < ema200 and rsi14 > 70:
                signal = '🔴 SELL'
            elif abs(cmp_ - ema200)/cmp_ <= 0.02 and 30 <= rsi14 <= 40:
                signal = '🟡 WATCH'

            if signal in ('🟢 BUY','🔴 SELL','🟡 WATCH'):
                add_to_alert_history(symbol, signal, cmp_, ema200, rsi14)
                # prepare message
                emoji = '⚡' if 'WATCH' in signal else ('📈' if 'BUY' in signal else '📉')
                alert_msg = f"{emoji} *Alert:* {symbol}\nCMP = {cmp_:.2f}\nEMA200 = {ema200:.2f}\nRSI14 = {rsi14:.2f}\nCondition: {signal}"
                send_telegram(alert_msg)

            results.append({
                'Symbol': symbol,
                'CMP': round(cmp_,2),
                '52W_Low': round(low_52w,2),
                '52W_High': round(high_52w,2),
                'EMA200': round(ema200,2),
                'RSI14': round(rsi14,2),
                'Signal': signal
            })
        except Exception as e:
            st.error(f"Error for {symbol}: {e}")

    if results:
        df_res = pd.DataFrame(results)
        summary_placeholder.dataframe(df_res, use_container_width=True, hide_index=True)
        ist = timezone(timedelta(hours=5, minutes=30))
        last_scan_time.caption(f"Last scan: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        summary_placeholder.warning('No valid data fetched.')

# Buttons and auto-scan
if run_now:
    run_scan()

test_telegram = st.button('📨 Send Test Telegram Alert')
if test_telegram:
    ok = send_telegram('✅ Test alert from Indian Stock Agent Bot!')
    if ok:
        st.success('✅ Telegram test alert sent successfully!')
    else:
        st.error('❌ Telegram send failed. Check your token or chat_id.')

if auto:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=int(interval)*1000, key='auto_refresh')
        run_scan()
    except Exception:
        st.warning('Auto-refresh package missing. Run: pip install streamlit-autorefresh')

st.divider()
show_alert_history()
