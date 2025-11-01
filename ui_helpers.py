import streamlit as st
import pandas as pd

def show_alert_history():
    if 'alert_history' not in st.session_state or st.session_state['alert_history'].empty:
        st.info('No alerts recorded yet.')
        return
    st.subheader('📜 Alert History')
    st.dataframe(st.session_state['alert_history'], use_container_width=True, hide_index=True)
    if st.button('🧹 Clear History'):
        st.session_state['alert_history'] = pd.DataFrame(columns=['Date & Time (IST)','Symbol','Signal','CMP','EMA200','RSI14'])
        st.success('Alert history cleared.')
