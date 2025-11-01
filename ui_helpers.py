"""
ui_helpers.py
---------------------------------
Contains all Streamlit UI helper functions used by the main app.

Responsibilities:
    • Display stock data, alerts, and history.
    • Manage conditional visibility (e.g., hide Clear History button when empty).
    • Layout organization for readability and consistency.
"""

import streamlit as st
import pandas as pd


# ---------------------------
# 📘 Show Alert History
# ---------------------------
def show_alert_history():
    """
    Displays the alert history table and conditionally shows 'Clear History' button.
    """
    if "alert_history" not in st.session_state or st.session_state["alert_history"].empty:
        st.info("ℹ️ No alert history available yet.")
        return

    st.subheader("📜 Alert History")
    st.dataframe(
        st.session_state["alert_history"],
        use_container_width=True,
        hide_index=True
    )

    if not st.session_state["alert_history"].empty:
        if st.button("🧹 Clear History"):
            st.session_state["alert_history"] = pd.DataFrame(
                columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
            )
            st.toast("✅ Alert history cleared.")


# ---------------------------
# 📘 Display Stock Summary Table
# ---------------------------
def display_stock_summary(df: pd.DataFrame):
    """
    Renders the main stock summary table with signal highlights.

    Parameters
    ----------
    df : pd.DataFrame
        Must include ['Symbol', 'CMP', 'EMA200', 'RSI14', 'Signal'] columns.
    """
    if df.empty:
        st.warning("⚠️ No data to display. Please upload or load a valid watchlist.")
        return

    st.subheader("📊 Current Stock Summary")
    styled_df = df.style.apply(_highlight_signals, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


# ---------------------------
# 📘 Internal Helper - Color Coding
# ---------------------------
def _highlight_signals(row):
    """Applies row-level color highlights for BUY/SELL/WATCH signals."""
    color = ""
    if row["Signal"] == "BUY":
        color = "background-color: #b7f7b7"
    elif row["Signal"] == "SELL":
        color = "background-color: #f7b7b7"
    else:
        color = "background-color: #f0f0f0"
    return [color] * len(row)


# ---------------------------
# 📘 Upload Watchlist File
# ---------------------------
def handle_file_upload(upload_callback):
    """
    Allows the user to upload a new Excel watchlist.
    Upon upload, triggers the provided callback function to push to GitHub.

    Parameters
    ----------
    upload_callback : callable
        Function to call with the uploaded file object.
    """
    uploaded_file = st.file_uploader("📂 Upload new watchlist (Excel)", type=["xlsx"])
    if uploaded_file is not None:
        st.info("⏫ Uploading new file to GitHub...")
        success = upload_callback(uploaded_file)
        if success:
            st.toast("✅ Watchlist replaced successfully!")
        else:
            st.error("🚫 Upload failed.")
