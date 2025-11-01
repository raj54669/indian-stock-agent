import os
import io
import pandas as pd
from github import Github
import streamlit as st
import requests

# --------------------------------------------
# 🔔 Telegram Utilities
# --------------------------------------------
def send_telegram(message: str):
    """Send a Telegram alert using bot token and chat ID."""
    token = st.secrets["TELEGRAM"]["BOT_TOKEN"]
    chat_id = st.secrets["TELEGRAM"]["CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Telegram send failed: {e}")
        return False


# --------------------------------------------
# 📂 GitHub Utilities
# --------------------------------------------
def load_excel_from_github():
    """Load the default watchlist Excel file from GitHub."""
    repo_name = st.secrets["GITHUB"]["REPO_NAME"]
    file_path = st.secrets["GITHUB"]["WATCHLIST_PATH"]
    token = st.secrets["GITHUB"]["TOKEN"]

    g = Github(token)
    repo = g.get_repo(repo_name)
    file_content = repo.get_contents(file_path)
    content = io.BytesIO(file_content.decoded_content)
    df = pd.read_excel(content)
    return df


def upload_to_github(uploaded_file):
    """
    Replace the existing GitHub watchlist Excel file with a new uploaded file.
    """
    repo_name = st.secrets["GITHUB"]["REPO_NAME"]
    file_path = st.secrets["GITHUB"]["WATCHLIST_PATH"]
    token = st.secrets["GITHUB"]["TOKEN"]

    g = Github(token)
    repo = g.get_repo(repo_name)

    content = uploaded_file.read()
    try:
        existing_file = repo.get_contents(file_path)
        repo.update_file(
            path=file_path,
            message="♻️ Updated watchlist via Streamlit app",
            content=content,
            sha=existing_file.sha,
            branch="main",
        )
        st.success("✅ Watchlist successfully updated in GitHub!")
    except Exception as e:
        st.error(f"GitHub upload failed: {e}")
