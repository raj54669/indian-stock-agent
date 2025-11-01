"""
github_utils.py
---------------------------------
Provides helper functions to:
    • Load the watchlist Excel file from a GitHub repository.
    • Replace (upload) a new Excel file to GitHub, committing the update.
    • Handle GitHub authentication, error recovery, and file integrity.

Requires secrets in .streamlit/secrets.toml:
    GITHUB_TOKEN
    GITHUB_REPO
    GITHUB_BRANCH
    GITHUB_FILE_PATH
"""

import base64
import io
import pandas as pd
import requests
import streamlit as st


# ---------------------------
# 📘 Load Watchlist from GitHub
# ---------------------------
def load_watchlist_from_github() -> pd.DataFrame:
    """
    Fetches and reads the Excel file from a GitHub repository.

    Returns
    -------
    pd.DataFrame
        The loaded watchlist DataFrame.
    """
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
        branch = st.secrets["GITHUB_BRANCH"]
        file_path = st.secrets["GITHUB_FILE_PATH"]
    except Exception:
        st.error("❌ Missing GitHub credentials in secrets.toml")
        return pd.DataFrame()

    api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}?ref={branch}"
    headers = {"Authorization": f"token {token}"}

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        content = response.json().get("content", "")
        decoded = base64.b64decode(content)
        df = pd.read_excel(io.BytesIO(decoded))
        st.success(f"✅ Watchlist loaded from GitHub: {file_path}")
        return df
    except Exception as e:
        st.warning(f"⚠️ Failed to load watchlist from GitHub: {e}")
        return pd.DataFrame()


# ---------------------------
# 📘 Replace (Upload) Watchlist to GitHub
# ---------------------------
def upload_watchlist_to_github(uploaded_file) -> bool:
    """
    Replaces the watchlist Excel file on GitHub with the uploaded file.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The Streamlit uploaded file object.

    Returns
    -------
    bool
        True if upload succeeded, False otherwise.
    """
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
        branch = st.secrets["GITHUB_BRANCH"]
        file_path = st.secrets["GITHUB_FILE_PATH"]
    except Exception:
        st.error("❌ Missing GitHub credentials in secrets.toml")
        return False

    # Step 1: Get the current file SHA (required for commit update)
    get_url = f"https://api.github.com/repos/{repo}/contents/{file_path}?ref={branch}"
    headers = {"Authorization": f"token {token}"}

    try:
        response = requests.get(get_url, headers=headers)
        response.raise_for_status()
        sha = response.json()["sha"]
    except Exception:
        sha = None  # File may not exist yet

    # Step 2: Encode uploaded file content
    content = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

    # Step 3: Prepare PUT request body
    commit_msg = f"📦 Updated watchlist via Streamlit upload"
    payload = {
        "message": commit_msg,
        "content": content,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    put_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"

    try:
        res = requests.put(put_url, headers=headers, json=payload)
        res.raise_for_status()
        st.success("✅ Watchlist successfully updated in GitHub.")
        return True
    except Exception as e:
        st.error(f"🚫 Failed to upload file to GitHub: {e}")
        return False
