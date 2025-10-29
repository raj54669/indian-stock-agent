import streamlit as st
import pandas as pd
import requests
import io
import yfinance as yf
import ta
import time

# ---- Load secrets ----
TELEGRAM_TOKEN = st.secrets["telegram"]["TELEGRAM_TOKEN"]
CHAT_ID = st.secrets["telegram"]["CHAT_ID"]

GITHUB_TOKEN = st.secrets["github"]["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["github"]["GITHUB_REPO"]
GITHUB_FILE_PATH = "watchlist.xlsx"

# ---- Telegram alert ----
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        st.error(f"Telegram send failed: {e}")

# ---- GitHub helper ----
def load_watchlist_from_github():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        file_content = r.json()["content"]
        decoded = io.BytesIO(base64.b64decode(file_content))
        return pd.read_excel(decoded)
    else:
        st.warning(f"⚠️ GitHub load failed ({r.status_code}).")
        return None

def upload_watchlist_to_github(file):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    get_resp = requests.get(url, headers=headers)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

    encoded = base64.b64encode(file.read()).decode()
    data = {
        "message": "Update watchlist.xlsx via Streamlit app",
        "content": encoded,
    }
    if sha:
        data["sha"] = sha

    resp = requests.put(url, headers=headers, json=data)
    if resp.status_code in (200, 201):
        st.success("✅ Uploaded new watchlist.xlsx to GitHub.")
    else:
        st.error(f"GitHub upload failed: {resp.json()}")

# ---- Streamlit UI ----
st.sidebar.header("⚙️ Settings")
st.sidebar.success("All credentials loaded from app secrets.")

uploaded = st.sidebar.file_uploader("Upload new watchlist (.xlsx)", type="xlsx")
if uploaded:
    upload_watchlist_to_github(uploaded)

df = load_watchlist_from_github()
if df is not None and "Symbol" in df.columns:
    st.success(f"Loaded {len(df)} stocks from GitHub. Tracking started.")
    # You can call your stock-tracking logic here
else:
    st.warning("⚠️ No valid Excel found. Upload or ensure `watchlist.xlsx` exists in GitHub.")
