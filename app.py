import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import time
import io
import base64
import os
import requests
from github import Github

# ---------- Load secrets ----------
TELEGRAM_TOKEN = st.secrets["telegram"]["TELEGRAM_TOKEN"]
CHAT_ID = st.secrets["telegram"]["CHAT_ID"]

GITHUB_TOKEN = st.secrets["github"]["GITHUB_TOKEN"]
GITHUB_REPO_NAME = st.secrets["github"]["GITHUB_REPO"]
GITHUB_BRANCH = "main"

# ---------- GitHub Repo Initialization ----------
GITHUB_REPO = None
if GITHUB_TOKEN:
    try:
        gh = Github(GITHUB_TOKEN)
        GITHUB_REPO = gh.get_repo(GITHUB_REPO_NAME)
    except Exception as e:
        st.warning(f"⚠️ GitHub repo init failed: {e}")

WATCHLIST_FILE = "watchlist.xlsx"

# ---------- Telegram Helper ----------
def send_telegram_message(message: str):
    """Send Telegram notification message."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, data=data)
    except Exception as e:
        st.error(f"Telegram send failed: {e}")

# ---------- GitHub Helpers ----------
def load_excel_from_github():
    """Load Excel file from GitHub repo."""
    try:
        contents = GITHUB_REPO.get_contents(WATCHLIST_FILE, ref=GITHUB_BRANCH)
        decoded = base64.b64decode(contents.content)
        df = pd.read_excel(io.BytesIO(decoded))
        st.success(f"✅ Loaded {len(df)} symbols from GitHub.")
        return df
    except Exception as e:
        st.warning(f"⚠️ Error loading from GitHub: {e}")
        return None


def upload_excel_to_github(file_data):
    """Upload or update Excel file in GitHub repo."""
    try:
        contents = GITHUB_REPO.get_contents(WATCHLIST_FILE, ref=GITHUB_BRANCH)
        sha = contents.sha
        GITHUB_REPO.update_file(
            WATCHLIST_FILE,
            "Update watchlist.xlsx via Streamlit",
            file_data.getvalue(),
            sha,
            branch=GITHUB_BRANCH,
        )
        st.success("✅ Updated watchlist.xlsx in GitHub.")
    except Exception:
        try:
            GITHUB_REPO.create_file(
                WATCHLIST_FILE,
                "Add watchlist.xlsx via Streamlit",
                file_data.getvalue(),
                branch=GITHUB_BRANCH,
            )
            st.success("✅ Uploaded new watchlist.xlsx to GitHub.")
        except Exception as e:
            st.error(f"❌ GitHub upload failed: {e}")


# ---------- RSI + EMA Logic ----------
def check_stock_conditions(symbol):
    """Check if stock meets RSI and EMA proximity conditions."""
    try:
        data = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if data.empty:
            return None

        data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
        data["EMA200"] = ta.trend.EMAIndicator(data["Close"], window=200).ema_indicator()

        last = data.iloc[-1]
        price = last["Close"]
        rsi = last["RSI"]
        ema = last["EMA200"]

        near_ema = ema * 0.98 <= price <= ema * 1.02
        rsi_ok = 30 < rsi < 40

        return {"Symbol": symbol, "Price": price, "RSI": rsi, "EMA200": ema, "NearEMA": near_ema, "RSI_OK": rsi_ok}

    except Exception as e:
        return {"Symbol": symbol, "Error": str(e)}


# ---------- Streamlit UI ----------
st.sidebar.header("⚙️ Settings")
st.sidebar.success("All credentials loaded from app secrets.")

uploaded = st.sidebar.file_uploader("Upload new watchlist (.xlsx)", type=["xlsx"])

if uploaded:
    upload_excel_to_github(uploaded)
    df = pd.read_excel(uploaded)
else:
    df = load_excel_from_github()

st.title("📈 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")

if df is not None and "Symbol" in df.columns:
    results = []
    progress = st.progress(0)
    total = len(df)

    for i, row in enumerate(df.itertuples(), start=1):
        res = check_stock_conditions(row.Symbol)
        if res:
            results.append(res)
        progress.progress(i / total)

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    triggered = results_df[(results_df["NearEMA"]) & (results_df["RSI_OK"])]
    if not triggered.empty:
        msg = "🚨 Stock Alert 🚨\n" + "\n".join(triggered["Symbol"])
        send_telegram_message(msg)
        st.success("Telegram alert sent!")

else:
    st.warning("⚠️ No valid Excel found. Please upload one with a 'Symbol' column.")
