# app.py - Indian Stock Auto Tracker (Streamlit + GitHub + Telegram)
import streamlit as st
import pandas as pd
import yfinance as yf
import io
import requests
import time
from typing import Optional

try:
    from github import Github
    HAS_PYGITHUB = True
except Exception:
    HAS_PYGITHUB = False

st.set_page_config(page_title="📈 Indian Stock Auto Tracker", layout="wide")

# -----------------------
# Secret handling helper
# -----------------------
def get_secret(key: str, section: Optional[str] = None, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    try:
        if section and section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        pass
    try:
        if section and section in st.secrets:
            sec = st.secrets[section]
            for trial in (key, key.upper()):
                if trial in sec:
                    return sec[trial]
    except Exception:
        pass
    return default

# -----------------------
# Load secrets
# -----------------------
TELEGRAM_TOKEN = get_secret("TELEGRAM_TOKEN", "telegram")
CHAT_ID = get_secret("CHAT_ID", "telegram")
GITHUB_TOKEN = get_secret("GITHUB_TOKEN", "github")
GITHUB_REPO = get_secret("GITHUB_REPO", "github")
GITHUB_FILE_PATH = get_secret("GITHUB_FILE_PATH", "github") or "watchlist.xlsx"

# Sidebar info
st.sidebar.header("Configuration")
if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.warning("⚠️ Telegram not set")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.info("Checking GitHub token...")
else:
    st.sidebar.error("GitHub token or repo missing")

# -----------------------
# Validate GitHub Token
# -----------------------
def check_github_token(token: str):
    if not token:
        return {"ok": False, "msg": "No token provided"}
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get("https://api.github.com/user", headers=headers, timeout=8)
        if r.status_code == 200:
            scopes = r.headers.get("X-OAuth-Scopes", "")
            return {"ok": True, "msg": "Valid", "scopes": scopes}
        else:
            msg = r.json().get("message", f"HTTP {r.status_code}")
            return {"ok": False, "msg": msg}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

check = check_github_token(GITHUB_TOKEN)
if check["ok"]:
    st.sidebar.success("✅ GitHub connected")
    if check.get("scopes"):
        st.sidebar.caption(f"Scopes: {check['scopes']}")
else:
    st.sidebar.error(f"❌ GitHub token error: {check['msg']}")

# -----------------------
# Load watchlist (Excel)
# -----------------------
def load_watchlist_from_github(token, repo_name, file_path):
    if not token or not repo_name:
        return None
    if HAS_PYGITHUB:
        try:
            g = Github(token)
            repo = g.get_repo(repo_name)
            file = repo.get_contents(file_path)
            df = pd.read_excel(io.BytesIO(file.decoded_content))
            return df
        except Exception as e:
            st.write(f"PyGithub error: {e}")
    # fallback REST API
    try:
        owner, repo = repo_name.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            df = pd.read_excel(io.BytesIO(r.content))
            return df
        else:
            msg = r.json().get("message", r.text)
            st.warning(f"GitHub API error: {msg}")
    except Exception as e:
        st.error(f"Failed to load watchlist: {e}")
    return None

@st.cache_data(ttl=300)
def get_watchlist():
    return load_watchlist_from_github(GITHUB_TOKEN, GITHUB_REPO, GITHUB_FILE_PATH)

watchlist_df = get_watchlist()

# -----------------------
# UI
# -----------------------
st.title("📊 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")

if watchlist_df is None or watchlist_df.empty:
    st.warning(f"⚠️ No valid `{GITHUB_FILE_PATH}` found in `{GITHUB_REPO}`")
    st.table(pd.DataFrame({"Symbol": ["RELIANCE.NS", "TCS.NS"]}))
    st.stop()

# clean up columns
watchlist_df.columns = [c.strip() for c in watchlist_df.columns]
if "Symbol" not in watchlist_df.columns:
    st.error("Excel must contain a column named 'Symbol'")
    st.stop()

st.success(f"Loaded {len(watchlist_df)} symbols from GitHub ✅")
st.dataframe(watchlist_df)

# -----------------------
# Telegram
# -----------------------
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        st.sidebar.error("Telegram not configured")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
        if r.status_code == 200:
            st.sidebar.success("Message sent to Telegram ✅")
        else:
            st.sidebar.error(f"Failed ({r.status_code})")
    except Exception as e:
        st.sidebar.error(str(e))

# -----------------------
# Stock logic
# -----------------------
def calc_indicators(symbol):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
    except Exception as e:
        st.write(f"{symbol} error: {e}")
        return None
    if df.empty:
        return None
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def analyze(symbol):
    df = calc_indicators(symbol)
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    close, ema, rsi = last["Close"], last["EMA200"], last["RSI"]
    signal = "NEUTRAL"
    if close > ema and 30 <= rsi <= 40:
        signal = "BUY"
    elif close < ema and 60 <= rsi <= 70:
        signal = "SELL"
    return {"Symbol": symbol, "Close": round(close,2), "EMA200": round(ema,2), "RSI": round(rsi,2), "Signal": signal}

# -----------------------
# Run Controls
# -----------------------
st.subheader("🔍 Scanner Controls")
col1, col2 = st.columns(2)
with col1:
    run_now = st.button("Run Scan")
    auto_mode = st.checkbox("Auto scan loop")
with col2:
    interval = st.number_input("Interval (seconds)", min_value=30, value=60)

def run_scan():
    rows, alerts = [], []
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()
    for s in symbols:
        res = analyze(s)
        if res:
            rows.append(res)
            if res["Signal"] in ["BUY", "SELL"]:
                alerts.append(f"{res['Symbol']}: {res['Signal']} (RSI={res['RSI']})")
    df = pd.DataFrame(rows)
    st.table(df)
    if alerts:
        msg = "⚠️ Alerts:\n" + "\n".join(alerts)
        st.warning(msg)
        send_telegram(msg)

if run_now:
    run_scan()

if auto_mode:
    st.warning("Running continuous loop. Stop manually.")
    while True:
        run_scan()
        time.sleep(interval)
