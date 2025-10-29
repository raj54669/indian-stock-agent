# app.py - Robust GitHub token checking + watchlist loader + Telegram alert example
import streamlit as st
import pandas as pd
import yfinance as yf
import io
import requests
import time
from typing import Optional

# Try import PyGithub but don't fail if not installed
try:
    from github import Github
    HAS_PYGITHUB = True
except Exception:
    HAS_PYGITHUB = False

st.set_page_config(page_title="Indian Stock Auto Tracker", layout="wide")

# -----------------------
# Helpers to read secrets (supports flat and sectioned TOML)
# -----------------------
def get_secret_flat_or_section(key: str, section: Optional[str] = None, default=None):
    # Try flat key first
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # If section provided, try st.secrets[section][key]
    try:
        if section and section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        pass
    # try nested with lower-case as some people put [github] -> GITHUB_TOKEN etc.
    try:
        if section and section in st.secrets:
            sec = st.secrets[section]
            # support both str keys and uppercase keys
            for trial in (key, key.upper()):
                if trial in sec:
                    return sec[trial]
    except Exception:
        pass
    return default

# Load Telegram secrets (support either flat or [telegram])
TELEGRAM_TOKEN = get_secret_flat_or_section("TELEGRAM_TOKEN", section="telegram")
CHAT_ID = get_secret_flat_or_section("CHAT_ID", section="telegram")

# Load GitHub secrets (support either flat or [github])
GITHUB_TOKEN = get_secret_flat_or_section("GITHUB_TOKEN", section="github")
GITHUB_REPO = get_secret_flat_or_section("GITHUB_REPO", section="github")
GITHUB_FILE_PATH = get_secret_flat_or_section("GITHUB_FILE_PATH", section="github") or "watchlist.xlsx"

# show masked info in sidebar (non-sensitive)
st.sidebar.header("Settings status")
if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("Telegram configured")
else:
    st.sidebar.warning("Telegram secrets not set - alerts disabled")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.info("GitHub token & repo present — validating...")
else:
    st.sidebar.error("GitHub token or repo missing in secrets")

# -----------------------
# GitHub token validation helper (REST call)
# -----------------------
def github_token_check(token: str) -> dict:
    """Return dict with 'ok' (bool) and message and scopes list if available."""
    if not token:
        return {"ok": False, "msg": "No token provided"}
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get("https://api.github.com/user", headers=headers, timeout=8)
        if r.status_code == 200:
            scopes = r.headers.get("X-OAuth-Scopes", "")
            return {"ok": True, "msg": "Token valid", "scopes": scopes}
        else:
            # return response JSON message if available
            try:
                j = r.json()
                msg = j.get("message", f"HTTP {r.status_code}")
            except Exception:
                msg = f"HTTP {r.status_code}"
            return {"ok": False, "msg": f"{msg} (status {r.status_code})", "status_code": r.status_code}
    except Exception as e:
        return {"ok": False, "msg": f"Network / request error: {e}"}

# Run token check and show results
gh_check = github_token_check(GITHUB_TOKEN)
if gh_check["ok"]:
    st.sidebar.success("GitHub token valid")
    if "scopes" in gh_check:
        st.sidebar.caption(f"Scopes: {gh_check.get('scopes')}")
else:
    st.sidebar.error(f"GitHub connection failed: {gh_check.get('msg')}")

# -----------------------
# Load watchlist: prefer PyGithub if available and repo usable; fallback to REST raw content
# -----------------------
def load_watchlist_from_pygithub(token: str, repo_name: str, path: str) -> Optional[pd.DataFrame]:
    try:
        if not HAS_PYGITHUB:
            return None
        g = Github(token)
        repo = g.get_repo(repo_name)
        file_content = repo.get_contents(path)
        raw = file_content.decoded_content
        df = pd.read_excel(io.BytesIO(raw))
        return df
    except Exception as e:
        st.write(f"PyGithub load failed: {e}")
        return None

def load_watchlist_from_api(token: str, repo_name: str, path: str) -> Optional[pd.DataFrame]:
    # Use GitHub contents raw endpoint with token
    try:
        owner, repo = repo_name.split("/", 1)
    except Exception:
        st.write("Repo name must be owner/repo")
        return None
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 200:
        try:
            raw_bytes = r.content
            df = pd.read_excel(io.BytesIO(raw_bytes))
            return df
        except Exception as e:
            st.write(f"Failed to parse Excel from GitHub: {e}")
            return None
    else:
        try:
            j = r.json()
            msg = j.get("message", r.text)
        except Exception:
            msg = r.text
        st.write(f"GitHub API load failed: {msg} (status {r.status_code})")
        return None

@st.cache_data(ttl=120)
def load_watchlist(token, repo_name, path):
    # Try PyGithub first (if installed and token looks valid)
    if HAS_PYGITHUB:
        df = load_watchlist_from_pygithub(token, repo_name, path)
        if df is not None:
            return df
    # Fallback to REST API
    df = load_watchlist_from_api(token, repo_name, path)
    return df

# Attempt load if token/repo present
watchlist_df = None
if GITHUB_TOKEN and GITHUB_REPO:
    watchlist_df = load_watchlist(GITHUB_TOKEN, GITHUB_REPO, GITHUB_FILE_PATH)

# -----------------------
# UI: show status and instructions
# -----------------------
st.title("📈 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")

if not watchlist_df:
    st.warning(f"No valid Excel found in GitHub. Please upload a file named `{GITHUB_FILE_PATH}` with a 'Symbol' column in repo `{GITHUB_REPO}`.")
    st.markdown("**Example format:**")
    st.table(pd.DataFrame({"Symbol": ["RELIANCE.NS", "TCS.NS"]}))
else:
    # normalize columns
    watchlist_df.columns = [c.strip() for c in watchlist_df.columns]
    if "Symbol" not in watchlist_df.columns:
        st.error("Excel loaded but did not contain a 'Symbol' column. Please rename the column to 'Symbol'.")
    else:
        st.success(f"Loaded {len(watchlist_df)} symbols from GitHub")
        st.dataframe(watchlist_df.head(200))

# -----------------------
# Telegram helper (simple)
# -----------------------
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        st.error("Telegram not configured")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=10)
        if r.status_code == 200:
            st.sidebar.success("Telegram message sent")
            return True
        else:
            st.sidebar.error(f"Telegram send failed: {r.status_code}")
            return False
    except Exception as e:
        st.sidebar.error(f"Telegram exception: {e}")
        return False

# -----------------------
# Indicators & scanner (minimal)
# -----------------------
def calc_rsi_ema(symbol: str):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
    except Exception as e:
        st.write(f"yfinance error for {symbol}: {e}")
        return None
    if df is None or df.empty:
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
    df = calc_rsi_ema(symbol)
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    close = float(last["Close"])
    ema = float(last["EMA200"])
    rsi = float(last["RSI"])
    signal = "Neutral"
    if close > ema and rsi < 30:
        signal = "BUY"
    elif close < ema and rsi > 70:
        signal = "SELL"
    return {"Symbol": symbol, "Close": round(close,2), "EMA200": round(ema,2), "RSI": round(rsi,2), "Signal": signal}

# -----------------------
# Controls
# -----------------------
st.subheader("Controls")
col1, col2 = st.columns([1,2])
with col1:
    run_now = st.button("Run Scan Now")
    auto = st.checkbox("Enable auto-scan loop (not recommended on Cloud)", key="auto_scan")
    interval = st.number_input("Auto interval (sec)", value=60, step=10)
with col2:
    st.write("Status & logs:")
    if gh_check["ok"]:
        st.write("- GitHub token OK")
    else:
        st.write(f"- GitHub token problem: {gh_check['msg']}")

def run_scan_once():
    if watchlist_df is None or "Symbol" not in watchlist_df.columns:
        st.error("No watchlist available")
        return
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()
    rows = []
    alerts = []
    with st.spinner(f"Scanning {len(symbols)} symbols..."):
        for s in symbols:
            r = analyze(s)
            if r:
                rows.append(r)
                if r["Signal"] in ("BUY", "SELL"):
                    alerts.append(f"{r['Symbol']}: {r['Signal']} (RSI={r['RSI']}, Close={r['Close']})")
    if rows:
        st.table(pd.DataFrame(rows))
    else:
        st.info("No results")

    if alerts:
        st.warning("Alerts found:\n" + "\n".join(alerts))
        if TELEGRAM_TOKEN and CHAT_ID:
            send_telegram("\n".join(alerts))

# run on user action or auto loop
if run_now:
    run_scan_once()

if auto:
    st.warning("Auto-scan loop started (will block this session). Use with care.")
    while True:
        run_scan_once()
        time.sleep(max(5, int(interval)))
