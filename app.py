# app.py
import streamlit as st
import pandas as pd
import time
import io
import requests
from datetime import datetime, timedelta

# External libs (ensure these are in requirements.txt)
# yfinance, ta, PyGithub, openpyxl must be installed
import yfinance as yf
import ta
from github import Github, GithubException

# ---------------------------
# Config / Secrets (from Streamlit TOML)
# ---------------------------
def get_secret_section(name, default=None):
    try:
        return st.secrets.get(name, default) if isinstance(st.secrets, dict) else default
    except Exception:
        return default

telegram_secrets = get_secret_section("telegram", {})
github_secrets = get_secret_section("github", {})

TELEGRAM_TOKEN = telegram_secrets.get("TELEGRAM_TOKEN") or st.secrets.get("TELEGRAM_TOKEN", None)
CHAT_ID = telegram_secrets.get("CHAT_ID") or st.secrets.get("CHAT_ID", None)

GITHUB_TOKEN = github_secrets.get("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN", None)
GITHUB_REPO = github_secrets.get("GITHUB_REPO") or st.secrets.get("GITHUB_REPO", None)
GITHUB_FILE_PATH = github_secrets.get("GITHUB_FILE_PATH", "watchlist.xlsx") or st.secrets.get("GITHUB_FILE_PATH", "watchlist.xlsx")
GITHUB_BRANCH = github_secrets.get("GITHUB_BRANCH", "main") or st.secrets.get("GITHUB_BRANCH", "main")

# ---------------------------
# Simple helpers
# ---------------------------
def send_telegram_message(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        st.warning("Telegram credentials not configured.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        r = requests.post(url, data=data, timeout=10)
        return r.ok
    except Exception as e:
        st.error(f"Telegram send error: {e}")
        return False

def connect_github():
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None, "GitHub token or repo not configured"
    try:
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)
        return repo, None
    except GithubException as ge:
        return None, f"GitHub error: {ge.data if hasattr(ge, 'data') else ge}"
    except Exception as e:
        return None, str(e)

def load_watchlist_from_github(repo, path, branch="main"):
    """Return a pandas DataFrame if ok, else raise."""
    file = repo.get_contents(path, ref=branch)
    raw = file.decoded_content
    stream = io.BytesIO(raw)
    df = pd.read_excel(stream)
    return df, file.sha

def upload_watchlist_to_github(repo, path, content_bytes, branch="main", message="Update watchlist.xlsx"):
    """Create or update file at path with content_bytes (bytes)."""
    try:
        existing = None
        try:
            existing = repo.get_contents(path, ref=branch)
        except GithubException as e:
            # 404 means not found — will create
            existing = None
        if existing:
            repo.update_file(existing.path, message, content_bytes, existing.sha, branch=branch)
        else:
            repo.create_file(path, message, content_bytes, branch=branch)
        return True, None
    except Exception as e:
        return False, str(e)

# ---------------------------
# Core computation: EMA 200 and RSI (14)
# ---------------------------
def compute_indicators(symbol, period_days=300):
    """Download history (period_days) and compute close, EMA200 and RSI(14). Return latest values."""
    try:
        ticker = yf.Ticker(symbol)
        # request enough days to compute 200 EMA (use 400 day window to be safe)
        hist = ticker.history(period=f"{period_days}d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return None, "No history"
        # Ensure 'Close' present
        df = hist.copy()
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None, "No close prices"
        close = df["Close"]
        # 200 EMA
        ema200 = close.ewm(span=200, adjust=False).mean()
        # RSI(14)
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        latest = {
            "price": float(close.iloc[-1]),
            "ema200": float(ema200.iloc[-1]) if len(ema200) >= 1 else None,
            "rsi14": float(rsi.iloc[-1]) if len(rsi) >= 1 else None,
            "last_date": close.index[-1].strftime("%Y-%m-%d"),
        }
        return latest, None
    except Exception as e:
        return None, str(e)

def check_conditions(latest, price_tolerance=0.02, rsi_low=30, rsi_high=40):
    """Return True if conditions met: RSI between [rsi_low,rsi_high] and price within ±price_tolerance of EMA200"""
    if not latest:
        return False, "No price data"
    price = latest.get("price")
    ema = latest.get("ema200")
    rsi = latest.get("rsi14")
    if price is None or ema is None or rsi is None:
        return False, "Incomplete indicators"
    near = (price > ema * (1 - price_tolerance)) and (price < ema * (1 + price_tolerance))
    rsi_ok = (rsi > rsi_low) and (rsi < rsi_high)
    reason = []
    if not near:
        reason.append(f"Price {price:.2f} not within ±{price_tolerance*100:.1f}% of EMA200 ({ema:.2f})")
    if not rsi_ok:
        reason.append(f"RSI {rsi:.2f} not between {rsi_low}-{rsi_high}")
    return (near and rsi_ok), "; ".join(reason)

# ---------------------------
# UI / Main
# ---------------------------
st.set_page_config(page_title="Indian Stock Auto Tracker (EMA + RSI)", layout="wide")

st.sidebar.markdown("## Settings")
secret_ok = True
if not TELEGRAM_TOKEN or not CHAT_ID:
    st.sidebar.warning("Telegram secrets not set. Alerts will be disabled.")
    secret_ok = False
else:
    st.sidebar.success("✅ All credentials loaded from app secrets.")

# Upload/watchlist management
st.sidebar.markdown("### Upload new watchlist (.xlsx)")
uploaded_file = st.sidebar.file_uploader("Drag and drop file (.xlsx) — this will replace the repo file", type=["xlsx"])

repo, repo_err = connect_github()
if repo_err:
    st.sidebar.error(f"GitHub init: {repo_err}")
else:
    st.sidebar.info(f"GitHub repo: {GITHUB_REPO} (branch {GITHUB_BRANCH})")

# Load watchlist from GitHub if exists
watchlist_df = None
watchlist_sha = None
gh_load_error = None

if repo:
    try:
        # try load from repo path
        file_contents = None
        try:
            file_contents = repo.get_contents(GITHUB_FILE_PATH, ref=GITHUB_BRANCH)
        except GithubException as ge:
            # not found or permissions
            gh_load_error = f"GitHub load error: {ge.data if hasattr(ge,'data') else ge}"
            file_contents = None

        if file_contents:
            # decode and read excel
            raw = file_contents.decoded_content
            watchlist_df = pd.read_excel(io.BytesIO(raw))
            watchlist_sha = file_contents.sha
    except Exception as e:
        gh_load_error = str(e)

# If user uploaded file via UI → upload to Github (if configured) and also use locally
if uploaded_file is not None:
    try:
        # read to dataframe to validate
        local_df = pd.read_excel(uploaded_file)
        if "Symbol" not in [c.strip() for c in local_df.columns]:
            st.sidebar.error("Uploaded Excel must contain a column named 'Symbol'.")
        else:
            # upload to GitHub
            if repo:
                try:
                    uploaded_file.seek(0)
                    bytes_content = uploaded_file.read()
                    ok, err = upload_watchlist_to_github(
                        repo, GITHUB_FILE_PATH, bytes_content, branch=GITHUB_BRANCH,
                        message=f"Uploaded watchlist via Streamlit UI @ {datetime.utcnow().isoformat()}Z"
                    )
                    if ok:
                        st.sidebar.success("GitHub upload succeeded - watchlist replaced.")
                        # update local watchlist_df to newly uploaded
                        watchlist_df = local_df.copy()
                    else:
                        st.sidebar.error(f"GitHub upload failed: {err}")
                except Exception as e:
                    st.sidebar.error(f"GitHub upload exception: {e}")
            else:
                st.sidebar.warning("GitHub not configured — only local upload available.")
                watchlist_df = local_df.copy()
    except Exception as e:
        st.sidebar.error(f"Failed reading uploaded Excel: {e}")

# Show any GitHub load warnings
if gh_load_error:
    st.warning(gh_load_error)

# If no watchlist (either in repo or uploaded), show message & small sample template
if watchlist_df is None:
    st.info("No valid Excel found in GitHub. Please upload a file named: " + GITHUB_FILE_PATH + " with a 'Symbol' column.")
    st.markdown(
        """
        **Required format example:**
        | Symbol |
        |--------|
        | RELIANCE.NS |
        | TCS.NS |
        """
    )
    # stop here (app will still allow uploads)
else:
    # normalize columns
    watchlist_df.columns = [c.strip() for c in watchlist_df.columns]
    if "Symbol" not in watchlist_df.columns:
        st.warning("Excel must contain 'Symbol' column.")
        watchlist_df = None

# Main layout
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("## Controls")
    st.write("Pick scan options and run the checks.")
    run_once = st.button("🔍 Run Scan Now")
    auto_run = st.checkbox("Run Auto Tracking (blocking loop in this session — not recommended for 24/7)", value=False)
    interval_seconds = st.number_input("Auto interval (seconds)", min_value=30, max_value=3600, value=60, step=30)

    # Display last run
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None
    if st.session_state["last_run"]:
        st.write("Last run:", st.session_state["last_run"])

with col2:
    st.title("📈 Indian Stock Auto Tracker (EMA + RSI Alert Bot)")
    st.write("Automatically checks RSI(14) and 200-day EMA proximity for the symbols in `watchlist.xlsx`.")
    if watchlist_df is not None:
        st.success(f"Loaded {len(watchlist_df)} stocks from watchlist.")
        st.dataframe(watchlist_df.head(50), use_container_width=True)

# Core scanning logic
def run_scan_and_report(df):
    results = []
    alerts = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    for idx, row in df.iterrows():
        symbol = str(row["Symbol"]).strip()
        if not symbol:
            continue
        latest, err = compute_indicators(symbol, period_days=500)
        if err:
            results.append(
                {"Symbol": symbol, "Status": "error", "Error": str(err)}
            )
            continue
        ok, reason = check_conditions(latest, price_tolerance=0.02, rsi_low=30, rsi_high=40)
        row_res = {
            "Symbol": symbol,
            "Date": latest.get("last_date"),
            "Price": latest.get("price"),
            "EMA200": latest.get("ema200"),
            "RSI14": latest.get("rsi14"),
            "Match": ok,
            "Reason": "" if ok else reason,
        }
        results.append(row_res)
        if ok:
            message = (
                f"ALERT: {symbol}\n"
                f"Price: {latest['price']:.2f}\n"
                f"EMA200: {latest['ema200']:.2f}\n"
                f"RSI(14): {latest['rsi14']:.2f}\n"
                f"Condition: Price within ±2% of EMA200 and RSI between 30-40\n"
                f"Time (UTC): {now}"
            )
            alerts.append((symbol, message))
    return pd.DataFrame(results), alerts

def _do_run_once():
    if watchlist_df is None:
        st.warning("No watchlist to scan.")
        return
    st.info("Running scan — this may take a little while depending on number of symbols.")
    df_results, alerts = run_scan_and_report(watchlist_df)
    st.session_state["last_run"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    # show table
    st.subheader("Scan Results")
    st.dataframe(df_results, use_container_width=True)
    # do alerts
    if alerts:
        st.success(f"{len(alerts)} symbols matched criteria. Sending Telegram alerts...")
        sent = 0
        for symbol, msg in alerts:
            ok = send_telegram_message(msg)
            if ok:
                sent += 1
            else:
                st.warning(f"Failed to send alert for {symbol}")
        st.info(f"Alerts sent: {sent}/{len(alerts)}")
    else:
        st.info("No alerts generated this run.")

# Run-once button
if run_once:
    _do_run_once()

# Auto loop (blocking) — run only if user checks it and accepts caveats
if auto_run:
    st.warning("Auto tracking started in this session. This is a blocking loop and will run until you stop the app or uncheck the box.")
    try:
        while True:
            _do_run_once()
            # flush logs, then sleep
            time.sleep(int(interval_seconds))
            # rerun to refresh UI
            st.experimental_rerun()
    except Exception as e:
        st.error(f"Auto-run exception: {e}")

# If not run yet, show placeholder for last results if there are any
if "last_run" in st.session_state and st.session_state["last_run"]:
    st.sidebar.write("Last run:", st.session_state["last_run"])
