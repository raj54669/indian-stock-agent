# app.py — Indian Stock Agent (ready-to-paste)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io, os, time, requests
from datetime import datetime, timedelta, timezone
from typing import Optional

# -----------------------
# Configuration / Constants
# -----------------------
EMA_SPAN = 200
RSI_PERIOD = 14
YF_PERIOD = "2y"
YF_INTERVAL = "1d"
IST = timezone(timedelta(hours=5, minutes=30))

# Optional auto-refresh support (safe import)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Alert Bot", layout="wide")
st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# Initialize session state for alert history and small control flags
if "alert_history" not in st.session_state or not isinstance(st.session_state.get("alert_history"), pd.DataFrame):
    st.session_state["alert_history"] = pd.DataFrame(
        columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
    )
# helper flag for confirm clear
if "confirm_clear" not in st.session_state:
    st.session_state["confirm_clear"] = False
# debug logs container
if "debug_logs" not in st.session_state:
    st.session_state["debug_logs"] = []

# -----------------------
# Secrets helpers
# -----------------------
def get_secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)

def get_secret_section(key: str, section: Optional[str] = None, default=None):
    try:
        if section and section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        pass
    return get_secret(key, default)

GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_REPO = get_secret("GITHUB_REPO")
GITHUB_BRANCH = get_secret("GITHUB_BRANCH", "main")
GITHUB_FILE_PATH = get_secret("GITHUB_FILE_PATH", "watchlist.xlsx")

# Telegram secrets
TELEGRAM_TOKEN = get_secret_section("TELEGRAM_TOKEN", section="telegram")
CHAT_ID = get_secret_section("CHAT_ID", section="telegram")

# -----------------------
# GitHub headers
# -----------------------
def github_raw_headers():
    auth_scheme = "Bearer" if str(GITHUB_TOKEN).startswith("github_pat_") else "token"
    return {
        "Authorization": f"{auth_scheme} {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "streamlit-indian-stock-agent"
    }

# -----------------------
# Load Excel from GitHub
# -----------------------
@st.cache_data(ttl=120)
def load_excel_from_github():
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return pd.DataFrame()
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}?ref={GITHUB_BRANCH}"
        r = requests.get(url, headers=github_raw_headers(), timeout=10)
        if r.status_code == 200:
            df = pd.read_excel(io.BytesIO(r.content))
            return df
        else:
            st.error(f"GitHub fetch failed: {r.status_code} - {r.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading from GitHub: {e}")
        return pd.DataFrame()

# -----------------------
# Sidebar: watchlist / settings / status (single source for status)
# -----------------------
st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

use_uploaded = False
watchlist_df = pd.DataFrame()
if uploaded_file is not None:
    try:
        df_up = pd.read_excel(uploaded_file)
        if "Symbol" not in df_up.columns:
            st.sidebar.error("Uploaded file must contain a 'Symbol' column.")
        else:
            watchlist_df = df_up
            use_uploaded = True
            st.sidebar.success(f"✅ Using uploaded watchlist ({len(watchlist_df)} symbols)")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")

if not use_uploaded:
    watchlist_df = load_excel_from_github()
    st.sidebar.info("Using GitHub watchlist as default source")

st.sidebar.header("Settings")
# Telegram indicator
if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.info("Telegram not configured — alerts disabled")

# GitHub indicator
if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.warning("GitHub credentials missing (set GITHUB_TOKEN and GITHUB_REPO)")

# Auto-scan defaults in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Auto-scan")
auto_default = True  # user asked default ON
auto = st.sidebar.checkbox("Enable Auto-scan (default ON)", value=auto_default, key="sidebar_auto")
interval = st.sidebar.number_input("Auto-scan Interval (sec)", value=60, step=5, min_value=5, key="sidebar_interval")
parallel_downloads = st.sidebar.checkbox("Parallel downloads (faster)", value=False, key="sidebar_parallel")

# Status shown in sidebar only (prevents duplicate)
st.sidebar.markdown("---")
st.sidebar.subheader("Status")
st.sidebar.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
st.sidebar.write(f"- GitHub Token: {'✅' if GITHUB_TOKEN else '❌'}")
st.sidebar.write(f"- Telegram Token: {'✅' if TELEGRAM_TOKEN and CHAT_ID else '❌'}")
try:
    st.sidebar.caption(f"📦 yfinance version: {yf.__version__}")
except Exception:
    pass

# sanitize watchlist: ensure Symbol column exists
if not watchlist_df.empty:
    if "Symbol" in watchlist_df.columns:
        watchlist_df["Symbol"] = watchlist_df["Symbol"].astype(str).str.strip()
        watchlist_df = watchlist_df[watchlist_df["Symbol"] != ""]
    else:
        watchlist_df = pd.DataFrame()

# -----------------------
# Telegram helper (use HTML parse mode for bold symbol)
# -----------------------
def send_telegram(message_html: str):
    """Send Telegram message using HTML parse mode for bold symbol."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": message_html, "parse_mode": "HTML"},
            timeout=10
        )
        return r.status_code == 200
    except Exception as e:
        # keep a debug log
        st.warning(f"Telegram send error: {e}")
        st.session_state["debug_logs"].append(f"Telegram send error: {e}")
        return False

# -----------------------
# Indicators helpers (kept and slightly parameterized)
# -----------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(x) for x in col if x is not None and str(x) != ""]).strip("_")
            for col in df.columns.values
        ]
    return df

def _find_close_column(df: pd.DataFrame):
    cols = [c for c in df.columns]
    for prefer in ("close", "adjclose", "adj_close", "adjusted_close"):
        for c in cols:
            if c.replace(" ", "").replace("_", "").lower() == prefer:
                return c
    for c in cols:
        if "close" in c.lower():
            return c
    return None

def calc_rsi_ema(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        if df is None or df.empty:
            return None
        df = _flatten_columns(df)
        close_col = _find_close_column(df)
        if close_col is None:
            return None
        df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        df["EMA200"] = df["Close"].ewm(span=EMA_SPAN, adjust=False, min_periods=1).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
        last_date = df.index.max()
        cutoff = last_date - timedelta(days=365)
        df_1y = df[df.index >= cutoff]
        if not df_1y.empty:
            h52 = df_1y["Close"].max()
            l52 = df_1y["Close"].min()
        else:
            h52 = df["Close"].max()
            l52 = df["Close"].min()
        df["52W_High"] = h52
        df["52W_Low"] = l52
        return df
    except Exception as e:
        st.session_state["debug_logs"].append(f"calc_rsi_ema error: {e}")
        return None

# -----------------------
# Centralized analyze() (keeps original alert conditions)
# -----------------------
def _try_yf_download(symbol: str):
    """Try a few fallbacks if yfinance fails for a suffix (simple heuristic)."""
    tries = [symbol]
    # common fallbacks: swap .NS <-> .BO or drop suffix
    if symbol.endswith(".NS"):
        base = symbol[:-3]
        tries.append(base + ".BO")
        tries.append(base)
    elif symbol.endswith(".BO"):
        base = symbol[:-3]
        tries.append(base + ".NS")
        tries.append(base)
    else:
        tries.append(symbol + ".NS")
        tries.append(symbol + ".BO")
    # try sequentially
    for s in tries:
        try:
            df = yf.download(s, period=YF_PERIOD, interval=YF_INTERVAL, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                if "Close" in df.columns or any("Close" in c for c in df.columns):
                    return df, s
        except Exception as e:
            st.session_state["debug_logs"].append(f"yf.download({s}) error: {e}")
            continue
    return None, None

def analyze(symbol: str):
    try:
        df, used_symbol = _try_yf_download(symbol)
        if df is None or df.empty:
            st.session_state["debug_logs"].append(f"No data for {symbol} (tried alternates).")
            return None
        df_ind = calc_rsi_ema(df)
        if df_ind is None or df_ind.empty:
            st.session_state["debug_logs"].append(f"calc_rsi_ema failed for {symbol}.")
            return None
        last = df_ind.iloc[-1]
        cmp_ = float(last["Close"]) if not pd.isna(last["Close"]) else None
        ema200 = float(last.get("EMA200", np.nan)) if not pd.isna(last.get("EMA200", np.nan)) else None
        rsi14 = float(last.get("RSI14", np.nan)) if not pd.isna(last.get("RSI14", np.nan)) else None
        low52 = float(last.get("52W_Low", np.nan)) if not pd.isna(last.get("52W_Low", np.nan)) else None
        high52 = float(last.get("52W_High", np.nan)) if not pd.isna(last.get("52W_High", np.nan)) else None
        if cmp_ is None or ema200 is None or rsi14 is None:
            st.session_state["debug_logs"].append(f"Incomplete indicators for {symbol}.")
            return None

        # Signal conditions (UNCHANGED logic)
        signal = "Neutral"
        alert_condition = ""
        # WATCH
        if (cmp_ * 0.98 <= ema200 <= cmp_ * 1.02) and (30 <= rsi14 <= 40):
            signal = "🟡 WATCH"
            alert_condition = "EMA200 within ±2% of CMP & RSI between 30–40"
        # BUY
        elif rsi14 < 30:
            signal = "🟢 BUY"
            alert_condition = "RSI < 30 (Oversold)"
        # SELL
        elif rsi14 > 70:
            signal = "🔴 SELL"
            alert_condition = "RSI > 70 (Overbought)"

        # Trend confirmation (simple CMP vs EMA200)
        trend = "Above EMA200 (bullish)" if cmp_ > ema200 else "Below EMA200 (bearish)"

        # Format Telegram (HTML) with bold symbol and trend
        emoji_for_msg = signal.split()[0] if signal != "Neutral" else "⚡"
        # signal name (text after emoji) if present
        signal_name = " ".join(signal.split()[1:]) if signal != "Neutral" else "Neutral"
        telegram_html = (
            f"{emoji_for_msg} <b>Alert: {signal}</b>\n"
            f"<b>{symbol}</b>\n"
            f"CMP = {cmp_:.2f}\n"
            f"EMA200 = {ema200:.2f}\n"
            f"RSI14 = {rsi14:.2f}\n"
            f"Trend: {trend}\n"
            f"Condition: {alert_condition if alert_condition else 'No active signal'}"
        )

        return {
            "Symbol": symbol,
            "CMP": round(cmp_, 2),
            "52W_Low": round(low52, 2) if not pd.isna(low52) else None,
            "52W_High": round(high52, 2) if not pd.isna(high52) else None,
            "EMA200": round(ema200, 2),
            "RSI14": round(rsi14, 2),
            "Signal": signal,
            "AlertCondition": alert_condition,
            "TelegramMessage": telegram_html
        }
    except Exception as e:
        st.session_state["debug_logs"].append(f"analyze({symbol}) error: {e}")
        return None

# -----------------------
# Main page UI
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

# Symbols list from watchlist
if watchlist_df.empty or "Symbol" not in watchlist_df.columns:
    st.warning("⚠️ No valid 'Symbol' column found in your watchlist Excel file.")
    symbols = []
else:
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()

st.subheader("📋 Combined Summary Table (Live Data)")

# placeholder + last scan caption
summary_placeholder = st.empty()
last_scan_time = st.empty()

# debug logs expander
with st.expander("🔍 Debug Logs", expanded=False):
    debug_text = "\n".join(st.session_state.get("debug_logs", [])) or "No debug logs yet."
    st.code(debug_text, language="")

# Controls (main page)
col_left, col_right = st.columns([1, 2])
with col_left:
    run_now = st.button("Run Scan Now")
    refresh_watchlist = st.button("Refresh Watchlist from GitHub")
    # Clear history initiator (shows confirm button)
    if st.button("🧹 Clear History"):
        st.session_state["confirm_clear"] = True
    if st.session_state.get("confirm_clear", False):
        if st.button("Confirm Clear History", key="confirm_clear_btn"):
            st.session_state["alert_history"] = pd.DataFrame(
                columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
            )
            st.session_state["confirm_clear"] = False
            st.success("✅ Alert history cleared.")
            # update debug log
            st.session_state["debug_logs"].append("Alert history cleared by user.")
    # export CSV
    if st.button("⬇️ Export Alert History as CSV"):
        csv = st.session_state["alert_history"].to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="alert_history.csv", mime="text/csv")

with col_right:
    st.markdown("**Status:**")
    # show a minimal status on main page (not duplicating all sidebar info)
    st.write(f"- Auto-scan: {'ON' if st.sidebar.session_state.get('sidebar_auto', False) else 'OFF'}")
    last_scan_display = st.session_state.get("last_scan_time")
    if last_scan_display:
        st.write(f"- Last scan: {last_scan_display} (Live)")

# -----------------------
# Run scan function with progress bar + logs
# -----------------------
def run_scan(symbols_list):
    results = []
    total = len(symbols_list)
    if total == 0:
        st.warning("No symbols to scan.")
        return results
    progress = st.progress(0)
    for i, sym in enumerate(symbols_list, start=1):
        st.session_state["debug_logs"].append(f"Starting analysis for {sym}")
        res = analyze(sym)
        if res:
            results.append(res)
            # send telegram only if alert condition non-neutral
            if res["Signal"] != "Neutral" and TELEGRAM_TOKEN and CHAT_ID:
                # Telegram HTML message is inside res["TelegramMessage"]
                send_telegram(res["TelegramMessage"])
                st.session_state["debug_logs"].append(f"Telegram sent for {sym}")
            else:
                st.session_state["debug_logs"].append(f"No alert for {sym} (Signal={res['Signal']})")
        else:
            st.session_state["debug_logs"].append(f"No result for {sym}")
        # update progress
        progress.progress(int(i/total * 100))
    progress.empty()
    # update last scan time
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")
    st.session_state["last_scan_time"] = now
    st.session_state["debug_logs"].append(f"Scan completed at {now}")
    return results

# handle refresh watchlist button
if refresh_watchlist:
    watchlist_df = load_excel_from_github()
    st.success("Watchlist refreshed from GitHub.")
    # update symbols
    if not watchlist_df.empty and "Symbol" in watchlist_df.columns:
        symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()

# perform scan if user pressed Run Scan Now or auto sidebar option triggered
should_auto_run = st.sidebar.session_state.get("sidebar_auto", False)
auto_interval = st.sidebar.session_state.get("sidebar_interval", 60)
if run_now or should_auto_run:
    # run scan
    results = run_scan(symbols)
    if results:
        # build DataFrame for main table (keep columns similar to original)
        df_out = pd.DataFrame(results)[["Symbol", "CMP", "52W_Low", "52W_High", "EMA200", "RSI14", "Signal"]]
        # display with use_container_width (no index)
        summary_placeholder.dataframe(df_out, use_container_width=True, hide_index=True)
    else:
        summary_placeholder.info("No valid data fetched.")
    # update last scan caption
    last_scan_time.caption(f"Last scan: {st.session_state.get('last_scan_time')} (Live)")

# Auto-refresh support via package
if st.sidebar.session_state.get("sidebar_auto", False):
    if st_autorefresh:
        st_autorefresh(interval=int(st.sidebar.session_state.get("sidebar_interval", 60)) * 1000, key="auto_refresh")
    else:
        st.warning("Auto-refresh package missing. Run: pip install streamlit-autorefresh")

# -----------------------
# Alert History display (preserve original columns)
# -----------------------
st.markdown("---")
st.subheader("📜 Alert History")
if st.session_state["alert_history"].empty:
    st.info("No alerts recorded yet. Run a scan to generate new alerts.")
else:
    st.dataframe(st.session_state["alert_history"], use_container_width=True, hide_index=True)

# show debug logs updated content (so they refresh)
with st.expander("🔍 Debug Logs (latest)", expanded=False):
    debug_text = "\n".join(st.session_state.get("debug_logs", [])) or "No debug logs yet."
    st.code(debug_text, language="")

# small footer controls
st.markdown("---")
if st.button("📨 Send Test Telegram Alert"):
    if TELEGRAM_TOKEN and CHAT_ID:
        ok = send_telegram("<b>✅ Test alert from Indian Stock Agent Bot!</b>")
        if ok:
            st.success("✅ Telegram test alert sent successfully!")
        else:
            st.error("❌ Telegram send failed. Check token/chat_id and logs.")
    else:
        st.warning("Telegram not configured. Set TELEGRAM_TOKEN and CHAT_ID in secrets or env.")

# end of file
