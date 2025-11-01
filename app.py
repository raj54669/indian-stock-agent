# app.py — Indian Stock Agent (Optimized, drop-in replacement)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io, os, requests, base64
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------
# CONFIG / CONSTANTS
# -----------------------
EMA_SPAN = 200
RSI_PERIOD = 14
ALERT_THROTTLE_MIN = 60            # minutes to throttle repeated alerts per symbol
CACHE_TTL = 300                    # seconds for cached downloads
MAX_WORKERS = 8                    # parallel threads for downloads
YF_PERIOD = "2y"
YF_INTERVAL = "1d"

IST = timezone(timedelta(hours=5, minutes=30))

# -----------------------
# STREAMLIT & UI SETUP
# -----------------------
st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Alert Bot (Optimized)", layout="wide")
st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# -----------------------
# Secrets helpers
# -----------------------
def get_secret(key: str, default=None):
    try:
        # st.secrets behaves like a mapping
        return st.secrets.get(key, os.getenv(key, default))
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
GITHUB_BRANCH = get_secret("GITHUB_BRANCH") or "main"
GITHUB_FILE_PATH = get_secret("GITHUB_FILE_PATH") or "watchlist.xlsx"

TELEGRAM_TOKEN = get_secret_section("TELEGRAM_TOKEN", section="telegram") or get_secret("TELEGRAM_TOKEN")
CHAT_ID = get_secret_section("CHAT_ID", section="telegram") or get_secret("CHAT_ID")

# -----------------------
# GitHub helpers (robust)
# -----------------------
def github_headers(token: Optional[str]) -> dict:
    if not token:
        return {}
    auth_scheme = "Bearer" if str(token).startswith("github_pat_") else "token"
    return {
        "Authorization": f"{auth_scheme} {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "streamlit-indian-stock-agent"
    }

@st.cache_data(ttl=CACHE_TTL)
def download_watchlist_from_github() -> pd.DataFrame:
    """
    Downloads Excel watchlist from GitHub repo configured in secrets.
    Returns DataFrame or empty DataFrame on failure.
    """
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return pd.DataFrame()
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}?ref={GITHUB_BRANCH}"
        r = requests.get(url, headers=github_headers(GITHUB_TOKEN), timeout=15)
        r.raise_for_status()
        data = r.json()
        content_b64 = data.get("content", "")
        if not content_b64:
            return pd.DataFrame()
        decoded = base64.b64decode(content_b64)
        return pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        # don't crash app — caller will handle empty DF
        st.session_state.setdefault("debug_logs", []).append(f"GitHub download failed: {e}")
        return pd.DataFrame()

def upload_watchlist_to_github(uploaded_file) -> bool:
    """
    Replace file at GITHUB_FILE_PATH with uploaded_file (Streamlit UploadedFile).
    Returns True on success.
    """
    if not (GITHUB_TOKEN and GITHUB_REPO):
        st.error("Missing GitHub credentials (GITHUB_TOKEN / GITHUB_REPO).")
        return False
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}?ref={GITHUB_BRANCH}"
        headers = github_headers(GITHUB_TOKEN)
        sha = None
        try:
            rr = requests.get(get_url, headers=headers, timeout=10)
            if rr.ok:
                sha = rr.json().get("sha")
        except Exception:
            sha = None

        content_b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        payload = {"message": "Updated watchlist via Streamlit", "content": content_b64, "branch": GITHUB_BRANCH}
        if sha:
            payload["sha"] = sha
        put_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}"
        res = requests.put(put_url, headers=headers, json=payload, timeout=20)
        res.raise_for_status()
        return True
    except Exception as e:
        st.error(f"GitHub upload failed: {e}")
        return False

# -----------------------
# Session-state initialization
# -----------------------
def init_session_state():
    if "alert_history" not in st.session_state or not isinstance(st.session_state["alert_history"], pd.DataFrame):
        st.session_state["alert_history"] = pd.DataFrame(columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"])
    if "last_alert_ts" not in st.session_state:
        # stores symbol -> ISO timestamp of last alert sent
        st.session_state["last_alert_ts"] = {}
    if "debug_logs" not in st.session_state:
        st.session_state["debug_logs"] = []
    if "auto_scan_enabled" not in st.session_state:
        st.session_state["auto_scan_enabled"] = True
    if "scan_interval" not in st.session_state:
        st.session_state["scan_interval"] = 60
    if "telegram_enabled" not in st.session_state:
        st.session_state["telegram_enabled"] = True

init_session_state()

# -----------------------
# Utilities: indicators (centralized)
# -----------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join([str(x) for x in col if x is not None and str(x) != ""]).strip("_") for col in df.columns.values]
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
        df = df.copy()
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
# Caching symbol downloads
# -----------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_symbol_history(symbol: str, period: str = YF_PERIOD, interval: str = YF_INTERVAL) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        return df
    except Exception as e:
        st.session_state["debug_logs"].append(f"yf.download failed for {symbol}: {e}")
        return pd.DataFrame()

# -----------------------
# Centralized analyze function used by UI & run_scan
# -----------------------
def analyze_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch symbol history (cached), compute indicators, return last-row summary dict or None.
    """
    df = get_symbol_history(symbol)
    if df is None or df.empty:
        return None
    df_ind = calc_rsi_ema(df)
    if df_ind is None or df_ind.empty:
        return None
    last = df_ind.iloc[-1]
    try:
        cmp_ = float(last["Close"])
        ema200 = float(last["EMA200"])
        rsi14 = float(last["RSI14"])
        low52 = float(last["52W_Low"])
        high52 = float(last["52W_High"])
    except Exception as e:
        st.session_state["debug_logs"].append(f"analyze_symbol parsing error for {symbol}: {e}")
        return None

    signal = "Neutral"
    alert_condition = ""
    if (cmp_ * 0.98 <= ema200 <= cmp_ * 1.02) and (30 <= rsi14 <= 40):
        signal = "🟡 WATCH"
        alert_condition = "EMA200 within ±2% of CMP & RSI between 30–40"
    elif rsi14 < 30:
        signal = "🟢 BUY"
        alert_condition = "RSI < 30 (Oversold)"
    elif rsi14 > 70:
        signal = "🔴 SELL"
        alert_condition = "RSI > 70 (Overbought)"

    return {
        "Symbol": symbol,
        "CMP": round(cmp_, 2),
        "52W_Low": round(low52, 2) if low52 is not None else None,
        "52W_High": round(high52, 2) if high52 is not None else None,
        "EMA200": round(ema200, 2),
        "RSI14": round(rsi14, 2),
        "Signal": signal,
        "AlertCondition": alert_condition
    }

# -----------------------
# Alert history helpers
# -----------------------
def add_alert_history(symbol: str, signal: str, cmp_: float, ema200: float, rsi14: float):
    ensure_alert_history()
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([{
        "Date & Time (IST)": ts,
        "Symbol": symbol,
        "Signal": signal,
        "CMP": round(float(cmp_), 2),
        "EMA200": round(float(ema200), 2),
        "RSI14": round(float(rsi14), 2)
    }])
    st.session_state["alert_history"] = pd.concat([st.session_state["alert_history"], new_row], ignore_index=True)

def ensure_alert_history():
    if "alert_history" not in st.session_state or not isinstance(st.session_state["alert_history"], pd.DataFrame):
        st.session_state["alert_history"] = pd.DataFrame(columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"])

# -----------------------
# Telegram helper with throttling
# -----------------------
def send_telegram_msg(message: str, symbol: Optional[str] = None) -> bool:
    """
    Sends telegram if configured & enabled.
    Throttling: prevents repeated alerts for same symbol within ALERT_THROTTLE_MIN.
    """
    if not st.session_state.get("telegram_enabled", True):
        st.session_state["debug_logs"].append("Telegram disabled via toggle.")
        return False

    token = TELEGRAM_TOKEN or get_secret_section("TELEGRAM_TOKEN", section="telegram")
    chat_id = CHAT_ID or get_secret_section("CHAT_ID", section="telegram")
    if not token or not chat_id:
        st.session_state["debug_logs"].append("Telegram secrets missing.")
        return False

    # throttle per symbol
    if symbol:
        last_ts_str = st.session_state["last_alert_ts"].get(symbol)
        if last_ts_str:
            try:
                last_ts = datetime.fromisoformat(last_ts_str)
                if (datetime.now() - last_ts).total_seconds() < ALERT_THROTTLE_MIN * 60:
                    st.session_state["debug_logs"].append(f"Throttled telegram for {symbol}.")
                    return False
            except Exception:
                pass

    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": message},
                          timeout=10)
        ok = r.status_code == 200
        if ok and symbol:
            st.session_state["last_alert_ts"][symbol] = datetime.now().isoformat()
        return ok
    except Exception as e:
        st.session_state["debug_logs"].append(f"send_telegram_msg failed: {e}")
        return False

# -----------------------
# Download & parallel analyze orchestration
# -----------------------
def run_scan_for_symbols(symbol_list: List[str], parallel: bool = True) -> pd.DataFrame:
    """
    Analyze provided symbols and return a results DataFrame.
    If telegram is enabled, send alerts (subject to throttling).
    """
    results = []
    logs = st.session_state.get("debug_logs", [])

    if parallel and symbol_list:
        # Use ThreadPoolExecutor to analyze symbols in parallel
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(symbol_list))) as ex:
            future_to_symbol = {ex.submit(analyze_symbol, s): s for s in symbol_list}
            for fut in as_completed(future_to_symbol):
                sym = future_to_symbol[fut]
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                        # Send alert if triggered
                        if res["Signal"] in ("🟢 BUY", "🔴 SELL", "🟡 WATCH"):
                            add_alert_history(sym, res["Signal"], res["CMP"], res["EMA200"], res["RSI14"])
                            msg = f"⚡ Alert: {sym}\\nCMP = {res['CMP']:.2f}\\nEMA200 = {res['EMA200']:.2f}\\nRSI14 = {res['RSI14']:.2f}\\nCondition: {res['AlertCondition']}"
                            send_telegram_msg(msg, symbol=sym)
                except Exception as e:
                    logs.append(f"Parallel analyze failed for {sym}: {e}")
    else:
        # sequential
        for s in symbol_list:
            try:
                res = analyze_symbol(s)
                if res:
                    results.append(res)
                    if res["Signal"] in ("🟢 BUY", "🔴 SELL", "🟡 WATCH"):
                        add_alert_history(s, res["Signal"], res["CMP"], res["EMA200"], res["RSI14"])
                        msg = f"⚡ Alert: {s}\\nCMP = {res['CMP']:.2f}\\nEMA200 = {res['EMA200']:.2f}\\nRSI14 = {res['RSI14']:.2f}\\nCondition: {res['AlertCondition']}"
                        send_telegram_msg(msg, symbol=s)
            except Exception as e:
                logs.append(f"Sequential analyze failed for {s}: {e}")

    # keep debug logs in session
    st.session_state["debug_logs"] = logs
    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()

# -----------------------
# UI: Sidebar (watchlist upload, settings)
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
            # attempt to upload to GitHub (best-effort)
            try:
                upload_watchlist_to_github(uploaded_file)
            except Exception as e:
                st.sidebar.info(f"GitHub push failed (non-blocking): {e}")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")

if not use_uploaded:
    watchlist_df = download_watchlist_from_github()
    st.sidebar.info("Using GitHub watchlist as default source")

st.sidebar.header("Settings")
st.sidebar.markdown("**Telegram**")
st.session_state["telegram_enabled"] = st.sidebar.checkbox("Enable Telegram Alerts", value=st.session_state["telegram_enabled"])
st.sidebar.markdown("**Auto-scan**")
st.session_state["auto_scan_enabled"] = st.sidebar.checkbox("Enable Auto-scan", value=st.session_state["auto_scan_enabled"])
st.session_state["scan_interval"] = st.sidebar.number_input("Auto-scan Interval (sec)", min_value=5, value=int(st.session_state["scan_interval"]), step=5)

st.sidebar.markdown("**Advanced**")
parallel_fetch = st.sidebar.checkbox("Parallel downloads (faster)", value=True)
st.sidebar.caption("Streams: yfinance caching + optional parallelism")

# status badges
st.sidebar.markdown("---")
st.sidebar.markdown("**Status**")
st.sidebar.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
st.sidebar.write(f"- GitHub Token: {'✅' if GITHUB_TOKEN else '❌'}")
st.sidebar.write(f"- Telegram Token: {'✅' if TELEGRAM_TOKEN else '❌'}")

# sanitize watchlist
if not watchlist_df.empty and "Symbol" in watchlist_df.columns:
    watchlist_df["Symbol"] = watchlist_df["Symbol"].astype(str).str.strip()
    watchlist_df = watchlist_df[watchlist_df["Symbol"] != ""]
else:
    watchlist_df = pd.DataFrame()

# -----------------------
# Main layout
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot (Optimized)")

if watchlist_df.empty or "Symbol" not in watchlist_df.columns:
    st.warning("⚠️ No valid 'Symbol' column found in your watchlist Excel file.")
    symbols = []
else:
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()

# Summary placeholder
st.subheader("📋 Combined Summary Table")
initial_df = pd.DataFrame({
    "Symbol": symbols if symbols else [],
    "CMP": ["" for _ in symbols] if symbols else [],
    "52W_Low": ["" for _ in symbols] if symbols else [],
    "52W_High": ["" for _ in symbols] if symbols else [],
    "EMA200": ["" for _ in symbols] if symbols else [],
    "RSI14": ["" for _ in symbols] if symbols else [],
    "Signal": ["" for _ in symbols] if symbols else [],
})
summary_placeholder = st.empty()
summary_placeholder.dataframe(initial_df, use_container_width=True, hide_index=True)
last_scan_time = st.caption("Will auto-update after scanning.")

# Controls
st.subheader("⚙️ Controls")
col1, col2 = st.columns([1, 2])
with col1:
    run_now = st.button("Run Scan Now")
    interval = st.number_input("Interval (sec)", value=int(st.session_state["scan_interval"]), step=5, min_value=5)
    st.session_state["scan_interval"] = int(interval)
    auto = st.checkbox("Enable Auto-scan", value=st.session_state["auto_scan_enabled"])
    st.session_state["auto_scan_enabled"] = bool(auto)
    st.write("")  # spacer
    if st.button("🔄 Refresh Watchlist from GitHub"):
        # Clear cached github call and reload
        download_watchlist_from_github.clear()
        watchlist_df = download_watchlist_from_github()
        st.experimental_rerun()

with col2:
    st.markdown("**Status:**")
    st.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    if auto:
        st.markdown(f"<span style='margin-left:10px;'>🔁 Auto-scan active — every {interval} seconds</span>", unsafe_allow_html=True)

# Search / filter for history
st.subheader("🔎 Alert History & Controls")
filter_col, export_col = st.columns([3,1])
with filter_col:
    search_q = st.text_input("Search alert history (symbol or signal)", value="")
with export_col:
    if not st.session_state["alert_history"].empty:
        csv_bytes = st.session_state["alert_history"].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export CSV", csv_bytes, file_name="alert_history.csv", mime="text/csv")
    else:
        st.write("")  # keep layout

# Run scan: show spinner and debug logs
if run_now:
    with st.spinner("Scanning symbols..."):
        res_df = run_scan_for_symbols(symbols, parallel=parallel_fetch)
        if not res_df.empty:
            # style table rows by Signal
            def color_row(row):
                sig = row["Signal"]
                if isinstance(sig, str):
                    if "BUY" in sig:
                        return ["background-color: #e6ffed"] * len(row)
                    if "SELL" in sig:
                        return ["background-color: #ffecec"] * len(row)
                    if "WATCH" in sig:
                        return ["background-color: #fff7e6"] * len(row)
                return [""] * len(row)
            styled = res_df.style.apply(color_row, axis=1)
            summary_placeholder.dataframe(styled, use_container_width=True)
        else:
            summary_placeholder.warning("No valid data fetched.")
        last_scan_time.caption(f"Last scan: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Auto-scan via autorefresh (non-blocking)
if st.session_state["auto_scan_enabled"]:
    if st_autorefresh:
        # keep safe minimal interval
        try:
            st_autorefresh(interval=int(st.session_state["scan_interval"]) * 1000, key="auto_refresh")
            # run scan on refresh
            res_df = run_scan_for_symbols(symbols, parallel=parallel_fetch)
            if not res_df.empty:
                styled = res_df.style.apply(lambda r: ["background-color: #e6ffed" if "BUY" in r["Signal"] else ("background-color: #ffecec" if "SELL" in r["Signal"] else ("background-color: #fff7e6" if "WATCH" in r["Signal"] else ""))], axis=1)
                summary_placeholder.dataframe(styled, use_container_width=True)
            last_scan_time.caption(f"Last scan: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        except Exception:
            st.warning("Auto-refresh package missing or failed. Install streamlit-autorefresh for auto-scan.")
    else:
        st.warning("Auto-refresh package missing. Run: pip install streamlit-autorefresh")

# Show Alert History with filtering
ensure_alert_history()
if st.session_state["alert_history"].empty:
    st.info("No alerts recorded yet. Run a scan to generate new alerts.")
else:
    df_hist = st.session_state["alert_history"]
    if search_q:
        mask = df_hist.apply(lambda row: row.astype(str).str.contains(search_q, case=False).any(), axis=1)
        df_show = df_hist[mask]
    else:
        df_show = df_hist
    st.dataframe(df_show, use_container_width=True, hide_index=True)
    if st.button("🧹 Clear History"):
        # require confirmation via checkbox to avoid accidental deletion
        if st.checkbox("Confirm clear history"):
            st.session_state["alert_history"] = pd.DataFrame(columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"])
            st.success("✅ Alert history cleared.")
            st.experimental_rerun()

# Debug logs expander
with st.expander("🔍 Debug Logs"):
    logs = st.session_state.get("debug_logs", [])
    if not logs:
        st.write("No debug logs.")
    else:
        for l in logs[-200:]:
            st.text(l)

# Test Telegram button
st.markdown("---")
if st.button("📨 Send Test Telegram Alert"):
    ok = send_telegram_msg("✅ Test alert from Indian Stock Agent Bot!", symbol=None)
    if ok:
        st.success("✅ Telegram test alert sent successfully!")
    else:
        st.error("❌ Telegram send failed. Check your token or chat_id and throttling settings.")
