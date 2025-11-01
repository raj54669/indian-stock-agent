# app.py — Final single-file (preserves original logic + fixes)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io, os, requests, base64
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------
# Config / constants
# -----------------------
EMA_SPAN = 200
RSI_PERIOD = 14
ALERT_THROTTLE_MIN = 60  # minutes to throttle repeated alerts per symbol
CACHE_TTL = 300  # seconds for cached downloads
MAX_WORKERS = 8
YF_PERIOD = "2y"
YF_INTERVAL = "1d"
IST = timezone(timedelta(hours=5, minutes=30))

# -----------------------
# Optional autorefresh import
# -----------------------
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None

# -----------------------
# Streamlit setup
# -----------------------
st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Alert Bot", layout="wide")
st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# -----------------------
# Secrets helpers
# -----------------------
def get_secret(key: str, default=None):
    try:
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
# GitHub helpers (watchlist load / upload)
# -----------------------
def _github_headers(token: Optional[str]):
    if not token:
        return {}
    auth_scheme = "Bearer" if str(token).startswith("github_pat_") else "token"
    return {
        "Authorization": f"{auth_scheme} {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "streamlit-indian-stock-agent"
    }

@st.cache_data(ttl=CACHE_TTL)
def load_excel_from_github() -> pd.DataFrame:
    """Load watchlist file from GitHub repo (returns empty df on failure)."""
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return pd.DataFrame()
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}?ref={GITHUB_BRANCH}"
        r = requests.get(url, headers=_github_headers(GITHUB_TOKEN), timeout=15)
        r.raise_for_status()
        data = r.json()
        content = data.get("content") or ""
        if not content:
            return pd.DataFrame()
        decoded = base64.b64decode(content)
        return pd.read_excel(io.BytesIO(decoded))
    except Exception:
        return pd.DataFrame()

def upload_watchlist_to_github(uploaded_bytes: bytes) -> bool:
    """Upload/replace watchlist file on GitHub. Best-effort; shows error to user."""
    if not (GITHUB_TOKEN and GITHUB_REPO):
        st.error("Missing GitHub credentials (GITHUB_TOKEN / GITHUB_REPO).")
        return False
    try:
        owner, repo = GITHUB_REPO.split("/", 1)
        get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{GITHUB_FILE_PATH}?ref={GITHUB_BRANCH}"
        headers = _github_headers(GITHUB_TOKEN)
        sha = None
        try:
            r1 = requests.get(get_url, headers=headers, timeout=10)
            if r1.ok:
                sha = r1.json().get("sha")
        except Exception:
            sha = None
        content_b64 = base64.b64encode(uploaded_bytes).decode("utf-8")
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
# Session state init
# -----------------------
def init_session_state():
    if "alert_history" not in st.session_state or not isinstance(st.session_state["alert_history"], pd.DataFrame):
        st.session_state["alert_history"] = pd.DataFrame(columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"])
    if "last_alert_ts" not in st.session_state:
        st.session_state["last_alert_ts"] = {}
    if "debug_logs" not in st.session_state:
        st.session_state["debug_logs"] = []
    if "scan_interval" not in st.session_state:
        st.session_state["scan_interval"] = 60
    if "auto_scan_enabled" not in st.session_state:
        st.session_state["auto_scan_enabled"] = True
    if "telegram_enabled" not in st.session_state:
        st.session_state["telegram_enabled"] = True
    if "parallel_fetch" not in st.session_state:
        st.session_state["parallel_fetch"] = True

init_session_state()

# -----------------------
# Indicator helpers (original logic preserved)
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
    """Return df enriched with EMA200, RSI14 and 52w high/low (or None)."""
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
# Per-symbol data caching (keyed by symbol)
# -----------------------
@st.cache_data(ttl=CACHE_TTL)
def get_symbol_history(symbol: str, period: str = YF_PERIOD, interval: str = YF_INTERVAL) -> pd.DataFrame:
    """Return history DataFrame for a symbol. Cached per-symbol by Streamlit."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None:
            return pd.DataFrame()
        # return a copy to avoid accidental shared-memory issues
        return df.copy()
    except Exception as e:
        st.session_state["debug_logs"].append(f"yf.download error {symbol}: {e}")
        return pd.DataFrame()

# -----------------------
# Central analyze function (original signals preserved)
# -----------------------
def analyze(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        df = get_symbol_history(symbol)
        if df is None or df.empty:
            return None
        df_ind = calc_rsi_ema(df)
        if df_ind is None or df_ind.empty:
            return None
        # use a copy of last row to ensure isolation
        last = df_ind.iloc[-1].copy()
        cmp_ = float(last["Close"])
        ema200 = float(last["EMA200"])
        rsi14 = float(last["RSI14"])
        low52 = float(last["52W_Low"])
        high52 = float(last["52W_High"])
        signal = "Neutral"
        alert_condition = ""
        # WATCH: EMA200 within ±2% & RSI 30–40
        if (cmp_ * 0.98 <= ema200 <= cmp_ * 1.02) and (30 <= rsi14 <= 40):
            signal = "🟡 WATCH"
            alert_condition = "EMA200 within ±2% of CMP & RSI between 30–40"
        elif rsi14 < 30:
            signal = "🟢 BUY"
            alert_condition = "RSI < 30 (Oversold)"
        elif rsi14 > 70:
            signal = "🔴 SELL"
            alert_condition = "RSI > 70 (Overbought)"
        telegram_msg = (
            f"⚡ Alert: {symbol}\n"
            f"CMP = {cmp_:.2f}\n"
            f"EMA200 = {ema200:.2f}\n"
            f"RSI14 = {rsi14:.2f}\n"
            f"Condition: {alert_condition if alert_condition else 'No active signal'}"
        )
        return {
            "Symbol": symbol,
            "CMP": round(cmp_, 2),
            "52W_Low": round(low52, 2),
            "52W_High": round(high52, 2),
            "EMA200": round(ema200, 2),
            "RSI14": round(rsi14, 2),
            "Signal": signal,
            "AlertCondition": alert_condition,
            "TelegramMessage": telegram_msg
        }
    except Exception as e:
        st.session_state["debug_logs"].append(f"analyze error {symbol}: {e}")
        return None

# -----------------------
# Alert history append
# -----------------------
def add_to_alert_history(symbol: str, signal: str, cmp_: float, ema200: float, rsi14: float):
    if "alert_history" not in st.session_state or not isinstance(st.session_state["alert_history"], pd.DataFrame):
        st.session_state["alert_history"] = pd.DataFrame(columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"])
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

# -----------------------
# Telegram sending (throttled)
# -----------------------
def send_telegram(message: str, symbol: Optional[str] = None) -> bool:
    if not st.session_state.get("telegram_enabled", True):
        st.session_state["debug_logs"].append("Telegram disabled.")
        return False
    token = TELEGRAM_TOKEN or get_secret_section("TELEGRAM_TOKEN", section="telegram")
    chat_id = CHAT_ID or get_secret_section("CHAT_ID", section="telegram")
    if not token or not chat_id:
        st.session_state["debug_logs"].append("Telegram secrets missing.")
        return False
    if symbol:
        last_iso = st.session_state["last_alert_ts"].get(symbol)
        if last_iso:
            try:
                last_dt = datetime.fromisoformat(last_iso)
                if (datetime.now() - last_dt).total_seconds() < ALERT_THROTTLE_MIN * 60:
                    st.session_state["debug_logs"].append(f"Throttled alert for {symbol}")
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
        st.session_state["debug_logs"].append(f"Telegram send error: {e}")
        return False

# -----------------------
# Run scan (parallel optional)
# -----------------------
def run_scan(symbols: List[str], parallel: bool = True) -> pd.DataFrame:
    results = []
    logs = st.session_state.get("debug_logs", [])
    if not symbols:
        return pd.DataFrame()
    try:
        if parallel and len(symbols) > 1:
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(symbols))) as ex:
                future_map = {ex.submit(analyze, s): s for s in symbols}
                for fut in as_completed(future_map):
                    s = future_map[fut]
                    try:
                        res = fut.result()
                        if res:
                            results.append(res)
                            if res["Signal"] in ("🟢 BUY", "🔴 SELL", "🟡 WATCH"):
                                add_to_alert_history(s, res["Signal"], res["CMP"], res["EMA200"], res["RSI14"])
                                send_telegram(res["TelegramMessage"], symbol=s)
                    except Exception as e:
                        logs.append(f"Error analyzing {s}: {e}")
        else:
            for s in symbols:
                try:
                    res = analyze(s)
                    if res:
                        results.append(res)
                        if res["Signal"] in ("🟢 BUY", "🔴 SELL", "🟡 WATCH"):
                            add_to_alert_history(s, res["Signal"], res["CMP"], res["EMA200"], res["RSI14"])
                            send_telegram(res["TelegramMessage"], symbol=s)
                except Exception as e:
                    logs.append(f"Error analyzing {s}: {e}")
    except Exception as e:
        logs.append(f"run_scan fatal error: {e}")
    st.session_state["debug_logs"] = logs
    return pd.DataFrame(results) if results else pd.DataFrame()

# -----------------------
# Sidebar: Watchlist + Settings
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
            watchlist_df = df_up.copy()
            use_uploaded = True
            st.sidebar.success(f"✅ Using uploaded watchlist ({len(watchlist_df)} symbols)")
            # attempt GitHub upload (best-effort)
            try:
                upload_watchlist_to_github(uploaded_file.getvalue())
                load_excel_from_github.clear()
            except Exception:
                pass
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")

if not use_uploaded:
    watchlist_df = load_excel_from_github()
    st.sidebar.info("Using GitHub watchlist as default source")

st.sidebar.header("Settings")
st.sidebar.subheader("Telegram")
st.session_state["telegram_enabled"] = st.sidebar.checkbox("Enable Telegram Alerts", value=st.session_state["telegram_enabled"])
st.sidebar.subheader("Auto-scan")
st.session_state["auto_scan_enabled"] = st.sidebar.checkbox("Enable Auto-scan", value=st.session_state["auto_scan_enabled"])
st.session_state["scan_interval"] = st.sidebar.number_input("Auto-scan Interval (sec)", min_value=5, value=int(st.session_state["scan_interval"]), step=5)
st.sidebar.subheader("Advanced")
st.session_state["parallel_fetch"] = st.sidebar.checkbox("Parallel downloads (faster)", value=st.session_state["parallel_fetch"])

# Sidebar status
st.sidebar.markdown("---")
st.sidebar.markdown("**Status**")
st.sidebar.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
st.sidebar.write(f"- GitHub Token: {'✅' if GITHUB_TOKEN else '❌'}")
st.sidebar.write(f"- Telegram Token: {'✅' if TELEGRAM_TOKEN else '❌'}")
try:
    st.sidebar.caption(f"📦 yfinance version: {yf.__version__}")
except Exception:
    pass

# sanitize watchlist
if not watchlist_df.empty and "Symbol" in watchlist_df.columns:
    watchlist_df["Symbol"] = watchlist_df["Symbol"].astype(str).str.strip()
    watchlist_df = watchlist_df[watchlist_df["Symbol"] != ""]
else:
    watchlist_df = pd.DataFrame()

# -----------------------
# Main UI: Title + Summary Table placeholder
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")
if watchlist_df.empty or "Symbol" not in watchlist_df.columns:
    st.warning("⚠️ No valid 'Symbol' column found in your watchlist Excel file.")
    symbols = []
else:
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()

st.subheader("📋 Combined Summary Table")
initial_df = pd.DataFrame({
    "Symbol": symbols if symbols else [],
    "CMP": ["" for _ in symbols] if symbols else [],
    "52W_Low": ["" for _ in symbols] if symbols else [],
    "52W_High": ["" for _ in symbols] if symbols else [],
    "EMA200": ["" for _ in symbols] if symbols else [],
    "RSI14": ["" for _ in symbols] if symbols else [],
    "Signal": ["" for _ in symbols] if symbols else [],
    "AlertCondition": ["" for _ in symbols] if symbols else []
})
summary_placeholder = st.empty()
# show placeholder table initially
summary_placeholder.dataframe(initial_df, use_container_width=True, hide_index=True)
last_scan_time = st.caption("Will auto-update after scanning.")

# -----------------------
# Controls (main)
# -----------------------
st.subheader("⚙️ Controls")
col1, col2 = st.columns([1, 2])
with col1:
    run_now = st.button("Run Scan Now")
    if st.button("🔁 Refresh Watchlist from GitHub"):
        load_excel_from_github.clear()
        watchlist_df = load_excel_from_github()
        st.experimental_rerun()
    st.write("")
    st.markdown("**Options**")
    st.write(f"- Parallel downloads: {'✅' if st.session_state['parallel_fetch'] else '❌'}")
with col2:
    st.markdown("**Status:**")
    st.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    if st.session_state.get("auto_scan_enabled", False):
        st.markdown(f"<span style='margin-left:10px;'>🔁 Auto-scan active — every {st.session_state['scan_interval']} seconds</span>", unsafe_allow_html=True)

# -----------------------
# Scan execution
# -----------------------
def _format_and_display(df_results: pd.DataFrame):
    if df_results.empty:
        summary_placeholder.warning("No valid data fetched.")
        return
    # ensure types and format
    df_results = df_results.copy()
    # enforce numeric types where expected
    for col in ("CMP", "EMA200", "RSI14", "52W_Low", "52W_High"):
        if col in df_results.columns:
            df_results[col] = pd.to_numeric(df_results[col], errors="coerce")
    # format floats
    styled = df_results.style.format({
        "CMP": "{:,.2f}",
        "EMA200": "{:,.2f}",
        "RSI14": "{:,.2f}",
        "52W_Low": "{:,.2f}",
        "52W_High": "{:,.2f}",
    })
    # simple row color based on signal
    def row_color(r):
        sig = str(r.get("Signal", ""))
        if "BUY" in sig:
            return ["background-color: #e8f8ee"] * len(r)
        if "SELL" in sig:
            return ["background-color: #fff1f0"] * len(r)
        if "WATCH" in sig:
            return ["background-color: #fff8e6"] * len(r)
        return [""] * len(r)
    styled = styled.apply(row_color, axis=1)
    summary_placeholder.dataframe(styled, use_container_width=True)
    last_scan_time.caption(f"Last scan: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")

if run_now:
    with st.spinner("Scanning..."):
        df_results = run_scan(symbols, parallel=st.session_state["parallel_fetch"])
        _format_and_display(df_results)

# -----------------------
# Auto-scan (safe)
# -----------------------
if st.session_state.get("auto_scan_enabled", False):
    if st_autorefresh:
        try:
            # Only sets up autorefresh - avoid forcing scan on every render to prevent heavy usage.
            st_autorefresh(interval=int(st.session_state["scan_interval"]) * 1000, key="auto_refresh")
            # If you want scans auto-triggered on refresh, you can enable a flag to call run_scan here.
        except Exception:
            st.warning("Auto-refresh misconfigured. Install streamlit-autorefresh to enable auto-scan.")
    else:
        st.warning("Auto-refresh package missing. Run: pip install streamlit-autorefresh")

# -----------------------
# Alert History UI
# -----------------------
st.subheader("📜 Alert History")
ensure_df = st.session_state.get("alert_history", pd.DataFrame())
search_q = st.text_input("Search alert history (symbol or signal)", value="")
if ensure_df.empty:
    st.info("No alerts recorded yet. Run a scan to generate new alerts.")
else:
    df_hist = ensure_df.copy()
    if search_q:
        mask = df_hist.apply(lambda row: row.astype(str).str.contains(search_q, case=False).any(), axis=1)
        df_show = df_hist[mask]
    else:
        df_show = df_hist
    # format display
    st.dataframe(df_show.reset_index(drop=True), use_container_width=True, hide_index=True)
    col_clear, col_export = st.columns([1, 1])
    with col_export:
        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export CSV", csv_bytes, file_name="alert_history.csv", mime="text/csv")
    with col_clear:
        if st.session_state["alert_history"].empty:
            st.button("🧹 Clear History", disabled=True)
        else:
            if st.button("🧹 Clear History"):
                # confirmation to avoid accidental clears
                if st.checkbox("Confirm clear history"):
                    st.session_state["alert_history"] = pd.DataFrame(columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"])
                    st.success("✅ Alert history cleared.")
                    # no forced rerun; clear applied to session_state and table refreshed automatically

# -----------------------
# Debug logs + Test Telegram
# -----------------------
with st.expander("🔍 Debug Logs"):
    logs = st.session_state.get("debug_logs", [])
    if not logs:
        st.write("No debug logs.")
    else:
        for l in logs[-300:]:
            st.text(l)

if st.button("📨 Send Test Telegram Alert"):
    ok = send_telegram("✅ Test alert from Indian Stock Agent Bot!", symbol=None)
    if ok:
        st.success("✅ Telegram test alert sent successfully!")
    else:
        st.error("❌ Telegram test alert failed. Check token/chat_id and throttling.")

# End of file
