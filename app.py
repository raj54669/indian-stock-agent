# app.py — Indian Stock Agent (ready-to-paste)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io, os, time, requests
from datetime import datetime, timedelta, timezone
from typing import Optional

# Optional auto-refresh support (safe import)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# Small page-padding tweak (kept from original)
st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# Initialize session state for alert history as DataFrame
if "alert_history" not in st.session_state:
    st.session_state.alert_history = pd.DataFrame(
        columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
    )

# Debug logs storage
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

# Clear confirmation state
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# -----------------------
# Streamlit Config
# -----------------------
st.set_page_config(page_title="📈 Indian Stock Agent – EMA + RSI Alert Bot", layout="wide")
st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# -----------------------
# === Improvement Constants (1. Hardcoded Constants) ===
# -----------------------
EMA_LONG = 200
EMA_SHORT = 50
RSI_PERIOD = 14
WATCH_PCT = 0.02  # ±2% threshold for WATCH
WATCH_RSI_LOW = 30
WATCH_RSI_HIGH = 40
BUY_RSI_THRESHOLD = 30
SELL_RSI_THRESHOLD = 70

IST = timezone(timedelta(hours=5, minutes=30))

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
# Sidebar: upload + status  (kept exactly as original)
# -----------------------
st.sidebar.header("📂 Watchlist Management")
uploaded_file = st.sidebar.file_uploader("Upload new watchlist (Excel)", type=["xlsx"])

# -----------------------
# Load watchlist (uploaded override or GitHub)
# -----------------------
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

if TELEGRAM_TOKEN and CHAT_ID:
    st.sidebar.success("✅ Telegram configured")
else:
    st.sidebar.info("Telegram not configured — alerts disabled")

if GITHUB_TOKEN and GITHUB_REPO:
    st.sidebar.success("✅ GitHub secrets present")
else:
    st.sidebar.warning("GitHub credentials missing (set GITHUB_TOKEN and GITHUB_REPO)")

# sanitize watchlist: ensure Symbol column exists
if not watchlist_df.empty:
    if "Symbol" in watchlist_df.columns:
        watchlist_df["Symbol"] = watchlist_df["Symbol"].astype(str).str.strip()
        watchlist_df = watchlist_df[watchlist_df["Symbol"] != ""]
    else:
        watchlist_df = pd.DataFrame()

# -----------------------
# Telegram helper (updated to support HTML parse_mode for bold symbol)
# -----------------------
def send_telegram(message: str, parse_mode: str = "HTML"):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False
    try:
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": parse_mode
        }
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data=payload,
            timeout=10
        )
        return r.status_code == 200
    except Exception as e:
        # don't crash app for telegram issues, just log warning
        st.warning(f"Telegram send error: {e}")
        return False

# -----------------------
# Robust indicators calculator + analyzer (kept logic but added EMA short)
# -----------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns into single-level strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(x) for x in col if x is not None and str(x) != ""])
            .strip("_")
            for col in df.columns.values
        ]
    return df

def _find_close_column(df: pd.DataFrame):
    """Find the best candidate for Close column (case-insensitive)."""
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
    """
    Input: raw dataframe from yfinance (may be MultiIndex).
    Output: df with EMA200, EMA50, RSI14, 52W_High, 52W_Low (or None on failure).
    """
    try:
        if df is None or df.empty:
            return None

        df = _flatten_columns(df)

        close_col = _find_close_column(df)
        if close_col is None:
            return None

        # Coerce to numeric and drop missing closes
        df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)

        # EMA long and short
        df[f"EMA{EMA_LONG}"] = df["Close"].ewm(span=EMA_LONG, adjust=False, min_periods=1).mean()
        df[f"EMA{EMA_SHORT}"] = df["Close"].ewm(span=EMA_SHORT, adjust=False, min_periods=1).mean()

        # RSI14 (Wilder smoothing via ewm alpha = 1/14)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

        # 52-week high/low using exactly 365 calendar days from last available date
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

    except Exception:
        return None

def analyze(symbol: str):
    """
    Download data and return a single-row dict with indicators and signal.
    Logic unchanged but uses constants and includes a trend flag (EMA crossover).
    """
    try:
        # Fetch 2 years of data for stable EMA200/EMA50
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        df_ind = calc_rsi_ema(df)
        if df_ind is None or df_ind.empty:
            return None

        last = df_ind.iloc[-1]

        cmp_ = float(last["Close"]) if not pd.isna(last["Close"]) else None
        ema200 = float(last[f"EMA{EMA_LONG}"]) if f"EMA{EMA_LONG}" in last.index and not pd.isna(last[f"EMA{EMA_LONG}"]) else None
        ema50 = float(last[f"EMA{EMA_SHORT}"]) if f"EMA{EMA_SHORT}" in last.index and not pd.isna(last[f"EMA{EMA_SHORT}"]) else None
        rsi14 = float(last["RSI14"]) if "RSI14" in last.index and not pd.isna(last["RSI14"]) else None
        low52 = float(last["52W_Low"]) if "52W_Low" in last.index and not pd.isna(last["52W_Low"]) else None
        high52 = float(last["52W_High"]) if "52W_High" in last.index and not pd.isna(last["52W_High"]) else None

        if cmp_ is None or ema200 is None or rsi14 is None:
            return None

        # -------------------------------
        # 🔔 Signal Conditions (kept original logic)
        # -------------------------------
        signal = "Neutral"
        alert_condition = ""

        # WATCH: EMA200 within ±2% of CMP & RSI between 30–40
        if (cmp_ * (1 - WATCH_PCT) <= ema200 <= cmp_ * (1 + WATCH_PCT)) and (WATCH_RSI_LOW <= rsi14 <= WATCH_RSI_HIGH):
            signal = "🟡 WATCH"
            alert_condition = "EMA200 within ±2% of CMP & RSI between 30–40"

        # BUY: RSI < 30
        elif rsi14 < BUY_RSI_THRESHOLD:
            signal = "🟢 BUY"
            alert_condition = "RSI < 30 (Oversold)"

        # SELL: RSI > 70
        elif rsi14 > SELL_RSI_THRESHOLD:
            signal = "🔴 SELL"
            alert_condition = "RSI > 70 (Overbought)"

        # Trend confirmation (EMA crossover) used only for alerts text
        trend_text = None
        if ema50 is not None:
            if ema50 > ema200:
                trend_text = f"Trend: Uptrend (EMA{EMA_SHORT} > EMA{EMA_LONG})"
            elif ema50 < ema200:
                trend_text = f"Trend: Downtrend (EMA{EMA_SHORT} < EMA{EMA_LONG})"
            else:
                trend_text = f"Trend: Neutral (EMA{EMA_SHORT} == EMA{EMA_LONG})"

        # Telegram message format (improved): Bold symbol using HTML parse mode + trend confirmation
        telegram_msg_lines = [
            "⚡ Alert:",
            f"<b>{symbol}</b>",
            f"CMP = {cmp_:.2f}",
            f"EMA{EMA_LONG} = {ema200:.2f}",
            f"RSI14 = {rsi14:.2f}",
            f"Condition: {alert_condition if alert_condition else 'No active signal'}"
        ]
        if trend_text:
            telegram_msg_lines.append(trend_text)

        telegram_msg = "\n".join(telegram_msg_lines)

        return {
            "Symbol": symbol,
            "CMP": round(cmp_, 2),
            "52W_Low": round(low52, 2) if low52 is not None else None,
            "52W_High": round(high52, 2) if high52 is not None else None,
            "EMA200": round(ema200, 2),
            "RSI14": round(rsi14, 2),
            "Signal": signal,
            "AlertCondition": alert_condition,
            "TelegramMessage": telegram_msg,
            "TrendText": trend_text or ""
        }

    except Exception as e:
        # Do not raise; return None and let caller handle
        st.error(f"{symbol}: {e}")
        return None

# -----------------------
# Main UI (kept original layout/text)
# -----------------------
st.title("📊 Indian Stock Agent – EMA + RSI Alert Bot")

if watchlist_df.empty or "Symbol" not in watchlist_df.columns:
    st.warning("⚠️ No valid 'Symbol' column found in your watchlist Excel file.")
    symbols = []
else:
    symbols = watchlist_df["Symbol"].dropna().astype(str).tolist()

# Combined summary placeholder (only one)
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

# -----------------------
# Controls (kept original)
# -----------------------
st.subheader("⚙️ Controls")
col1, col2 = st.columns([1, 2])

with col1:
    run_now = st.button("Run Scan Now", key="run_now_btn")
    interval = st.number_input("Interval (sec)", value=60, step=5, min_value=5, key="interval_input")
    auto = st.checkbox("Enable Auto-scan", key="auto_chk")

with col2:
    st.markdown("**Status:**")
    st.write(f"- GitHub Repo: `{GITHUB_REPO or 'N/A'}`")
    st.write(f"- Token: {'✅' if GITHUB_TOKEN else '❌'}")
    if auto:
        st.markdown(
            f"<span style='margin-left:10px;'>🔁 Auto-scan active — every {interval} seconds</span>",
            unsafe_allow_html=True
        )

# Move yfinance version info into sidebar (kept)
try:
    st.sidebar.caption(f"📦 yfinance version: {yf.__version__}")
except Exception:
    pass

# -----------------------
# Run Scan (with Progress bar + Debug logs)
# -----------------------
def run_scan():
    results = []
    st.session_state.debug_logs = []  # reset debug logs every run
    total = len(symbols)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, symbol in enumerate(symbols, start=1):
        try:
            st.session_state.debug_logs.append(f"Fetching {symbol} ...")
            status_text.text(f"Fetching {symbol} ({idx}/{total})")
            # fetch 1 year of daily to compute indicators (same as original run_scan)
            df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)

            if df is None or df.empty:
                msg = f"⚠️ No data for {symbol}"
                st.session_state.debug_logs.append(msg)
                # original behavior: continue
                continue

            df = calc_rsi_ema(df)
            if df is None or df.empty:
                st.session_state.debug_logs.append(f"❌ {symbol}: no indicator data")
                continue

            last = df.iloc[-1]

            # safe extraction
            cmp_ = float(last["Close"])
            ema200 = float(last[f"EMA{EMA_LONG}"])
            ema50 = float(last[f"EMA{EMA_SHORT}"]) if f"EMA{EMA_SHORT}" in last.index else None
            rsi14 = float(last["RSI14"])
            high_52w = float(last["52W_High"])
            low_52w = float(last["52W_Low"])

            # Determine signal (kept original conditions)
            signal = "Neutral"
            condition_desc = None

            if cmp_ > ema200 and rsi14 < BUY_RSI_THRESHOLD:
                signal = "🟢 BUY"
                condition_desc = "RSI < 30 and CMP above EMA200"
            elif cmp_ < ema200 and rsi14 > SELL_RSI_THRESHOLD:
                signal = "🔴 SELL"
                condition_desc = "RSI > 70 and CMP below EMA200"
            elif abs(cmp_ - ema200) / cmp_ <= WATCH_PCT and WATCH_RSI_LOW <= rsi14 <= WATCH_RSI_HIGH:
                signal = "🟡 WATCH"
                condition_desc = "EMA200 within ±2% of CMP & RSI between 30–40"

            # Add to alert history + telegram if a signal triggered (KEEP ORIGINAL)
            if signal in ("🟢 BUY", "🔴 SELL", "🟡 WATCH"):
                add_to_alert_history(symbol, signal, cmp_, ema200, rsi14)

                # Build Telegram message (new format with bold symbol and trend confirmation)
                trend_text = ""
                if ema50 is not None:
                    if ema50 > ema200:
                        trend_text = f"\nTrend: Uptrend (EMA{EMA_SHORT} > EMA{EMA_LONG})"
                    elif ema50 < ema200:
                        trend_text = f"\nTrend: Downtrend (EMA{EMA_SHORT} < EMA{EMA_LONG})"
                    else:
                        trend_text = f"\nTrend: Neutral (EMA{EMA_SHORT} == EMA{EMA_LONG})"

                emoji = "⚡" if "WATCH" in signal else ("📈" if "BUY" in signal else "📉")
                # HTML bold for symbol; send_telegram uses parse_mode=HTML
                alert_msg = (
                    f"{emoji} Alert:\n"
                    f"<b>{symbol}</b>\n"
                    f"CMP = {cmp_:.2f}\n"
                    f"EMA{EMA_LONG} = {ema200:.2f}\n"
                    f"RSI14 = {rsi14:.2f}\n"
                    f"Condition: {condition_desc}{trend_text}"
                )
                send_telegram(alert_msg, parse_mode="HTML")

            # Append result row (kept original columns)
            results.append({
                "Symbol": symbol,
                "CMP": round(cmp_, 2),
                "52W_Low": round(low_52w, 2),
                "52W_High": round(high_52w, 2),
                "EMA200": round(ema200, 2),
                "RSI14": round(rsi14, 2),
                "Signal": signal
            })

            st.session_state.debug_logs.append(f"✅ OK {symbol}: CMP={cmp_} EMA200={ema200} RSI={rsi14}")

        except Exception as e:
            msg = f"❌ {symbol}: {e}"
            st.session_state.debug_logs.append(msg)
            # keep same behavior: show error but continue
            st.error(msg)

        # update progress
        progress = int((idx / total) * 100) if total else 100
        progress_bar.progress(progress)
        status_text.text(f"Progress: {progress}%")

    # Update combined table
    if results:
        df_out = pd.DataFrame(results)
        # ensure Signal column contains icons (already included)
        summary_placeholder.dataframe(df_out, use_container_width=True, hide_index=True)
        last_scan_time.caption(f"Last scan: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        summary_placeholder.warning("No valid data fetched.")

    # finished: keep debug logs collapsed available
    progress_bar.empty()
    status_text.empty()

# -----------------------
# 🔔 Alert History Table (with Confirm Clear button) — (3. Confirm Clear button)
# -----------------------
def add_to_alert_history(symbol: str, signal: str, cmp_: float, ema200: float, rsi14: float):
    """
    Append a new alert entry to the Streamlit session state's alert history DataFrame.
    """
    if "alert_history" not in st.session_state or not isinstance(st.session_state["alert_history"], pd.DataFrame):
        st.session_state["alert_history"] = pd.DataFrame(
            columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
        )

    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([{
        "Date & Time (IST)": ts,
        "Symbol": symbol,
        "Signal": signal,
        "CMP": round(float(cmp_), 2),
        "EMA200": round(float(ema200), 2),
        "RSI14": round(float(rsi14), 2),
    }])
    st.session_state["alert_history"] = pd.concat(
        [st.session_state["alert_history"], new_row],
        ignore_index=True
    )
    expected_cols = ["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
    st.session_state["alert_history"] = st.session_state["alert_history"][expected_cols]

st.subheader("📜 Alert History")

# Ensure alert_history exists
if "alert_history" not in st.session_state or not isinstance(st.session_state["alert_history"], pd.DataFrame):
    st.session_state["alert_history"] = pd.DataFrame(
        columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
    )

# show table if exists
if not st.session_state["alert_history"].empty:
    st.dataframe(st.session_state["alert_history"], use_container_width=True, hide_index=True)

    # First click shows confirm button; second click actually clears
    if not st.session_state.confirm_clear:
        if st.button("🧹 Clear History"):
            # show confirm
            st.session_state.confirm_clear = True
            st.warning("Please confirm clearing alert history. Click the confirm button below to proceed.")
    else:
        # show explicit confirm button
        if st.button("Confirm Clear History (final)"):
            st.session_state["alert_history"] = pd.DataFrame(
                columns=["Date & Time (IST)", "Symbol", "Signal", "CMP", "EMA200", "RSI14"]
            )
            st.success("✅ Alert history cleared!")
            st.session_state.confirm_clear = False
            # refresh UI to reflect cleared table
            st.experimental_rerun()
        # allow cancel
        if st.button("Cancel"):
            st.session_state.confirm_clear = False
            st.info("Clear cancelled.")

else:
    st.info("No alerts recorded yet. Run a scan to generate new alerts.")

# -----------------------
# Debug Logs (2. Debug logs)
# -----------------------
with st.expander("🔍 Debug Logs", expanded=False):
    if not st.session_state.debug_logs:
        st.info("No debug logs yet.")
    else:
        # print logs in order; show icons for types
        for line in st.session_state.debug_logs:
            # simple heuristics for emoji
            if line.startswith("✅") or "OK" in line:
                st.success(line)
            elif line.startswith("❌") or "Error" in line or "No data" in line or line.startswith("⚠️"):
                st.error(line)
            else:
                st.write(line)

# -----------------------
# Buttons and Actions (kept original)
# -----------------------
if run_now:
    run_scan()

test_telegram = st.button("📨 Send Test Telegram Alert")
if test_telegram:
    success = send_telegram("✅ Test alert from Indian Stock Agent Bot!", parse_mode="HTML")
    if success:
        st.success("✅ Telegram test alert sent successfully!")
    else:
        st.error("❌ Telegram send failed. Check your token or chat_id.")

# -----------------------
# Auto-scan via streamlit-autorefresh (kept original behavior)
# -----------------------
if auto:
    if st_autorefresh:
        refresh_interval_ms = int(interval) * 1000
        st_autorefresh(interval=refresh_interval_ms, key="auto_refresh")
        run_scan()
    else:
        st.warning("⚠️ Auto-refresh package missing. Run: pip install streamlit-autorefresh")

# end of file
