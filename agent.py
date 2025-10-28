# agent.py
import time
import datetime
import requests
import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta

# CONFIG via environment variables (set these in Render)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")  # your Telegram chat id
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))  # seconds

# Safety
if not TELEGRAM_TOKEN or not CHAT_ID:
    raise Exception("Set TELEGRAM_TOKEN and CHAT_ID environment variables in your Render service.")

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("Telegram send error:", e)

def fetch_indicator_data(symbol):
    """
    Fetch daily data for indicators (EMA200 & RSI14).
    Return: dict with ema200, rsi14, last_daily_close, last_daily_date
    """
    # 1 year daily history to compute 200-day EMA reliably
    df = yf.download(symbol, period="1y", interval="1d", progress=False, threads=False)
    if df.empty or "Close" not in df:
        raise ValueError(f"No daily data for {symbol}")

    df = df.dropna(subset=["Close"])
    # Ensure we have enough data
    if len(df) < 220:
        # proceed but EMA may be less reliable
        pass

    df["EMA200"] = ta.ema(df["Close"], length=200)
    df["RSI14"] = ta.rsi(df["Close"], length=14)

    last_row = df.iloc[-1]
    return {
        "ema200": float(last_row["EMA200"]),
        "rsi14": float(last_row["RSI14"]),
        "last_daily_close": float(last_row["Close"]),
        "last_daily_date": last_row.name.strftime("%Y-%m-%d")
    }

def fetch_latest_price(symbol):
    """
    Try to fetch latest intraday price (most recent 1-minute bar). Fall back to the daily close.
    """
    try:
        intraday = yf.download(symbol, period="2d", interval="1m", progress=False, threads=False)
        if intraday is not None and not intraday.empty:
            last = intraday.iloc[-1]
            price = float(last["Close"])
            timestamp = intraday.index[-1].to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
            return price, timestamp
    except Exception as e:
        print("intraday fetch error:", e)

    # fallback
    try:
        daily = yf.download(symbol, period="2d", interval="1d", progress=False, threads=False)
        if daily is not None and not daily.empty:
            price = float(daily.iloc[-1]["Close"])
            timestamp = daily.index[-1].strftime("%Y-%m-%d")
            return price, timestamp
    except Exception as e:
        print("daily fallback error:", e)

    raise ValueError(f"Could not fetch price for {symbol}")

def check_symbol(symbol):
    try:
        indicators = fetch_indicator_data(symbol)
        price, price_time = fetch_latest_price(symbol)

        ema = indicators["ema200"]
        rsi = indicators["rsi14"]

        # Condition: price within ±2% of EMA200 AND RSI between 30 and 40
        near_ema = (0.98 * ema) < price < (1.02 * ema)
        rsi_ok = (30 < rsi < 40)

        triggered = near_ema and rsi_ok

        result = {
            "symbol": symbol,
            "price": price,
            "price_time": price_time,
            "ema200": ema,
            "rsi14": rsi,
            "near_ema": near_ema,
            "rsi_ok": rsi_ok,
            "triggered": triggered,
            "checked_at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }

        if triggered:
            msg = (
                f"📊 [LIVE ALERT]\n"
                f"Stock: {symbol}\n"
                f"Price: ₹{price:.2f} (as of {price_time})\n"
                f"EMA200: ₹{ema:.2f}\n"
                f"RSI(14): {rsi:.2f}\n"
                f"Condition: Price within ±2% of 200-EMA AND RSI 30–40.\n"
                f"Checked at: {result['checked_at']}"
            )
            send_telegram(msg)

        # log to console for debugging in Render logs
        print(result)
        return result
    except Exception as e:
        print(f"Error checking {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}

def load_watchlist(path="watchlist.txt"):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    return lines

def main_loop():
    watchlist = load_watchlist()
    if not watchlist:
        raise Exception("watchlist.txt is empty or not present in repo.")
    print(f"Starting agent for {len(watchlist)} symbols. Interval = {CHECK_INTERVAL}s")
    while True:
        start = time.time()
        for sym in watchlist:
            check_symbol(sym)
            # small pause to reduce bursts
            time.sleep(1)
        # sleep until next interval
        elapsed = time.time() - start
        to_sleep = max(5, CHECK_INTERVAL - elapsed)
        time.sleep(to_sleep)

if __name__ == "__main__":
    main_loop()
