# agent.py
import os
import sys
import time
import datetime
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta

WATCHLIST_FILE = "watchlist.txt"
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

if not TELEGRAM_TOKEN or not CHAT_ID:
    sys.exit("Missing TELEGRAM_TOKEN or CHAT_ID environment variables. Set them in Render service settings.")

def load_watchlist(path=WATCHLIST_FILE):
    try:
        with open(path, "r") as f:
            return [l.strip().upper() for l in f if l.strip() and not l.strip().startswith("#")]
    except Exception:
        return []

def safe_float(v):
    try:
        if v is None:
            return None
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram send failed:", e)

def fetch_indicators(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False, threads=False)
        if df is None or df.empty:
            return None
        df["EMA200"] = ta.ema(df["Close"], length=200)
        df["RSI14"] = ta.rsi(df["Close"], length=14)
        last = df.iloc[-1]
        ema = safe_float(last.get("EMA200"))
        rsi = safe_float(last.get("RSI14"))
        close_price = safe_float(last.get("Close"))
        if ema is None or rsi is None or close_price is None:
            return None

        latest_price = close_price
        try:
            intr = yf.download(symbol, period="2d", interval="1m", progress=False, threads=False)
            if intr is not None and not intr.empty:
                val = safe_float(intr["Close"].iloc[-1])
                if val is not None:
                    latest_price = val
        except Exception:
            latest_price = close_price

        return {"ema": ema, "rsi": rsi, "price": latest_price}
    except Exception as e:
        print("fetch_indicators error:", e)
        return None

def main():
    watchlist = load_watchlist()
    if not watchlist:
        print("watchlist empty; add symbols to watchlist.txt")
        return
    print("Agent started for:", watchlist)
    while True:
        for sym in watchlist:
            info = fetch_indicators(sym)
            if info:
                ema = info["ema"]
                rsi = info["rsi"]
                price = info["price"]
                near_ema = (0.98 * ema) <= price <= (1.02 * ema)
                rsi_ok = 30 <= rsi <= 40
                if near_ema and rsi_ok:
                    msg = (
                        f"📊 ALERT: {sym}\n"
                        f"Price: ₹{price:.2f}\n"
                        f"EMA200: ₹{ema:.2f}\n"
                        f"RSI(14): {rsi:.2f}\n"
                        f"Condition: Price near EMA200 & RSI 30-40\n"
                        f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    send_telegram(msg)
            time.sleep(1)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
