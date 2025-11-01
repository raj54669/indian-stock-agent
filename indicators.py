import pandas as pd
import numpy as np

# --------------------------------------------
# ✅ Calculate RSI (modular, standalone)
# --------------------------------------------
def calculate_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a pandas Series of close prices."""
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --------------------------------------------
# ✅ Calculate EMA (modular, standalone)
# --------------------------------------------
def calculate_ema(close_series: pd.Series, span: int = 200) -> pd.Series:
    """Compute EMA from a pandas Series of close prices."""
    return close_series.ewm(span=span, adjust=False).mean()


# --------------------------------------------
# ✅ Analyze stock logic (unchanged core)
# --------------------------------------------
def analyze_stock(df: pd.DataFrame):
    """Perform EMA + RSI analysis and return latest metrics."""
    df["EMA200"] = calculate_ema(df["Close"], span=200)
    df["RSI14"] = calculate_rsi(df["Close"], period=14)

    last = df.iloc[-1]
    cmp_ = float(last["Close"])
    ema200 = float(last["EMA200"])
    rsi14 = float(last["RSI14"])

    signal = "Neutral"
    condition_desc = None

    if cmp_ > ema200 and rsi14 < 30:
        signal = "🟢 BUY"
        condition_desc = "RSI < 30 and CMP above EMA200"

    elif cmp_ < ema200 and rsi14 > 70:
        signal = "🔴 SELL"
        condition_desc = "RSI > 70 and CMP below EMA200"

    elif abs(cmp_ - ema200) / cmp_ <= 0.02 and 30 <= rsi14 <= 40:
        signal = "🟡 WATCH"
        condition_desc = "EMA200 within ±2% of CMP & RSI between 30–40"

    return cmp_, ema200, rsi14, signal, condition_desc
