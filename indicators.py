"""
indicators.py
---------------------------------
Contains indicator calculations and signal generation logic.
This module focuses purely on data science logic — no Streamlit or I/O operations.

Functions:
    - calculate_ema200(df)
    - calculate_rsi14(df)
    - generate_signal(df)
"""

import pandas as pd
import numpy as np

# ---------------------------
# 📘 Exponential Moving Average (EMA200)
# ---------------------------
def calculate_ema200(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the 200-period Exponential Moving Average (EMA) for 'Close' prices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'Close' column.

    Returns
    -------
    pd.Series
        The EMA200 values.
    """
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column to compute EMA200.")
    return df["Close"].ewm(span=200, adjust=False).mean()


# ---------------------------
# 📘 Relative Strength Index (RSI14)
# ---------------------------
def calculate_rsi14(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the 14-period RSI using the standard Wilder's method.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'Close' column.

    Returns
    -------
    pd.Series
        The RSI14 values (0–100 scale).
    """
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column to compute RSI.")

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


# ---------------------------
# 📘 Signal Generator
# ---------------------------
def generate_signal(df: pd.DataFrame) -> str:
    """
    Generates BUY/SELL/WATCH signals based on EMA200 and RSI14 strategy.

    Rules:
        - BUY  : Price > EMA200 and RSI14 < 70
        - SELL : Price < EMA200 and RSI14 > 30
        - WATCH: Otherwise

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Close', 'EMA200', and 'RSI14' columns.

    Returns
    -------
    str
        One of "BUY", "SELL", or "WATCH".
    """
    try:
        close = df["Close"].iloc[-1]
        ema200 = df["EMA200"].iloc[-1]
        rsi = df["RSI14"].iloc[-1]
    except IndexError:
        return "WATCH"  # Empty or malformed DataFrame

    if close > ema200 and rsi < 70:
        return "BUY"
    elif close < ema200 and rsi > 30:
        return "SELL"
    else:
        return "WATCH"
