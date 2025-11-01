import pandas as pd
import numpy as np
from datetime import timedelta

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ['_'.join([str(x) for x in col if x is not None and str(x)!='']).strip('_') for col in df.columns.values]
    return df

def _find_close_column(df: pd.DataFrame):
    cols = [c for c in df.columns]
    for prefer in ('close','adjclose','adj_close','adjusted_close'):
        for c in cols:
            if c.replace(' ','').replace('_','').lower() == prefer:
                return c
    for c in cols:
        if 'close' in c.lower():
            return c
    return None

def calc_rsi_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMA200, RSI14 and 52W high/low and attach to df. Returns df or raises."""
    if df is None or df.empty:
        raise ValueError('Empty dataframe')
    df = _flatten_columns(df)
    close_col = _find_close_column(df)
    if close_col is None:
        raise KeyError('Close column not found')
    df = df.copy()
    df['Close'] = pd.to_numeric(df[close_col], errors='coerce')
    df = df.dropna(subset=['Close'])
    df.index = pd.to_datetime(df.index)

    df['EMA200'] = df['Close'].ewm(span=200, adjust=False, min_periods=1).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta>0, 0.0)
    loss = -delta.where(delta<0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI14'] = 100.0 - (100.0 / (1.0 + rs))

    last_date = df.index.max()
    cutoff = last_date - timedelta(days=365)
    df_1y = df[df.index >= cutoff]
    if not df_1y.empty:
        h52 = df_1y['Close'].max()
        l52 = df_1y['Close'].min()
    else:
        h52 = df['Close'].max()
        l52 = df['Close'].min()

    df['52W_High'] = h52
    df['52W_Low'] = l52
    return df

def analyze(symbol: str, period='2y'):
    """Download data with yfinance and return analysis dict mirroring original implementation."""
    import yfinance as yf
    df = yf.download(symbol, period=period, interval='1d', progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    try:
        df_ind = calc_rsi_ema(df)
    except Exception:
        return None
    last = df_ind.iloc[-1]
    cmp_ = float(last['Close'])
    ema200 = float(last['EMA200'])
    rsi14 = float(last['RSI14'])
    low52 = float(last['52W_Low'])
    high52 = float(last['52W_High'])

    signal = 'Neutral'
    alert_condition = ''
    if (cmp_ * 0.98 <= ema200 <= cmp_ * 1.02) and (30 <= rsi14 <= 40):
        signal = '🟡 ⚠️ Watch'
        alert_condition = 'EMA200 within ±2% of CMP & RSI between 30–40'
    elif rsi14 < 30:
        signal = '🟢 🔼 BUY'
        alert_condition = 'RSI < 30 (Oversold)'
    elif rsi14 > 70:
        signal = '🔴 🔽 SELL'
        alert_condition = 'RSI > 70 (Overbought)'

    telegram_msg = (
        f"⚡ Alert: {symbol}\n"
        f"CMP = {cmp_:.2f}\n"
        f"EMA200 = {ema200:.2f}\n"
        f"RSI14 = {rsi14:.2f}\n"
        f"Condition: {alert_condition if alert_condition else 'No active signal'}"
    )

    return {
        'Symbol': symbol,
        'CMP': round(cmp_,2),
        '52W_Low': round(low52,2),
        '52W_High': round(high52,2),
        'EMA200': round(ema200,2),
        'RSI14': round(rsi14,2),
        'Signal': signal,
        'AlertCondition': alert_condition,
        'TelegramMessage': telegram_msg
    }
