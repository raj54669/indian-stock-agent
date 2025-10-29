import streamlit as st
import yfinance as yf
import pandas_ta as ta
import requests
import os
import time
import threading
import pandas as pd

# Function to send Telegram alerts
def send_telegram_alert(message):
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print(f"Telegram alert sent: {message}")
    else:
        print("Failed to send Telegram alert.")

# Function to fetch stock data and calculate EMA, RSI
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y", interval="1d")  # 1 year of data
    
    # Calculate 200-day EMA
    data['200_EMA'] = ta.ema(data['Close'], length=200)

    # Calculate RSI (14-period)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    
    return data

# Function to check conditions for EMA and RSI
def check_conditions(ticker):
    data = get_stock_data(ticker)

    latest_price = data['Close'][-1]
    ema200_value = data['200_EMA'][-1]
    rsi_value = data['RSI'][-1]

    # Condition 1: Price near 200-day EMA (within ±2%)
    if (latest_price > (ema200_value * 0.98)) and (latest_price < (ema200_value * 1.02)):
        ema_condition_met = True
    else:
        ema_condition_met = False

    # Condition 2: RSI between 30 and 40
    if 30 < rsi_value < 40:
        rsi_condition_met = True
    else:
        rsi_condition_met = False

    return ema_condition_met, rsi_condition_met, latest_price, ema200_value, rsi_value

# Function to load watchlist from an Excel file
def load_watchlist_from_excel(file_path="watchlist.xlsx"):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        # Assuming the file has a column named 'Ticker' with stock symbols
        watchlist = df['Ticker'].dropna().tolist()
        return watchlist
    except Exception as e:
        print(f"Error loading watchlist: {e}")
        return []

# Function to track multiple stocks from a watchlist
def track_stock_from_watchlist():
    # Load stock tickers from the watchlist Excel file
    watchlist = load_watchlist_from_excel()

    if not watchlist:
        print("No stocks in the watchlist.")
        return

    for ticker in watchlist:
        ema_condition_met, rsi_condition_met, latest_price, ema200_value, rsi_value = check_conditions(ticker)
        
        # Check if both conditions are met
        if ema_condition_met and rsi_condition_met:
            message = f"🚨 Alert: {ticker}\n" \
                      f"Price: ${latest_price}\n" \
                      f"200-day EMA: ${ema200_value}\n" \
                      f"RSI: {rsi_value}\n" \
                      "Conditions met: Price near EMA and RSI between 30-40."
            send_telegram_alert(message)
        else:
            print(f"{ticker}: Conditions not met")

    time.sleep(60)  # Wait for a minute before checking again

# Run this function to track stock every minute
def start_tracking():
    track_thread = threading.Thread(target=track_stock_from_watchlist)
    track_thread.daemon = True  # Allow thread to exit when the main program exits
    track_thread.start()

# Streamlit interface
st.title("📊 Stock Price Monitor")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

if ticker:
    st.write(f"Tracking stock: {ticker}")
    # Start the background task to track the stock price
    start_tracking()

# Display latest price and chart for input stock
if ticker:
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")
    st.line_chart(data['Close'])
