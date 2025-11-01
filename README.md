# 📈 Indian Stock Agent – EMA + RSI Alert Bot

A modular, production-ready Streamlit app for Indian stock market scanning, combining:
- EMA200 and RSI14 technical indicators  
- 52-week high/low tracking  
- Telegram alert notifications  
- GitHub-based Excel watchlist management (with upload + commit history)  
- Auto-scan mode (enabled by default)  

---

## 🚀 Features
✅ Load watchlist directly from GitHub  
✅ Upload Excel to replace GitHub file (creates new commit each time)  
✅ Auto-scan and Telegram alerts for BUY, SELL, WATCH signals  
✅ Session-based alert history  
✅ Clean UI with auto-hide clear history button  
✅ Modular code structure for easy maintenance  

---

## 📁 Project Structure
project/
│
├── app.py # Main Streamlit entry point
├── indicators.py # RSI, EMA200, and signal logic
├── alerts.py # Telegram + alert history
├── github_utils.py # Load/save Excel on GitHub
├── ui_helpers.py # Streamlit UI helpers
└── tests/
├── test_indicators.py # Unit tests for indicators
└── test_alerts.py # Unit tests for alerts
