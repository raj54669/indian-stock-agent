# Indian Stock Monitor (Cloud)

Files:
- agent.py : runs as a background worker to check triggers and send Telegram alerts
- dashboard.py : Streamlit web dashboard showing live stats
- watchlist.txt : list of NSE tickers (one per line)

Deploy:
1. Push repo to GitHub.
2. Sign up / login to Render, connect GitHub.
3. Create Web Service for dashboard: run `streamlit run dashboard.py --server.port $PORT`
4. Create Background Worker for agent: run `python agent.py`
5. On both services, set environment variables:
   - TELEGRAM_TOKEN
   - CHAT_ID
   - (optional) CHECK_INTERVAL
