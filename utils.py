# Compatibility helpers used by different app variants.
from github_utils import load_watchlist_from_github, upload_watchlist_to_github
from alerts import send_telegram, add_to_alert_history

def load_excel_from_github():
    return load_watchlist_from_github()

def upload_to_github(uploaded_file):
    return upload_watchlist_to_github(uploaded_file)
