import base64, io
import pandas as pd
import requests
import streamlit as st

def github_raw_headers(token):
    auth_scheme = 'token' if not str(token).startswith('github_pat_') else 'Bearer'
    return {
        'Authorization': f'{auth_scheme} {token}',
        'Accept': 'application/vnd.github.v3.raw',
        'User-Agent': 'streamlit-indian-stock-agent'
    }

@st.cache_data(ttl=120)
def load_watchlist_from_github():
    try:
        token = st.secrets['GITHUB_TOKEN']
        repo = st.secrets['GITHUB_REPO']
        branch = st.secrets.get('GITHUB_BRANCH','main')
        path = st.secrets.get('GITHUB_FILE_PATH','watchlist.xlsx')
    except Exception:
        return pd.DataFrame()
    url = f'https://api.github.com/repos/{repo}/contents/{path}?ref={branch}'
    try:
        r = requests.get(url, headers=github_raw_headers(token), timeout=10)
        r.raise_for_status()
        data = r.json()
        content = data.get('content','')
        decoded = base64.b64decode(content)
        return pd.read_excel(io.BytesIO(decoded))
    except Exception:
        return pd.DataFrame()

def upload_watchlist_to_github(uploaded_file) -> bool:
    try:
        token = st.secrets['GITHUB_TOKEN']
        repo = st.secrets['GITHUB_REPO']
        branch = st.secrets.get('GITHUB_BRANCH','main')
        path = st.secrets.get('GITHUB_FILE_PATH','watchlist.xlsx')
    except Exception:
        st.error('Missing GitHub secrets')
        return False

    get_url = f'https://api.github.com/repos/{repo}/contents/{path}?ref={branch}'
    headers = github_raw_headers(token)
    sha = None
    try:
        r = requests.get(get_url, headers=headers, timeout=10)
        r.raise_for_status()
        sha = r.json().get('sha')
    except Exception:
        sha = None

    content_b64 = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    payload = {'message':'Updated watchlist via Streamlit','content':content_b64,'branch':branch}
    if sha:
        payload['sha']=sha
    put_url = f'https://api.github.com/repos/{repo}/contents/{path}'
    try:
        res = requests.put(put_url, headers=headers, json=payload, timeout=20)
        res.raise_for_status()
        return True
    except Exception as e:
        st.error(f'GitHub upload failed: {e}')
        return False
