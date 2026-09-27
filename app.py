import sys
import os
import json
import urllib3
import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="AI Stock Analytics Pro - Ultimate Edition", page_icon="💎", layout="wide")

# --- מאגר קטגוריות מותאם אישית ---
THEMATIC_TICKERS = {
    "טכנולוגיה": ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'ACN', 'CSCO', 'INTC', 'QCOM', 'IBM'],
    "קריפטו": ['COIN', 'MSTR', 'MARA', 'RIOT', 'CLSK', 'HUT', 'BITF'],
    "ביטקויין": ['BTC-USD', 'IBIT', 'FBTC', 'ARKB', 'BITB', 'BITO'],
    "בינה מלאכותית": ['NVDA', 'AMD', 'SMCI', 'PLTR', 'MSFT', 'GOOGL', 'META', 'TSM', 'ASML', 'CRWD', 'PANW'],
    "מחשב קוונטי": ['IONQ', 'QBTS', 'RGTI', 'IBM', 'HON', 'GOOGL'],
    "גיימינג": ['EA', 'TTWO', 'RBLX', 'NTES', 'SONY', 'MSFT', 'TCEHY'],
    "זהב": ['GLD', 'IAU', 'GDX', 'NEM', 'GOLD', 'AEM', 'FNV'],
    "אנרגיה ותשתיות": ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PXD', 'VLO', 'NEP', 'BIP'],
    "קרנות ישראליות": ['EIS', 'IZRL', 'ISRA', 'ITEQ', 'TA35.TA', 'TA125.TA', 'LEUMI.TA', 'POALIM.TA', 'NICE.TA'],
    "בריאות": ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'PFE', 'DHR', 'AMGN', 'ISRG'],
    "פיננסים": ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C'],
    "תקשורת": ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS'],
    "תעשייה": ['CAT', 'GE', 'UNP', 'HON', 'BA', 'UPS', 'RTX', 'LMT', 'DE', 'ADP'],
    "תחום הצריכה": ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'WMT', 'PG', 'KO', 'PEP', 'COST'],
    "שירותים": ['NEE', 'DUK', 'SO', 'SRE', 'AEP', 'D', 'EXC', 'XEL'],
    "נדל\"ן": ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'SPG', 'WELL']
}

@st.cache_data
def get_stock_universe():
    israeli_stocks = THEMATIC_TICKERS["קרנות ישראליות"]
    etfs = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'TLT']
    all_thematic = [t for sublist in THEMATIC_TICKERS.values() for t in sublist]
    
    # מאגר הגיבוי
    sp500_hardcoded = [
        'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 
        'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 
        'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'AAP', 'AMAT', 'APTV', 'ACGL', 
        'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 
        'BAC', 'BK', 'BBWI', 'BAX', 'BDX', 'BRK-B', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BX', 'BA', 'BKNG', 
        'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF-B', 'BG', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 
        'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'COR', 'CNC', 'CNP', 
        'CDAY', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 
        'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 
        'COO', 'CPRT', 'GLW', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 
        'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 
        'DOW', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'LLY', 'EMR', 
        'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETY', 'EVRG', 'ES', 'EXC', 
        'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 
        'FE', 'FI', 'FLT', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GEHC', 
        'GEN', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'HAL', 'HIG', 'HAS', 
        'HCA', 'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 
        'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY', 'IR', 'PODD', 'INTC', 
        'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JKHY', 'J', 'JNJ', 
        'JCI', 'JPM', 'JNPR', 'K', 'KVVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 
        'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 
        'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 
        'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 
        'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 
        'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 
        'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PANW', 'PARA', 'PAYX', 
        'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 
        'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 
        'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 
        'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SNA', 
        'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 
        'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 
        'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UHS', 'VLO', 
        'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VTRS', 'VICI', 'V', 'VMC', 'WAB', 'WBA', 'WMT', 
        'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WRK', 'WY', 'WHR', 'WMB', 'WTW', 
        'GWW', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS'
    ]
    
    try:
        headers = {'User-Agent': 'AIStockPro/2.0 (contact@example.com)'}
        url = 'https://www.sec.gov/files/company_tickers.json'
        res = requests.get(url, headers=headers, timeout=10)
        data = res.json()
        sec_tickers = [item['ticker'] for item in data.values()]
        massive_universe = sorted(list(set(sec_tickers + all_thematic + israeli_stocks + etfs)))
        return massive_universe, sec_tickers, israeli_stocks
    except Exception:
        massive_universe = sorted(list(set(sp500_hardcoded + all_thematic + israeli_stocks + etfs)))
        return massive_universe, sp500_hardcoded, israeli_stocks

WATCHLIST_FILE = "watchlist.json"
DEFAULT_WATCHLIST = ['SPY', 'QQQ', 'BTC-USD', 'NVDA', 'LEUMI.TA', 'TSLA']

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            pass
    return DEFAULT_WATCHLIST.copy()

def save_watchlist(watchlist):
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(watchlist, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"שגיאה בשמירת הרשימה לקובץ: {e}")

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

def send_telegram_msg(bot_token, chat_id, text):
    if not bot_token or not chat_id:
        return False, "נא להגדיר Token ו-Chat ID בסרגל הצד."
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return True, "התראה נשלחה בהצלחה לטלגרם! 📱"
        return False, f"שגיאה מהשרת: {res.text}"
    except Exception as e:
        return False, str(e)

def fetch_yahoo_chart(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
    }
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=8)
        if response.status_code == 200:
            data = response.json()
            return data.get('chart', {}).get('result')
    except Exception:
        pass
    return None

@st.cache_data(ttl=1800)
def fetch_live_data(raw_ticker):
    ticker = raw_ticker.strip().upper()
    tickers_to_try = [ticker]
    
    if '.' in ticker and not ticker.endswith('.TA'):
        tickers_to_try.append(ticker.replace('.', '-'))
    if not ticker.endswith('.TA'):
        tickers_to_try.append(f"{ticker}.TA")

    result = None
    successful_ticker = None

    for t in tickers_to_try:
        result = fetch_yahoo_chart(t)
        if result and len(result) > 0:
            successful_ticker = t
            break

    if not result or 'timestamp' not in result[0] or not result[0]['timestamp']:
        return None, None, None

    meta = result[0]['meta']
    currency_code = meta.get('currency', 'USD')
    timestamps = result[0]['timestamp']
    quote = result[0]['indicators']['quote'][0]
    
    df = pd.DataFrame({
        'Date': pd.to_datetime(timestamps, unit='s'),
        'Open': quote.get('open'),
        'Close': quote.get('close'),
        'High': quote.get('high'),
        'Low': quote.get('low'),
        'Volume': quote.get('volume')
    }).dropna()
    
    if len(df) < 60:
        return None, None, None

    sp500_result = fetch_yahoo_chart('^GSPC')
    if sp500_result and 'timestamp' in sp500_result[0]:
        sp_timestamps = sp500_result[0]['timestamp']
        sp_closes = sp500_result[0]['indicators']['quote'][0]['close']
        df_sp = pd.DataFrame({
            'Date': pd.to_datetime(sp_timestamps, unit='s'),
            'SP500_Close': sp_closes
        }).dropna()
        df = pd.merge(df, df_sp, on='Date', how='left').ffill()
    else:
        df['SP500_Close'] = df['Close']

    if currency_code in ['ILA', 'ILS']:
        if currency_code == 'ILA':
            df['Open'] /= 100
            df['Close'] /= 100
            df['High'] /= 100
            df['Low'] /= 100
        curr = "₪"
    else:
        curr = "$"
        
    return df, curr, successful_ticker

def process_features_and_model(df):
    df['Return'] = df['Close'].pct_change()
    df['SP500_Return'] = df['SP500_Close'].pct_change()
    df['Lag_1'] = df['Return'].shift(1)
    df['Lag_2'] = df['Return'].shift(2)
    
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Vol_SMA_20'] = df['Volume'].rolling(20).mean()
    df['Vol_SMA_50'] = df['Volume'].rolling(50).mean()
    
    df['SMA_20_Ratio'] = df['Close'] / df['SMA_20']
    df['SMA_Trend'] = df['SMA_20'] / df['SMA_50']
    
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)

    # --- הוספת בולינג'ר באנדס ---
    std_20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * std_20)
    df['BB_Lower'] = df['SMA_20'] - (2 * std_20)
    # מדד רוחב הרצועות (לזיהוי Squeeze)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']
    
    df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-8)

    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    df['ATR'] = atr14
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr14)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
    df['OBV_Trend'] = (df['OBV'] - df['OBV_EMA']) / (df['Volume'].rolling(20).mean() + 1e-8)
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    tp_diff = typical_price.diff()
    pos_mf = raw_money_flow.where(tp_diff > 0, 0).rolling(14).sum()
    neg_mf = raw_money_flow.where(tp_diff < 0, 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_mf / (neg_mf + 1e-8))))

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-8))))
    
    df['Target_Short'] = ((df['Close'].shift(-1) - df['Close']) / df['Close'] > 0.003).astype(int)
    df['Target_Long'] = ((df['Close'].shift(-20) - df['Close']) / df['Close'] > 0.015).astype(int)
    
    features = ['Return', 'SP500_Return', 'Lag_1', 'Lag_2', 'SMA_20_Ratio', 'SMA_Trend', 
                'CMF', 'MFI', 'BB_Pos', 'RSI', 'Vol_Ratio', 'ATR_Ratio', 'ADX', 'OBV_Trend']
    
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['ATR_Ratio'] = atr14 / df['Close']

    df_short = df.dropna(subset=features + ['Target_Short'])
    df_long = df.dropna(subset=features + ['Target_Long'])
    
    if len(df_short) < 30 or len(df_long) < 30:
        return None, None, None, None, None, None

    latest_today = df[features].iloc[-1:]

    split_s = int(len(df_short) * 0.75)
    X_tr_s, X_te_s = df_short[features].iloc[:split_s], df_short[features].iloc[split_s+1:]
    y_tr_s, y_te_s = df_short['Target_Short'].iloc[:split_s], df_short['Target_Short'].iloc[split_s+1:]
    
    model_short = HistGradientBoostingClassifier(max_iter=80, max_depth=3, min_samples_leaf=15, l2_regularization=5.0, random_state=42)
    model_short.fit(X_tr_s, y_tr_s)
    acc_short = model_short.score(X_te_s, y_te_s) * 100
    
    split_l = int(len(df_long) * 0.75)
    X_tr_l, X_te_l = df_long[features].iloc[:split_l], df_long[features].iloc[split_l+20:]
    y_tr_l, y_te_l = df_long['Target_Long'].iloc[:split_l], df_long['Target_Long'].iloc[split_l+20:]
    
    model_long = HistGradientBoostingClassifier(max_iter=80, max_depth=3, min_samples_leaf=15, l2_regularization=5.0, random_state=42)
    model_long.fit(X_tr_l, y_tr_l)
    acc_long = model_long.score(X_te_l, y_te_l) * 100
    
    prob_short = model_short.predict_proba(latest_today)[0][1] * 100
    prob_long = model_long.predict_proba(latest_today)[0][1] * 100
    avg_prob = (prob_short + prob_long) / 2
    
    return df, prob_short, prob_long, avg_prob, acc_short, acc_long

all_tickers, us_stocks, israeli_stocks = get_stock_universe()

st.sidebar.title("🎮 מצבי עבודה")
app_mode = st.sidebar.radio("בחר תצוגה:", ["🔍 ניתוח מניה בודדת", "📋 סורק רשימת מעקב", "🚀 צייד הזדמנויות שוק"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ ניהול רשימת מעקב")
new_ticker_man = st.sidebar.text_input("הקלד סימול מניה להוספה (למשל AAPL):").strip().upper()

if st.sidebar.button("➕ הוסף לרשימה"):
    if new_ticker_man and new_ticker_man not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker_man)
        save_watchlist(st.session_state.watchlist)
        st.sidebar.success(f"התווספה ונשמרה: {new_ticker_man}")
        st.rerun()

remove_ticker = st.sidebar.selectbox("הסר מניה מהרשימה:", ["-- בחר --"] + st.session_state.watchlist)
if st.sidebar.button("🗑️ הסר מהרשימה") and remove_ticker != "-- בחר --":
    st.session_state.watchlist.remove(remove_ticker)
    save_watchlist(st.session_state.watchlist)
    st.sidebar.warning(f"הוסרה ונשמרה: {remove_ticker}")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📱 הגדרות בוט טלגרם")
tg_token = st.sidebar.text_input("Telegram Bot Token:", value="8979601396:AAFQjLLDf81HJPh8RjkpcpzQYxYAAHd8jpw", type="password")
tg_chat_id = st.sidebar.text_input("Telegram Chat ID:", value="5117812191")

# --- 1. מצב ניתוח מניה בודדת ---
if app_mode == "🔍 ניתוח מניה בודדת":
    st.title("🔍 ניתוח מעמיק, התראות בלייב וסימולציה היסטורית")
    target_ticker = st.text_input("הקלד סימול מניה (למשל TSLA, BTC-USD, ICL.TA):", value="SPY").strip().upper()

    if target_ticker:
        with st.spinner(f"מנתח לעומק את {target_ticker}..."):
            df, curr, actual_ticker = fetch_live_data(target_ticker)
            
        if df is None:
            st.error(f"לא נשלפו נתונים עבור '{target_ticker}'.")
        else:
            processed = process_features_and_model(df)
            if processed[0] is not None:
                df, prob_short, prob_long, avg_prob, acc_short, acc_long = processed
                
                current_price = df['Close'].iloc[-1]
                adx_val = df['ADX'].iloc[-1]
                latest_cmf = df['CMF'].iloc[-1]
                macd_h = df['MACD_Hist'].iloc[-1]
                macd_h_prev = df['MACD_Hist'].iloc[-2]
                bb_width = df['BB_Width'].iloc[-1]

                # מסדרים את התצוגה ב-6 עמודות נקיות
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("מחיר", f"{curr}{current_price:.2f}")
                col2.metric("הסתברות AI", f"{avg_prob:.1f}%")
                col3.metric("ADX (מגמה)", f"{adx_val:.1f}")
                col4.metric("CMF (מוסדיים)", f"{latest_cmf:+.2f}")
                
                if macd_h > 0 and macd_h > macd_h_prev:
                    col5.metric("MACD מומנטום", "🟢 מאיץ")
                elif macd_h > 0:
                    col5.metric("MACD מומנטום", "🟡 חיובי")
                else:
                    col5.metric("MACD מומנטום", "🔴 שלילי")
                    
                # הוספת אינדיקציית קפיץ של בולינג'ר
                if bb_width < 0.10: # אם הרצועות במרחק של פחות מ-10% זו מזו
                    col6.metric("רצועות בולינג'ר", "🔥 קפיץ דרוך (Squeeze)")
                else:
                    col6.metric("רצועות בולינג'ר", "🌊 תנועה רחבה")
                
                if avg_prob >= 54 and adx_val > 25 and macd_h > 0 and macd_h > macd_h_prev:
                    st.success("🎯 **יהלום! איתות קנייה חזק ומומנטום שמאיץ (STRONG BUY)**")
                elif avg_prob >= 48: 
                    st.warning("🟡 **המתנה / ניטרלי (HOLD)**")
                else: 
                    st.error("🔴 **מכירה / סיכון (SELL)**")

                vol_colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in df.iterrows()]
                
                fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.3, 0.15, 0.15, 0.2, 0.2],
                                    subplot_titles=("מחיר, ממוצעים ורצועות בולינג'ר", "RSI", "נפח מסחר", "ADX & CMF", "MACD (מומנטום פריצה)"))
                
                # ציור גרף המחיר והוספת רצועות הבולינג'ר
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash'), name='BB Upper', showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(150, 150, 150, 0.1)', name='BB Lower', showlegend=False), row=1, col=1)
                fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='נרות'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1.5), name='SMA 20'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
                fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
                
                fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=vol_colors, name='Volume'), row=3, col=1)
                
                cmf_colors = ['green' if val >= 0 else 'red' for val in df['CMF']]
                fig.add_trace(go.Bar(x=df['Date'], y=df['CMF'], marker_color=cmf_colors, name='CMF'), row=4, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['ADX'], line=dict(color='black', width=2), name='ADX'), row=4, col=1)
                
                macd_colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
                fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], marker_color=macd_colors, name='MACD Hist'), row=5, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], line=dict(color='blue', width=1.5), name='MACD'), row=5, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], line=dict(color='orange', width=1.5), name='Signal'), row=5, col=1)
                
                fig.update_layout(height=950, xaxis_rangeslider_visible=False, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

# --- 2. מצב רשימת מעקב ---
elif app_mode == "📋 סורק רשימת מעקב":
    st.title("📋 סורק רשימת מעקב")
    if not st.session_state.watchlist:
        st.warning("רשימת המעקב שלך ריקה.")
    else:
        results = []
        progress_bar = st.progress(0)
        for idx, ticker in enumerate(st.session_state.watchlist):
            df, curr, actual_ticker = fetch_live_data(ticker)
            if df is not None:
                processed = process_features_and_model(df)
                if processed[0] is not None:
                    df, p_short, p_long, avg_p, acc_s, acc_l = processed
                    price, adx_v, cmf_v = df['Close'].iloc[-1], df['ADX'].iloc[-1], df['CMF'].iloc[-1]
                    macd_h = df['MACD_Hist'].iloc[-1]
                    macd_h_prev = df['MACD_Hist'].iloc[-2]
                    
                    rec = "🎯 יהלום קנייה" if (avg_p >= 54 and adx_v > 25 and macd_h > 0 and macd_h > macd_h_prev) else ("🟡 המתנה" if avg_p >= 48 else "🔴 מכירה")
                    results.append({"סימול": actual_ticker, "מחיר": f"{curr}{price:.2f}", "ציון AI": f"{avg_p:.1f}%", "המלצה": rec, "ADX": f"{adx_v:.1f}", "CMF": f"{cmf_v:+.2f}"})
            progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
        progress_bar.empty()
        
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            if st.button("📲 שלח דוח לטלגרם"):
                msg = "📋 *דוח רשימת מעקב:*\n\n" + "\n".join([f"• {r['סימול']}: {r['המלצה']} (AI: {r['ציון AI']})" for r in results])
                success, res_msg = send_telegram_msg(tg_token, tg_chat_id, msg)
                st.success(res_msg) if success else st.error(res_msg)

# --- 3. צייד הזדמנויות שוק (הסורק האכזרי) ---
else:
    st.title("🚀 צייד הזדמנויות אלגוריתמי (Sniper Mode)")
    st.markdown("סריקה חסרת רחמים. המערכת תחזיר רק מניות המציגות: מגמה חזקה (ADX>25), כניסת כסף (CMF>0), הסתברות AI>54%, **ופריצת MACD מאיצה כלפי מעלה.**")
    
    options = ["הכל (סריקה מלאה של 10,000+ מניות!)"] + list(THEMATIC_TICKERS.keys())
    scan_group = st.selectbox("בחר קטגוריה לסריקה:", options)
    
    if st.button("🔎 התחל בסריקת השוק"):
        if scan_group == "הכל (סריקה מלאה של 10,000+ מניות!)":
            target_list = all_tickers
        else:
            target_list = THEMATIC_TICKERS[scan_group]
        
        st.info(f"מתחיל סריקת צלף של {len(target_list)} נכסים... מחפש רק את היהלומים. סריקה מלאה תיקח מספר שעות.")
        
        opportunities = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ticker in enumerate(target_list):
            status_text.text(f"מנתח את {ticker} ({idx+1}/{len(target_list)})...")
            
            time.sleep(0.25)
            
            df, curr, actual_ticker = fetch_live_data(ticker)
            if df is not None:
                processed = process_features_and_model(df)
                if processed[0] is not None:
                    df, p_short, p_long, avg_p, acc_s, acc_l = processed
                    adx_v, cmf_v = df['ADX'].iloc[-1], df['CMF'].iloc[-1]
                    macd_h = df['MACD_Hist'].iloc[-1]
                    macd_h_prev = df['MACD_Hist'].iloc[-2]
                    bb_width = df['BB_Width'].iloc[-1]
                    
                    if avg_p >= 54 and adx_v >= 25 and cmf_v > 0 and macd_h > 0 and macd_h > macd_h_prev:
                        price = df['Close'].iloc[-1]
                        rsi_v = df['RSI'].iloc[-1]
                        
                        # דיווח האם המניה ב-Squeeze
                        bb_status = "🔥 קפיץ דרוך" if bb_width < 0.10 else "🌊 תנועה רחבה"
                        
                        opportunities.append({
                            "סימול": actual_ticker,
                            "מחיר": f"{curr}{price:.2f}",
                            "ציון AI": f"{avg_p:.1f}%",
                            "כוח (ADX)": f"{adx_v:.1f}",
                            "כסף מוסדי (CMF)": f"{cmf_v:+.2f}",
                            "מצב בולינג'ר": bb_status,
                            "מומנטום MACD": "מאיץ חיובי 🚀",
                        })
            progress_bar.progress((idx + 1) / len(target_list))
            
        progress_bar.empty()
        status_text.empty()
        
        if opportunities:
            st.success(f"💎 הפילטר עבד! מתוך {len(target_list)} נכסים, נמצאו רק {len(opportunities)} יהלומים שמראים פריצה עכשיו.")
            res_df = pd.DataFrame(opportunities)
            st.dataframe(res_df, use_container_width=True)
            
            if st.button("📲 שלח התראות קנייה לטלגרם"):
                msg = "💎 *יהלומים זוהו בצייד ההזדמנויות:*\n\n"
                for r in opportunities:
                    bb_alert = "(בסקוויז!)" if "קפיץ" in r['מצב בולינג'ר'] else ""
                    msg += f"🔥 *{r['סימול']}* {bb_alert}\nמחיר: {r['מחיר']} | AI: {r['ציון AI']} | MACD מאיץ\n"
                success, res_msg = send_telegram_msg(tg_token, tg_chat_id, msg)
                st.success(res_msg) if success else st.error(res_msg)
        else:
            st.warning(f"הסריקה הסתיימה. הפילטר הנוקשה לא מצא אף מניה ב-{scan_group} שראויה לתואר 'יהלום' כרגע. השוק חלש היום.")
