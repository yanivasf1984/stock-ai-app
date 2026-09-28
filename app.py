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
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="AI Stock Analytics Pro - Command Center", page_icon="💎", layout="wide")

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
    fallback_list = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'BRK-B', 'LLY', 'AVGO']
    
    try:
        # משיכת ~1500 מניות העילית של ארה"ב (S&P 1500) מויקיפדיה
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]['Symbol'].tolist()
        sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]['Symbol'].tolist()
        elite_1500 = list(set(sp500 + sp400 + sp600))
        
        # ניקוי סימולים (החלפת נקודה במקף עבור יאהו)
        elite_1500 = [s.replace('.', '-') for s in elite_1500]
        massive_universe = sorted(list(set(elite_1500 + all_thematic + israeli_stocks + etfs)))
        return massive_universe, elite_1500, israeli_stocks
    except Exception:
        # Fallback במקרה של שגיאת רשת
        massive_universe = sorted(list(set(fallback_list + all_thematic + israeli_stocks + etfs)))
        return massive_universe, fallback_list, israeli_stocks

WATCHLIST_FILE = "watchlist.json"
PORTFOLIO_FILE = "portfolio.json"
DEFAULT_WATCHLIST = ['SPY', 'QQQ', 'BTC-USD', 'NVDA', 'LEUMI.TA', 'TSLA']

def load_json_file(filepath, default_value):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return default_value.copy()

def save_json_file(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"שגיאה בשמירת הקובץ {filepath}: {e}")

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_json_file(WATCHLIST_FILE, DEFAULT_WATCHLIST)
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_json_file(PORTFOLIO_FILE, [])

def send_telegram_msg(bot_token, chat_id, text):
    if not bot_token or not chat_id:
        return False, "נא להגדיר Token ו-Chat ID בסרגל הצד."
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return True, "התראה נשלחה!"
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
        response = requests.get(url, headers=headers, verify=False, timeout=5)
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
    
    df['VWMA_20'] = (df['Close'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)
    df['Vol_SMA_20'] = df['Volume'].rolling(20).mean()
    
    # 🐋 אנומליות ווליום (Whales)
    df['Whale_Buy'] = (df['Volume'] > df['Vol_SMA_20'] * 2.5) & (df['Close'] > df['Open'])
    
    # 👑 עוצמה יחסית (Relative Strength)
    df['RS'] = df['Close'] / df['SP500_Close']
    df['RS_SMA_20'] = df['RS'].rolling(20).mean()
    
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)

    std_20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * std_20)
    df['BB_Lower'] = df['SMA_20'] - (2 * std_20)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']

    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # 🗜️ מנגנון הקפיץ (TTM Squeeze)
    df['KC_Upper'] = df['SMA_20'] + (1.5 * df['ATR'])
    df['KC_Lower'] = df['SMA_20'] - (1.5 * df['ATR'])
    df['Squeeze_On'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])

    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-8))))
    
    df['Target_Short'] = ((df['Close'].shift(-1) - df['Close']) / df['Close'] > 0.003).astype(int)
    
    features = ['Return', 'SP500_Return', 'Lag_1', 'Lag_2', 'CMF', 'RSI', 'ADX']
    df_short = df.dropna(subset=features + ['Target_Short'])
    
    if len(df_short) < 30:
        return None, None, None, None, None, None

    latest_today = df[features].iloc[-1:]
    split_s = int(len(df_short) * 0.75)
    X_tr_s, y_tr_s = df_short[features].iloc[:split_s], df_short['Target_Short'].iloc[:split_s]
    model_short = HistGradientBoostingClassifier(max_iter=80, max_depth=3, min_samples_leaf=15, l2_regularization=5.0, random_state=42)
    model_short.fit(X_tr_s, y_tr_s)
    
    prob_short = model_short.predict_proba(latest_today)[0][1] * 100
    
    return df, prob_short, prob_short, prob_short, None, None

all_tickers, us_stocks, israeli_stocks = get_stock_universe()

st.sidebar.title("🎮 מצבי עבודה")
app_mode = st.sidebar.radio("בחר תצוגה:", ["🔍 ניתוח מניה בודדת", "📋 סורק רשימת מעקב", "🚀 צייד הזדמנויות שוק", "💼 ניהול תיק השקעות"])

st.sidebar.markdown("---")
st.sidebar.header("⏰ טייס אוטומטי")
auto_refresh_enabled = st.sidebar.checkbox("הפעל רענון ברקע", value=False)
if auto_refresh_enabled:
    refresh_interval = st.sidebar.slider("דקות:", 1, 30, 5)
    if app_mode != "🚀 צייד הזדמנויות שוק":
        st_autorefresh(interval=refresh_interval * 60 * 1000, key="auto_refresh_timer")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ ניהול רשימת מעקב")
new_ticker_man = st.sidebar.text_input("הקלד סימול:").strip().upper()
if st.sidebar.button("➕ הוסף לרשימה") and new_ticker_man:
    if new_ticker_man not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker_man)
        save_json_file(WATCHLIST_FILE, st.session_state.watchlist)
        st.rerun()

remove_ticker = st.sidebar.selectbox("הסר מניה:", ["-- בחר --"] + st.session_state.watchlist)
if st.sidebar.button("🗑️ הסר מהרשימה") and remove_ticker != "-- בחר --":
    st.session_state.watchlist.remove(remove_ticker)
    save_json_file(WATCHLIST_FILE, st.session_state.watchlist)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📱 טלגרם")
tg_token = st.sidebar.text_input("Token:", value="8979601396:AAFQjLLDf81HJPh8RjkpcpzQYxYAAHd8jpw", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID:", value="5117812191")

if app_mode == "🔍 ניתוח מניה בודדת":
    st.title("🔍 ניתוח צלף מקצועי")
    target_ticker = st.text_input("הקלד סימול מניה:", value="SPY").strip().upper()
    if target_ticker:
        with st.spinner(f"מנתח {target_ticker}..."):
            df, curr, actual_ticker = fetch_live_data(target_ticker)
            if df is not None:
                processed = process_features_and_model(df)
                if processed[0] is not None:
                    df, prob, _, avg_prob, _, _ = processed
                    st.metric("מחיר", f"{curr}{df['Close'].iloc[-1]:.2f}")
                    st.success("ניתוח הושלם. בגרסה זו התצוגה הגרפית צומצמה כדי לפנות מקום לצייד המורחב.")

elif app_mode == "📋 סורק רשימת מעקב":
    st.title("📋 סורק רשימת מעקב")
    if not st.session_state.watchlist: st.warning("הרשימה ריקה.")
    else:
        results = []
        for ticker in st.session_state.watchlist:
            df, curr, actual_ticker = fetch_live_data(ticker)
            if df is not None:
                processed = process_features_and_model(df)
                if processed[0] is not None:
                    df = processed[0]
                    results.append({"סימול": actual_ticker, "מחיר": df['Close'].iloc[-1]})
        if results:
            st.dataframe(pd.DataFrame(results))

elif app_mode == "🚀 צייד הזדמנויות שוק":
    st.title("🚀 צייד הלווייתנים (עם מסנן ה-VIP)")
    st.markdown("סורק חכם שפוסל מיד מניות זבל (מתחת ל-5$ או מחזור קטן ממיליון) ובוחן רק חברות עילית עם מומנטום!")
    scan_group = st.selectbox("בחר קטגוריה לסריקה:", ["הכל (~1,500 מניות S&P 1500 העילית)"] + list(THEMATIC_TICKERS.keys()))
    
    if st.button("🔎 התחל בסריקת צלף חכמה"):
        target_list = all_tickers if scan_group == "הכל (~1,500 מניות S&P 1500 העילית)" else THEMATIC_TICKERS[scan_group]
        opportunities = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ticker in enumerate(target_list):
            status_text.text(f"בודק בראדאר את {ticker} ({idx+1}/{len(target_list)})...")
            
            # בדיקה מהירה מאוד לפני הפעלת ה-AI הכבד
            df, curr, actual_ticker = fetch_live_data(ticker)
            if df is not None and not df.empty:
                current_price = df['Close'].iloc[-1]
                avg_volume = df['Volume'].tail(20).mean()
                
                # 🛑 מסנן הרדאר החכם (The VIP Filter) 🛑
                # מדלגים מיידית על מניות מתחת ל-5$ או עם מחזור קטן מ-1,000,000 (חוסך המון משאבים!)
                if current_price >= 5.0 and avg_volume >= 1000000:
                    status_text.text(f"🎯 {ticker} עברה את מסנן ה-VIP! מפעיל AI...")
                    processed = process_features_and_model(df)
                    
                    if processed[0] is not None:
                        df = processed[0]
                        avg_p = processed[3]
                        
                        if avg_p >= 54 and df['ADX'].iloc[-1] >= 25 and df['CMF'].iloc[-1] > 0 and df['MACD_Hist'].iloc[-1] > 0 and current_price > df['VWMA_20'].iloc[-1]:
                            whale = "🐋 כן!" if df['Whale_Buy'].iloc[-1] else "לא"
                            sqz = "🗜️ דחוס" if df['Squeeze_On'].iloc[-1] else "משוחרר"
                            opportunities.append({
                                "סימול": actual_ticker, "מחיר": f"{curr}{current_price:.2f}", 
                                "AI": f"{avg_p:.1f}%", "לווייתן": whale, "קפיץ": sqz
                            })
            
            # מניעת חסימת שרת נוספת
            time.sleep(0.1)
            progress_bar.progress((idx + 1) / len(target_list))
            
        progress_bar.empty(); status_text.empty()
        
        if opportunities:
            st.success(f"💎 הפילטר עבד! נמצאו {len(opportunities)} יהלומים מתוך מניות העילית.")
            st.dataframe(pd.DataFrame(opportunities), use_container_width=True)
            if st.button("📲 שלח התראות קנייה לטלגרם"):
                msg = "💎 *יהלומים זוהו בצייד ה-VIP:*\n\n" + "\n".join([f"🔥 *{r['סימול']}* | לווייתן: {r['לווייתן']}" for r in opportunities])
                send_telegram_msg(tg_token, tg_chat_id, msg)
        else:
            st.warning("לא נמצאו יהלומים שעומדים בכל הקריטריונים הנוקשים כרגע.")

elif app_mode == "💼 ניהול תיק השקעות":
    st.title("💼 תחנת פיקוד: תיק השקעות")
    if not st.session_state.portfolio: st.info("התיק שלך ריק כרגע.")
    else:
        st.write("מנהל תיק פעיל - רענן לשונית או המתן לרענון האוטומטי.")
