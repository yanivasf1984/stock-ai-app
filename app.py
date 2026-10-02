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
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import random
import concurrent.futures

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# חייב להיות הפקודה הראשונה של Streamlit
st.set_page_config(page_title="AI Stock Analytics Pro - BLINK Edition", page_icon="💎", layout="wide")

THEMATIC_TICKERS = {
    "טכנולוגיה": ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'ACN', 'CSCO', 'INTC', 'QCOM', 'IBM'],
    "קריפטו (תעודות סל)": ['COIN', 'MSTR', 'MARA', 'RIOT', 'CLSK', 'HUT', 'BITF'],
    "ביטקויין (תעודות סל)": ['IBIT', 'FBTC', 'ARKB', 'BITB', 'BITO'],
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
    
    try:
        sp500_df = pd.read_csv('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
        blink_safe_tickers = sp500_df['Symbol'].tolist()
        blink_safe_tickers = [str(s).replace('.', '-') for s in blink_safe_tickers]
        massive_universe = sorted(list(set(blink_safe_tickers + all_thematic + israeli_stocks + etfs)))
        
        if 'BTC-USD' in massive_universe: massive_universe.remove('BTC-USD')
            
        return massive_universe, blink_safe_tickers, israeli_stocks
    except Exception:
        pass

    fallback_list = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA']
    massive_universe = sorted(list(set(fallback_list + all_thematic + israeli_stocks + etfs)))
    if 'BTC-USD' in massive_universe: massive_universe.remove('BTC-USD')
    return massive_universe, fallback_list, israeli_stocks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_FILE = os.path.join(BASE_DIR, "watchlist.json")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")
DEFAULT_WATCHLIST = ['SPY', 'QQQ', 'IBIT', 'NVDA', 'LEUMI.TA', 'TSLA']

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
        st.error(f"שגיאה בשמירת {filepath}: {e}")

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_json_file(WATCHLIST_FILE, DEFAULT_WATCHLIST)
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_json_file(PORTFOLIO_FILE, [])

def send_telegram_msg(bot_token, chat_id, text):
    if not bot_token or not chat_id:
        return False, "נא להגדיר Token ו-Chat ID"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=3)
        if res.status_code == 200: return True, "התראה נשלחה!"
        return False, f"שגיאה מהשרת: {res.text}"
    except Exception as e:
        return False, str(e)

def fetch_yahoo_chart(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=4)
        if response.status_code == 200:
            return response.json().get('chart', {}).get('result')
    except Exception:
        pass
    return None

# מונע כפילויות בפניות ליאהו - שומר את נתוני ה-S&P 500 בזיכרון (חסכון של 50% מהבקשות!)
@st.cache_data(ttl=1800)
def get_sp500_df():
    res = fetch_yahoo_chart('^GSPC')
    if res and 'timestamp' in res[0]:
        df_sp = pd.DataFrame({
            'Date': pd.to_datetime(res[0]['timestamp'], unit='s'),
            'SP500_Close': res[0]['indicators']['quote'][0]['close']
        }).dropna()
        return df_sp
    return None

@st.cache_data(ttl=1800)
def fetch_live_data(raw_ticker):
    ticker = raw_ticker.strip().upper()
    is_israeli = ticker in THEMATIC_TICKERS["קרנות ישראליות"] or ticker.endswith('.TA')
    if is_israeli:
        tickers_to_try = [ticker] if ticker.endswith('.TA') else [f"{ticker}.TA"]
    else:
        tickers_to_try = [ticker, ticker.replace('.', '-')]

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
        'Open': quote.get('open'), 'Close': quote.get('close'),
        'High': quote.get('high'), 'Low': quote.get('low'), 'Volume': quote.get('volume')
    }).dropna()
    
    if len(df) < 60: return None, None, None

    # שולפים מהזיכרון במקום ליצור פנייה חדשה ליאהו
    df_sp = get_sp500_df()
    if df_sp is not None:
        df = pd.merge(df, df_sp, on='Date', how='left').ffill()
    else:
        df['SP500_Close'] = df['Close']

    if currency_code in ['ILA', 'ILS']:
        if currency_code == 'ILA':
            df[['Open', 'Close', 'High', 'Low']] /= 100
        curr = "₪"
    else: curr = "$"
        
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
    
    df['Whale_Buy'] = (df['Volume'] > df['Vol_SMA_20'] * 2.5) & (df['Close'] > df['Open'])
    df['RS'] = df['Close'] / df['SP500_Close']
    df['RS_SMA_20'] = df['RS'].rolling(20).mean()
    
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
    df['CMF'] = (mf_multiplier * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)

    std_20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * std_20)
    df['BB_Lower'] = df['SMA_20'] - (2 * std_20)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']

    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    tr = pd.DataFrame({'tr1': df['High'] - df['Low'], 'tr2': abs(df['High'] - df['Close'].shift(1)), 'tr3': abs(df['Low'] - df['Close'].shift(1))}).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()
    
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
    df['ADX'] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)).ewm(alpha=1/14, adjust=False).mean()

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
    df['OBV_Trend'] = (df['OBV'] - df['OBV_EMA']) / (df['Volume'].rolling(20).mean() + 1e-8)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-8))))
    
    df['Target_Short'] = ((df['Close'].shift(-1) - df['Close']) / df['Close'] > 0.003).astype(int)
    df['Target_Long'] = ((df['Close'].shift(-20) - df['Close']) / df['Close'] > 0.015).astype(int)
    
    features = ['Return', 'SP500_Return', 'Lag_1', 'Lag_2', 'CMF', 'RSI', 'ADX', 'OBV_Trend']
    df_short = df.dropna(subset=features + ['Target_Short'])
    df_long = df.dropna(subset=features + ['Target_Long'])
    
    if len(df_short) < 30 or len(df_long) < 30:
        return None, None, None, None, None, None

    latest_today = df[features].iloc[-1:]

    split_s = int(len(df_short) * 0.75)
    model_short = HistGradientBoostingClassifier(max_iter=80, max_depth=3, min_samples_leaf=15, l2_regularization=5.0, random_state=42)
    model_short.fit(df_short[features].iloc[:split_s], df_short['Target_Short'].iloc[:split_s])
    
    split_l = int(len(df_long) * 0.75)
    model_long = HistGradientBoostingClassifier(max_iter=80, max_depth=3, min_samples_leaf=15, l2_regularization=5.0, random_state=42)
    model_long.fit(df_long[features].iloc[:split_l], df_long['Target_Long'].iloc[:split_l])
    
    prob_short = model_short.predict_proba(latest_today)[0][1] * 100
    prob_long = model_long.predict_proba(latest_today)[0][1] * 100
    
    return df, prob_short, prob_long, (prob_short + prob_long) / 2, None, None

all_tickers, us_stocks, israeli_stocks = get_stock_universe()

st.sidebar.title("🎮 מצבי עבודה")
app_mode = st.sidebar.radio("בחר תצוגה:", ["🔍 ניתוח מניה בודדת", "📋 סורק רשימת מעקב", "🚀 צייד הזדמנויות שוק", "💼 ניהול תיק השקעות"])

st.sidebar.markdown("---")
st.sidebar.header("⏰ טייס אוטומטי")
auto_refresh_enabled = st.sidebar.checkbox("הפעל רענון ברקע", value=False)
if auto_refresh_enabled:
    refresh_interval = st.sidebar.slider("דקות:", 1, 30, 5)
    if app_mode == "🚀 צייד הזדמנויות שוק":
        st.sidebar.warning("⏸️ מושהה זמנית במסך סריקה.")
    else:
        st_autorefresh(interval=refresh_interval * 60 * 1000, key="auto_refresh_timer")
        st.sidebar.success(f"✅ פועל כל {refresh_interval} דקות.")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ ניהול רשימת מעקב")
new_ticker_man = st.sidebar.text_input("הקלד סימול:").strip().upper()
if st.sidebar.button("➕ הוסף לרשימה") and new_ticker_man:
    if new_ticker_man not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker_man)
        save_json_file(WATCHLIST_FILE, st.session_state.watchlist)
        st.rerun()

remove_ticker = st.sidebar.selectbox("הסר מניה:", ["-- בחר --"] + st.session_state.watchlist)
if st.sidebar.button("🗑️ הסר") and remove_ticker != "-- בחר --":
    st.session_state.watchlist.remove(remove_ticker)
    save_json_file(WATCHLIST_FILE, st.session_state.watchlist)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📱 טלגרם")
tg_token = st.sidebar.text_input("Token:", value="8979601396:AAFQjLLDf81HJPh8RjkpcpzQYxYAAHd8jpw", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID:", value="5117812191")

if app_mode == "🔍 ניתוח מניה בודדת":
    st.title("🔍 ניתוח צלף מקצועי (כולל לווייתנים וקפיץ)")
    target_ticker = st.text_input("הקלד סימול מניה (למשל TSLA, IBIT):", value="SPY").strip().upper()

    if target_ticker:
        with st.spinner(f"מנתח נתוני שוק עבור {target_ticker}..."):
            df, curr, actual_ticker = fetch_live_data(target_ticker)
            
        if df is None:
            st.error(f"לא נשלפו נתונים עבור '{target_ticker}'.")
        else:
            processed = process_features_and_model(df)
            if processed[0] is not None:
                df, _, _, avg_prob, _, _ = processed
                
                current_price = df['Close'].iloc[-1]
                adx_val = df['ADX'].iloc[-1]
                macd_h = df['MACD_Hist'].iloc[-1]
                macd_h_prev = df['MACD_Hist'].iloc[-2]
                current_vwma = df['VWMA_20'].iloc[-1]
                is_squeeze = df['Squeeze_On'].iloc[-1]
                is_whale = df['Whale_Buy'].iloc[-1]
                rs_current = df['RS'].iloc[-1]
                rs_sma = df['RS_SMA_20'].iloc[-1]

                st.markdown("### דשבורד כוח מוסדי")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("מחיר", f"{curr}{current_price:.2f}")
                col2.metric("הסתברות פריצה (AI)", f"{avg_prob:.1f}%")
                col3.metric("ADX (מגמה)", f"{adx_val:.1f}")
                col4.metric("VWMA (מוסדיים)", "🟢 בפנים" if current_price > current_vwma else "🔴 בחוץ")
                
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("MACD מומנטום", "🟢 מאיץ" if (macd_h > 0 and macd_h > macd_h_prev) else ("🟡 מתהפך" if macd_h > 0 else "🔴 חלש"))
                col6.metric("קפיץ (TTM Squeeze)", "🗜️ דחוס (ממתין)" if is_squeeze else "💨 משוחרר")
                col7.metric("פעילות לווייתנים", "🐋 קנייה ענקית זוהתה!" if is_whale else "רגיל")
                col8.metric("עוצמה יחסית (RS)", "👑 חזקה מהשוק" if rs_current > rs_sma else "🐢 חלשה מהשוק")

                if avg_prob >= 52 and adx_val >= 20 and macd_h > 0 and current_price > current_vwma:
                    alert_text = "🎯 **יהלום! איתות קנייה חזק + מומנטום + תמיכת קונים (STRONG BUY)**"
                    if not is_squeeze: alert_text += " [הקפיץ השתחרר!]"
                    st.success(alert_text)
                elif avg_prob >= 48: 
                    st.warning("🟡 **המתנה / ניטרלי (HOLD)**")
                else: 
                    st.error("🔴 **מכירה / סיכון (SELL)**")

                fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                                    row_heights=[0.3, 0.1, 0.15, 0.15, 0.15, 0.15],
                                    subplot_titles=("מחיר, בולינג'ר ו-VWMA מוסדי", "RSI", "נפח מסחר (זהב = כניסת לווייתן 🐋)", 
                                                    "ADX & CMF (כסף חכם)", "MACD (נקודות = TTM Squeeze)", "עוצמה יחסית (Relative Strength vs SP500)"))
                
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash'), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(150, 150, 150, 0.1)', showlegend=False), row=1, col=1)
                fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='נרות'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1.5), name='SMA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['VWMA_20'], line=dict(color='magenta', width=2, dash='dot'), name='VWMA'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
                fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
                
                vol_colors = ['gold' if row['Whale_Buy'] else ('green' if row['Close'] >= row['Open'] else 'red') for _, row in df.iterrows()]
                fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=vol_colors, name='Volume'), row=3, col=1)
                
                cmf_colors = ['green' if val >= 0 else 'red' for val in df['CMF']]
                fig.add_trace(go.Bar(x=df['Date'], y=df['CMF'], marker_color=cmf_colors, name='CMF'), row=4, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['ADX'], line=dict(color='black', width=2), name='ADX'), row=4, col=1)
                
                macd_colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
                fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], marker_color=macd_colors, name='MACD Hist'), row=5, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], line=dict(color='blue', width=1.5), name='MACD'), row=5, col=1)
                squeeze_colors = ['red' if sq else 'green' for sq in df['Squeeze_On']]
                fig.add_trace(go.Scatter(x=df['Date'], y=[0]*len(df), mode='markers', marker=dict(color=squeeze_colors, size=6, symbol='circle'), name='Squeeze'), row=5, col=1)
                
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RS'], line=dict(color='royalblue', width=2), name='RS Line'), row=6, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RS_SMA_20'], line=dict(color='gray', width=1.5, dash='dot'), name='RS SMA'), row=6, col=1)
                
                fig.update_layout(height=1100, xaxis_rangeslider_visible=False, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("💼 הוסף לתיק ההשקעות האישי")
                with st.form("add_to_portfolio_form"):
                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                    entry_date = col_f1.date_input("תאריך כניסה", datetime.today())
                    qty = col_f2.number_input("כמות מניות", min_value=0.01, value=10.0, step=1.0)
                    tp = col_f3.number_input("יעד רווח (TP)", min_value=0.0, value=current_price*1.1, step=0.5)
                    sl = col_f4.number_input("קטיעת הפסד (SL)", min_value=0.0, value=current_price*0.9, step=0.5)
                    
                    if st.form_submit_button("➕ תעד עסקה בתיק"):
                        st.session_state.portfolio.append({
                            "ticker": actual_ticker, "entry_date": entry_date.strftime("%Y-%m-%d"),
                            "entry_price": current_price, "qty": qty, "tp": tp, "sl": sl, "alert_sent": None 
                        })
                        save_json_file(PORTFOLIO_FILE, st.session_state.portfolio)
                        st.success(f"העסקה נשמרה בתיק! 💼")

elif app_mode == "📋 סורק רשימת מעקב":
    st.title("📋 סורק רשימת מעקב")
    if not st.session_state.watchlist: st.warning("רשימת המעקב שלך ריקה.")
    else:
        results = []
        progress_bar = st.progress(0)
        for idx, ticker in enumerate(st.session_state.watchlist):
            df, curr, actual_ticker = fetch_live_data(ticker)
            if df is not None:
                processed = process_features_and_model(df)
                if processed[0] is not None:
                    df = processed[0]
                    avg_p = processed[3]
                    price = df['Close'].iloc[-1]
                    
                    whale = "🐋" if df['Whale_Buy'].iloc[-1] else ""
                    sqz = "🗜️" if df['Squeeze_On'].iloc[-1] else ""
                    rs_icon = "👑" if df['RS'].iloc[-1] > df['RS_SMA_20'].iloc[-1] else ""
                    
                    rec = "🎯 יהלום" if (avg_p >= 52 and df['ADX'].iloc[-1] >= 20 and df['MACD_Hist'].iloc[-1] > 0 and price > df['VWMA_20'].iloc[-1]) else ("🟡 המתנה" if avg_p >= 48 else "🔴 מכירה")
                    results.append({"סימול": actual_ticker, "מחיר": f"{curr}{price:.2f}", "המלצה": rec, "AI": f"{avg_p:.1f}%", "אינדיקטורים": f"{whale} {sqz} {rs_icon}"})
            progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
        progress_bar.empty()
        
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)

elif app_mode == "🚀 צייד הזדמנויות שוק":
    st.title("🚀 צייד הזדמנויות אלגוריתמי (Sniper Mode - BLINK Edition)")
    st.markdown("סורק מניות מובילות בלבד התואמות לפלטפורמת BLINK. **עבר אופטימיזציה לחסכון בבקשות API ומהירות שיא!**")
    
    if 'last_scan_results' not in st.session_state:
        st.session_state.last_scan_results = None

    scan_group = st.selectbox("בחר קטגוריה לסריקה:", [f"הכל (רדאר מסתובב: סורק את כל {len(all_tickers)} המניות התואמות)"] + list(THEMATIC_TICKERS.keys()))
    
    def process_single_ticker(ticker):
        try:
            df, curr, actual_ticker = fetch_live_data(ticker)
            if df is not None:
                processed = process_features_and_model(df)
                if processed[0] is not None:
                    df = processed[0]
                    avg_p = processed[3]
                    price = df['Close'].iloc[-1]
                    
                    if avg_p >= 52 and df['ADX'].iloc[-1] >= 20 and df['MACD_Hist'].iloc[-1] > 0 and price > df['VWMA_20'].iloc[-1]:
                        whale = "🐋 כן!" if df['Whale_Buy'].iloc[-1] else "לא"
                        sqz = "🗜️ דחוס" if df['Squeeze_On'].iloc[-1] else "משוחרר"
                        rs_status = "👑 חזקה" if df['RS'].iloc[-1] > df['RS_SMA_20'].iloc[-1] else "חלשה"
                        
                        return {
                            "סימול": actual_ticker, "מחיר": f"{curr}{price:.2f}", "AI": f"{avg_p:.1f}%", 
                            "לווייתן": whale, "קפיץ": sqz, "עוצמה RS": rs_status
                        }
        except Exception:
            pass
        return None

    if st.button("🔎 התחל בסריקת צלף מקבילית"):
        # קוראים קודם ל-SP500 פעם אחת מראש, כדי שכל שאר המניות לא יתקעו!
        with st.spinner("טוען מדדי בסיס (S&P 500)..."):
            get_sp500_df()
            
        if "רדאר מסתובב" in scan_group:
            target_list = all_tickers
            st.info(f"⚡ סורק כעת במקביל את כל {len(target_list)} המניות (מותאם BLINK)...")
        else:
            target_list = THEMATIC_TICKERS[scan_group]
            
        opportunities = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        completed = 0
        # מנוע מקבילי של 20 סורקים!
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_ticker = {executor.submit(process_single_ticker, ticker): ticker for ticker in target_list}
            for future in concurrent.futures.as_completed(future_to_ticker):
                completed += 1
                ticker = future_to_ticker[future]
                
                if completed % 10 == 0 or completed == len(target_list):
                    status_text.text(f"⚡ מנתח במקביל... השלים {completed}/{len(target_list)} מניות (אחרון: {ticker})")
                    progress_bar.progress(completed / len(target_list))
                
                res = future.result()
                if res:
                    opportunities.append(res)
            
        progress_bar.empty(); status_text.empty()
        st.session_state.last_scan_results = opportunities

    if st.session_state.last_scan_results is not None:
        if len(st.session_state.last_scan_results) > 0:
            st.success(f"💎 נמצאו {len(st.session_state.last_scan_results)} יהלומים התואמים ל-BLINK.")
            st.dataframe(pd.DataFrame(st.session_state.last_scan_results), use_container_width=True)
            if st.button("📲 שלח התראות קנייה לטלגרם"):
                msg = "💎 *יהלומים זוהו בצייד (מוסדיים בפנים):*\n\n" + "\n".join([f"🔥 *{r['סימול']}* | קפיץ: {r['קפיץ']} | לווייתן: {r['לווייתן']}" for r in st.session_state.last_scan_results])
                send_telegram_msg(tg_token, tg_chat_id, msg)
        else:
            st.warning("לא נמצאו יהלומים בסריקה האחרונה. המתן להזדמנויות חדשות!")

elif app_mode == "💼 ניהול תיק השקעות":
    st.title("💼 תחנת פיקוד: תיק השקעות וניטור בריאות")
    if not st.session_state.portfolio: st.info("התיק שלך ריק כרגע.")
    else:
        portfolio_data = []
        total_invested_usd = 0.0
        total_current_usd = 0.0
        portfolio_updated = False 
        
        progress_bar = st.progress(0)
        for idx, trade in enumerate(st.session_state.portfolio):
            ticker = trade['ticker']
            df, curr, actual_ticker = fetch_live_data(ticker)
            if df is not None:
                processed = process_features_and_model(df)
                if processed[0] is not None:
                    df = processed[0]
                    price = df['Close'].iloc[-1]
                    invested = trade['entry_price'] * trade['qty']
                    current_val = price * trade['qty']
                    
                    usd_invested = invested if curr == "$" else invested / 3.7
                    usd_current = current_val if curr == "$" else current_val / 3.7
                    total_invested_usd += usd_invested
                    total_current_usd += usd_current
                    
                    health = "🟢 תקין" if price > df['VWMA_20'].iloc[-1] else "🚨 שבר תמיכה"
                    alert = "⏳ פתוח"
                    
                    if price >= trade['tp']:
                        alert = "🎯 יעד הושג!"
                        if trade.get('alert_sent') != 'tp': 
                            send_telegram_msg(tg_token, tg_chat_id, f"🎯 *Take Profit!*\nהמניה *{actual_ticker}* הגיעה ליעד: {curr}{price:.2f}")
                            trade['alert_sent'] = 'tp'
                            portfolio_updated = True
                    elif price <= trade['sl']:
                        alert = "🛑 חתוך!"
                        if trade.get('alert_sent') != 'sl': 
                            send_telegram_msg(tg_token, tg_chat_id, f"🛑 *Stop Loss!*\nהמניה *{actual_ticker}* שברה סטופ לוס: {curr}{price:.2f}")
                            trade['alert_sent'] = 'sl'
                            portfolio_updated = True
                    
                    portfolio_data.append({
                        "סימול": actual_ticker, "כמות": trade['qty'], "כניסה": f"{curr}{trade['entry_price']:.2f}",
                        "נוכחי": f"{curr}{price:.2f}", "תשואה (%)": f"{((price / trade['entry_price']) - 1) * 100:+.2f}%",
                        "סטטוס מוסדי": health, "התראה": alert
                    })
            progress_bar.progress((idx + 1) / len(st.session_state.portfolio))
        
        progress_bar.empty()
        if portfolio_updated: save_json_file(PORTFOLIO_FILE, st.session_state.portfolio)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("סה\"כ השקעה ($)", f"${total_invested_usd:,.2f}")
        c2.metric("שווי נוכחי ($)", f"${total_current_usd:,.2f}")
        c3.metric("P&L ($)", f"${(total_current_usd - total_invested_usd):,.2f}")
        
        if portfolio_data:
            st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True)
            trade_to_remove = st.selectbox("בחר סימול למחיקה:", ["-- בחר --"] + [t['ticker'] for t in st.session_state.portfolio])
            if st.button("🗑️ סגור עסקה") and trade_to_remove != "-- בחר --":
                st.session_state.portfolio = [t for t in st.session_state.portfolio if t['ticker'] != trade_to_remove]
                save_json_file(PORTFOLIO_FILE, st.session_state.portfolio)
                st.rerun()
