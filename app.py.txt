import sys
import os
import json
import subprocess

# 1. מנגנון הרצה אוטומטי מתוך IDLE
IS_STREAMLIT = os.environ.get("RUNNING_IN_STREAMLIT") == "1"

if __name__ == "__main__" and not IS_STREAMLIT:
    for pkg in ["requests", "pandas", "numpy", "scikit-learn", "streamlit", "plotly"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    
    print("\n" + "="*60)
    print("   [+] מפעיל את שרת ה-AI הכל-כולל-הכל (Backtest + Telegram + Cloud)...   ")
    print("="*60)
    
    env = os.environ.copy()
    env["RUNNING_IN_STREAMLIT"] = "1"
    script_path = os.path.abspath(__file__)
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path], env=env)
    sys.exit()

# ---------------------------------------------------------
# קוד האפליקציה הרץ בדפדפן (Streamlit Dashboard):
# ---------------------------------------------------------
import urllib3
import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="AI Stock Analytics Pro - Master Edition", page_icon="📈", layout="wide")

# --- ניהול שמירה אוטומטית של רשימת מעקב ---
WATCHLIST_FILE = "watchlist.json"
DEFAULT_WATCHLIST = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'LUMI.TA', 'AMZN', 'GOOGL', 'META', 'POLI.TA', 'TEVA']

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

# --- מנגנון התראות לטלגרם ---
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
    df['SMA_20_Ratio'] = df['Close'] / df['SMA_20']
    df['SMA_Trend'] = df['SMA_20'] / df['SMA_50']
    
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)

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

    std_20 = df['Close'].rolling(20).std()
    df['BB_Pos'] = (df['Close'] - (df['SMA_20'] - 2*std_20)) / (4 * std_20 + 1e-8)
    
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
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (exp1 - exp2) / df['Close']
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['ATR_Ratio'] = atr14 / df['Close']

    df['Target_Short'] = ((df['Close'].shift(-1) - df['Close']) / df['Close'] > 0.003).astype(int)
    df['Target_Long'] = ((df['Close'].shift(-20) - df['Close']) / df['Close'] > 0.015).astype(int)
    
    features = ['Return', 'SP500_Return', 'Lag_1', 'Lag_2', 'SMA_20_Ratio', 'SMA_Trend', 
                'CMF', 'MFI', 'BB_Pos', 'RSI', 'MACD', 'Vol_Ratio', 'ATR_Ratio', 'ADX', 'OBV_Trend']
    
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

# --- מודול בדיקה לאחור (Backtesting Engine) ---
def run_backtest(df):
    df_bt = df.copy()
    # איתות קנייה: ממוצע 20 מעל 50 + CMF חיובי + ADX > 20
    df_bt['Signal'] = np.where(
        (df_bt['SMA_20'] > df_bt['SMA_50']) & (df_bt['CMF'] > 0) & (df_bt['ADX'] > 20), 1, 0
    )
    df_bt['Strategy_Return'] = df_bt['Signal'].shift(1) * df_bt['Return']
    
    cum_strategy = (1 + df_bt['Strategy_Return'].fillna(0)).cumprod() - 1
    cum_benchmark = (1 + df_bt['Return'].fillna(0)).cumprod() - 1
    
    df_bt['Cum_Strategy'] = cum_strategy * 100
    df_bt['Cum_Benchmark'] = cum_benchmark * 100
    
    total_strat = cum_strategy.iloc[-1] * 100
    total_bench = cum_benchmark.iloc[-1] * 100
    
    active_days = df_bt[df_bt['Strategy_Return'] != 0]
    win_rate = (len(active_days[active_days['Strategy_Return'] > 0]) / len(active_days)) * 100 if len(active_days) > 0 else 0
    
    return df_bt, total_strat, total_bench, win_rate

# --- סרגל צדדי ---
st.sidebar.title("🎮 מצבי עבודה")
app_mode = st.sidebar.radio("בחר תצוגה:", ["🔍 ניתוח מניה בודדת", "📋 רשימת מעקב (Watchlist)"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ ניהול רשימת מעקב")
new_ticker = st.sidebar.text_input("הוסף מניה לרשימת המעקב:").strip().upper()
if st.sidebar.button("➕ הוסף לרשימה") and new_ticker:
    if new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        save_watchlist(st.session_state.watchlist)
        st.sidebar.success(f"התווספה ונשמרה: {new_ticker}")

remove_ticker = st.sidebar.selectbox("הסר מניה מהרשימה:", ["-- בחר --"] + st.session_state.watchlist)
if st.sidebar.button("🗑️ הסר מהרשימה") and remove_ticker != "-- בחר --":
    st.session_state.watchlist.remove(remove_ticker)
    save_watchlist(st.session_state.watchlist)
    st.sidebar.warning(f"הוסרה ונשמרה: {remove_ticker}")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📱 הגדרות בוט טלגרם (אופציונלי)")
tg_token = st.sidebar.text_input("Telegram Bot Token:", type="password")
tg_chat_id = st.sidebar.text_input("Telegram Chat ID:")

# ---------------------------------------------------------
# מצב 1: ניתוח מניה בודדת
# ---------------------------------------------------------
if app_mode == "🔍 ניתוח מניה בודדת":
    st.title("🔍 ניתוח מעמיק, התראות בלייב וסימולציה היסטורית")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔎 חיפוש מניה")
    target_ticker = st.sidebar.text_input("הקלד סימול מניה (למשל AAPL, TSLA, LUMI.TA):", value="AAPL").strip().upper()

    if target_ticker:
        with st.spinner(f"מנתח לעומק את {target_ticker}..."):
            df, curr, actual_ticker = fetch_live_data(target_ticker)
            
        if df is None:
            st.error(f"לא נשלפו נתונים עבור הסימול '{target_ticker}'.")
        else:
            processed = process_features_and_model(df)
            if processed[0] is None:
                st.error("אין מספיק היסטוריית מסחר רציפה לאימון מודל AI במניה זו.")
            else:
                df, prob_short, prob_long, avg_prob, acc_short, acc_long = processed
                
                current_price = df['Close'].iloc[-1]
                latest_atr = df['ATR'].iloc[-1]
                latest_rsi = df['RSI'].iloc[-1]
                adx_val = df['ADX'].iloc[-1]
                latest_cmf = df['CMF'].iloc[-1]
                obv_trend_val = df['OBV_Trend'].iloc[-1]
                sma20 = df['SMA_20'].iloc[-1]
                sma50 = df['SMA_50'].iloc[-1]
                
                stop_loss = current_price - (2 * latest_atr)
                take_profit = current_price + (4 * latest_atr)
                risk_per_share = current_price - stop_loss

                st.subheader(f"דוח ניתוח מניה: {actual_ticker}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("מחיר נוכחי", f"{curr}{current_price:.2f}")
                
                adx_status = "מגמה חזקה 🔥" if adx_val > 25 else "דשדוש (רעש) 💤"
                col2.metric("עוצמת מגמה (ADX)", f"{adx_val:.1f}", adx_status, delta_color="off")
                
                cmf_status = "כסף נכנס 🟢" if latest_cmf > 0.05 else ("כסף יוצא 🔴" if latest_cmf < -0.05 else "מאוזן 🟡")
                col3.metric("זרימת מוסדיים (CMF)", f"{latest_cmf:+.2f}", cmf_status)
                
                obv_status = "איסוף סחורה 🟢" if obv_trend_val > 0.5 else ("פיזור סחורה 🔴" if obv_trend_val < -0.5 else "ניטרלי 🟡")
                col4.metric("נפח צבור (OBV)", f"{obv_trend_val:+.2f}", obv_status)

                # --- התראות בלייב ושליחה לטלגרם ---
                st.markdown("---")
                st.markdown("### 🔔 התראות מערכת אוטומטיות (Real-Time Alerts)")
                
                alerts = []
                if adx_val > 25 and latest_cmf > 0.05 and current_price > sma20:
                    alerts.append(("success", "🚀 **איתות מומנטום חיובי:** כסף מוסדי נכנס בעוצמה במגמה עולה ברורה."))
                if latest_rsi > 70:
                    alerts.append(("warning", "⚠️ **אזהרת קניית-יתר (RSI > 70):** המניה מתוחה מדי, סיכון גבוה לתיקון טכני בקרוב."))
                elif latest_rsi < 30:
                    alerts.append(("info", "💡 **אזהרת מכירת-יתר (RSI < 30):** המנייה במחיר מוזל קיצונית, אפשרות לזינוק חזרה."))
                if current_price < sma50:
                    alerts.append(("error", "🚨 **אזהרת מגמה ראשית:** המחיר נסחר מתחת לממוצע 50 (שליטת מוכרים)."))
                if latest_cmf < -0.05 and current_price > sma20:
                    alerts.append(("warning", "⚡ **איתות סטייה (Divergence):** המחיר עולה אך כסף מוסדי בורח החוצה - זהירות ממלכודת!"))

                if not alerts:
                    st.info("ℹ️ אין התראות חריגות כרגע. המניה נסחרת בתנאי שוק רגילים.")
                else:
                    for alert_type, msg in alerts:
                        if alert_type == "success": st.success(msg)
                        elif alert_type == "warning": st.warning(msg)
                        elif alert_type == "error": st.error(msg)
                        elif alert_type == "info": st.info(msg)

                # כפתור שליחה לטלגרם
                if st.button("📲 שלח דוח ניתוח זה לטלגרם"):
                    rec_text = "BUY" if (avg_prob >= 54 and adx_val > 20) else ("HOLD" if avg_prob >= 48 else "SELL")
                    msg_body = (
                        f"📊 *דוח AI מעודכן עבור {actual_ticker}*\n"
                        f"• מחיר: {curr}{current_price:.2f}\n"
                        f"• המלצה: *{rec_text}*\n"
                        f"• הסתברות AI: {avg_prob:.1f}%\n"
                        f"• ADX: {adx_val:.1f} | CMF: {latest_cmf:+.2f}\n"
                        f"• Stop Loss: {curr}{stop_loss:.2f}\n"
                        f"• Target: {curr}{take_profit:.2f}"
                    )
                    success, res_msg = send_telegram_msg(tg_token, tg_chat_id, msg_body)
                    if success: st.success(res_msg)
                    else: st.error(res_msg)

                st.markdown("---")

                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### 🤖 הסתברויות עתידיות (Machine Learning)")
                    st.write(f"**טווח קצר (1-5 ימים):** {prob_short:.1f}% הסתברות לעלייה *(דיוק: {acc_short:.1f}%)*")
                    st.progress(int(np.clip(prob_short, 0, 100)))
                    
                    st.write(f"**טווח רחוק (חודש קדימה):** {prob_long:.1f}% הסתברות לעלייה *(דיוק: {acc_long:.1f}%)*")
                    st.progress(int(np.clip(prob_long, 0, 100)))

                with col_right:
                    st.markdown("### 🎯 המלצת מודל סופית")
                    if avg_prob >= 54 and adx_val > 20:
                        st.success("🟢 **קנייה חזקה (STRONG BUY)**")
                        st.write(f"מומנטום כיווני חזק בשילוב {cmf_status}.")
                    elif avg_prob >= 48:
                        st.warning("🟡 **המתנה / ניטרלי (HOLD)**")
                        st.write("אינדיקטורים מאוזנים או חוסר מגמה (דשדוש). מומלץ להמתין.")
                    else:
                        st.error("🔴 **מכירה / סיכון (SELL)**")
                        if latest_cmf > 0.05:
                            st.write("⚠️ **אזהרת מלכודת קונים (Bull Trap):** למרות כניסת כסף זמנית, ה-AI מזהה סיכון גבוה לתיקון חריף משיא.")
                        else:
                            st.write("לחץ מכירות, זרימת כסף שלילית החוצה או שבירת תמיכות.")

                # --- מחשבון ניהול סיכונים ---
                st.markdown("---")
                st.markdown("### 🛡️ ניהול סיכונים ותכנון עסקה (Risk Management)")
                
                r_col1, r_col2, r_col3 = st.columns(3)
                r_col1.metric("🛑 קטיעת הפסד (Stop-Loss)", f"{curr}{stop_loss:.2f}", f"-{((current_price-stop_loss)/current_price)*100:.1f}%")
                r_col2.metric("🎯 יעד רווח (Take-Profit)", f"{curr}{take_profit:.2f}", f"+{((take_profit-current_price)/current_price)*100:.1f}%")
                r_col3.metric("⚖️ יחס סיכון / סיכוי", "1 : 2.0", "מבנה מוסדי תקין")

                st.markdown("#### 📐 מחשבון כמות מניות לקנייה:")
                c_calc1, c_calc2 = st.columns(2)
                with c_calc1:
                    portfolio_size = st.number_input("גודל התיק שלך ($ או ₪):", value=10000, step=1000)
                with c_calc2:
                    risk_pct = st.slider("אחוז סיכון מותר לעסקה זו (%):", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

                max_loss_allowed = portfolio_size * (risk_pct / 100)
                shares_to_buy = int(max_loss_allowed / risk_per_share) if risk_per_share > 0 else 0
                total_investment = shares_to_buy * current_price

                st.info(f"💡 **המלצת כמות לעסקה:** לקניית **{shares_to_buy}** מניות בסכום כולל של **{curr}{total_investment:,.2f}**.\n\n"
                        f"אם העסקה תיגע ב-Stop Loss ({curr}{stop_loss:.2f}), ההפסד המקסימלי שלך יהיה **{curr}{max_loss_allowed:,.2f}** ({risk_pct}% מהתיק).")

                # --- מודול בדיקה לאחור (Backtesting Engine Display) ---
                st.markdown("---")
                with st.expander("🧪 **לחץ כאן לצפייה בסימולציה היסטורית (Backtesting)**", expanded=False):
                    df_bt, total_strat, total_bench, win_rate = run_backtest(df)
                    
                    b_col1, b_col2, b_col3 = st.columns(3)
                    b_col1.metric("תשואת אלגוריתם ה-AI", f"{total_strat:+.1f}%")
                    b_col2.metric("תשואת קנה והחזק (Benchmark)", f"{total_bench:+.1f}%")
                    b_col3.metric("אחוז עסקאות מרוויחות", f"{win_rate:.1f}%")
                    
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['Cum_Strategy'], mode='lines', name='אסטרטגיית AI', line=dict(color='green', width=2)))
                    fig_bt.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['Cum_Benchmark'], mode='lines', name='קנה והחזק', line=dict(color='gray', dash='dash')))
                    fig_bt.update_layout(title="השוואת תשואה מצטברת ב-2 השנים האחרונות (%)", template="plotly_white", height=400)
                    st.plotly_chart(fig_bt, use_container_width=True)

                # --- גרפים טכניים ---
                st.markdown("---")
                st.markdown("### 📊 ניתוח ויזואלי מתקדם (ממוצעים, CMF, ADX)")
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, 
                                    row_heights=[0.5, 0.25, 0.25],
                                    subplot_titles=("מחיר, ממוצעים נעים וקווי Stop/Profit", "זרימת כסף (CMF)", "עוצמת מגמה (ADX)"))
                
                fig.add_trace(go.Candlestick(
                    x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                    name='נרות יפניים'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='ממוצע 20', line=dict(color='orange', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='ממוצע 50', line=dict(color='purple', width=1.5)), row=1, col=1)

                fig.add_hline(y=stop_loss, line_dash="dot", line_color="red", annotation_text="Stop Loss", row=1, col=1)
                fig.add_hline(y=take_profit, line_dash="dot", line_color="green", annotation_text="Take Profit", row=1, col=1)

                colors = ['green' if val >= 0 else 'red' for val in df['CMF']]
                fig.add_trace(go.Bar(x=df['Date'], y=df['CMF'], name='זרימת כסף (CMF)', marker_color=colors), row=2, col=1)
                
                fig.add_trace(go.Scatter(x=df['Date'], y=df['ADX'], mode='lines', name='ADX', line=dict(color='blue', width=2)), row=3, col=1)
                fig.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="סף מגמה חזקה", row=3, col=1)
                
                fig.update_layout(
                    xaxis3_title="תאריך",
                    template="plotly_white",
                    height=700,
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# מצב 2: רשימת מעקב
# ---------------------------------------------------------
else:
    st.title("📋 רשימת מעקב וסריקת מניות מרוכזת")
    st.markdown("סריקה בלייב של כל המניות ברשימה האישית שלך עם המלצות AI ומדדי מפתח.")
    
    if not st.session_state.watchlist:
        st.warning("רשימת המעקב שלך ריקה. הוסף מניות בסרגל הצד.")
    else:
        results = []
        progress_bar = st.progress(0)
        
        for idx, ticker in enumerate(st.session_state.watchlist):
            df, curr, actual_ticker = fetch_live_data(ticker)
            if df is not None:
                processed = process_features_and_model(df)
                if processed[0] is not None:
                    df, p_short, p_long, avg_p, acc_s, acc_l = processed
                    price = df['Close'].iloc[-1]
                    adx_v = df['ADX'].iloc[-1]
                    cmf_v = df['CMF'].iloc[-1]
                    obv_v = df['OBV_Trend'].iloc[-1]
                    
                    if avg_p >= 54 and adx_v > 20:
                        rec = "🟢 קנייה חזקה"
                    elif avg_p >= 48:
                        rec = "🟡 המתנה (HOLD)"
                    else:
                        rec = "🔴 מכירה / סיכון"
                        
                    trend_status = "🔥 חזקה" if adx_v > 25 else "💤 דשדוש"
                    money_status = "🟢 כניסה" if cmf_v > 0.05 else ("🔴 יציאה" if cmf_v < -0.05 else "🟡 ניטרלי")
                    
                    results.append({
                        "סימול": actual_ticker,
                        "מחיר נוכחי": f"{curr}{price:.2f}",
                        "הסתברות AI (ממוצע)": f"{avg_p:.1f}%",
                        "המלצת מודל": rec,
                        "עוצמת מגמה (ADX)": f"{adx_v:.1f} ({trend_status})",
                        "זרימת כספים (CMF)": money_status,
                        "איסוף סחורה (OBV)": "🟢 כן" if obv_v > 0.5 else "🔴 לא",
                        "דיוק היסטורי": f"{acc_s:.1f}%"
                    })
            progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
            
        progress_bar.empty()
        
        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            
            if st.button("📲 שלח את כל תוצאות הסריקה לטלגרם"):
                summary_text = "📋 *דוח סריקת רשימת מעקב בלייב:*\n\n"
                for row in results:
                    summary_text += f"• *{row['סימול']}*: {row['מחיר נוכחי']} | {row['המלצת מודל']} ({row['הסתברות AI (ממוצע)']})\n"
                success, res_msg = send_telegram_msg(tg_token, tg_chat_id, summary_text)
                if success: st.success(res_msg)
                else: st.error(res_msg)
        else:
            st.error("לא נמצאו נתונים תקינים עבור המניות ברשימת המעקב.")
