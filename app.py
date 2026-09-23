import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import requests

# הגדרות תצוגה
st.set_page_config(page_title="AI Quant Fund", layout="wide", initial_sidebar_state="expanded")

# פונקציה חכמה שמושכת את כל 500 מניות מדד ה-S&P 500 לייצור רשימת בחירה
@st.cache_data
def get_stock_universe():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)[0]
        tickers = table['Symbol'].tolist()
        tickers.extend(['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']) # תוספת תעודות סל
        return sorted(list(set(tickers)))
    except:
        return sorted(['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL'])

def send_telegram_message(token, chat_id, text):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text})
    except Exception as e:
        pass

def main():
    # טעינת מאגר המניות
    all_tickers = get_stock_universe()
    default_index = all_tickers.index('SPY') if 'SPY' in all_tickers else 0

    # הגדרת זיכרון לרשימת המעקב האישית
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['SPY', 'QQQ', 'NVDA', 'TSLA']

    # --- סרגל צד: ניהול רשימת מעקב (Watchlist) ---
    st.sidebar.header("📋 רשימת מעקב אישית")
    
    new_ticker = st.sidebar.selectbox("חפש מניה להוספה:", all_tickers)
    if st.sidebar.button("➕ הוסף לרשימה"):
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            st.sidebar.success(f"{new_ticker} נוספה למעקב!")
    
    st.sidebar.markdown("---")
    st.sidebar.write("**המניות שלך:**")
    for t in st.session_state.watchlist:
        st.sidebar.markdown(f"🔹 **{t}**")
        
    if st.sidebar.button("🗑️ נקה רשימה"):
        st.session_state.watchlist = []
        st.rerun()

    # --- תחילת המסך הראשי ---
    st.title("🤖 AI Quant Fund - פלטפורמת מסחר וסריקה")
    
    tab_trade, tab_backtest, tab_settings = st.tabs(["📊 מסוף מסחר ואינדיקטורים", "🔄 סימולציית אסטרטגיות (Backtest)", "⚙️ סורק אוטומטי וטלגרם"])
    
    with tab_trade:
        col1, col2 = st.columns([4, 1])
        with col2:
            # התיבה הזו עכשיו כוללת השלמה אוטומטית מתוך מאגר ה-S&P 500
            ticker = st.selectbox("חפש ובחר מניה לניתוח:", all_tickers, index=default_index)
            period = st.selectbox("תקופת זמן:", ["3mo", "6mo", "1y", "2y", "5y"], index=1)
            
        if ticker:
            df = yf.Ticker(ticker).history(period=period)
            if not df.empty:
                df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
                df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
                df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
                df['Vol_SMA_50'] = df['Volume'].rolling(window=50).mean()
                colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in df.iterrows()]
                
                with col1:
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                        vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
                    
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                                 low=df['Low'], close=df['Close'], name='מחיר'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='מחיר SMA 20'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='מחיר SMA 50'), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
                    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
                    
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='זרימת כסף'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Vol_SMA_20'], line=dict(color='blue', width=1), name='כסף SMA 20'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Vol_SMA_50'], line=dict(color='orange', width=1), name='כסף SMA 50'), row=3, col=1)
                    
                    fig.update_layout(height=750, xaxis_rangeslider_visible=False, margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 🧠 מודל החלטות")
                    current_rsi = df['RSI'].iloc[-1]
                    current_vol = df['Volume'].iloc[-1]
                    vol_sma20 = df['Vol_SMA_20'].iloc[-1]
                    
                    money_trend = "🟢 כסף נכנס גבוה מהממוצע" if current_vol > vol_sma20 else "🔴 כסף נכנס נמוך מהממוצע"
                    
                    if current_rsi < 30:
                        st.success("🟢 איתות קנייה (Oversold)")
                    elif current_rsi > 70:
                        st.error("🔴 איתות מכירה (Overbought)")
                    else:
                        st.warning("🟡 המתנה (נטרלי)")
                        
                    st.metric("RSI נוכחי", f"{current_rsi:.1f}")
                    st.metric("מחיר סגירה אחרון", f"${df['Close'].iloc[-1]:.2f}")
                    st.write(money_trend)

    with tab_backtest:
        st.subheader("מנוע בדיקת אסטרטגיה היסטורית (Backtest)")
        if 'df' in locals() and not df.empty:
            if st.button("▶️ הרץ סימולציה על נתוני העבר"):
                initial_capital = 10000 
                capital = initial_capital
                position = 0
                
                for i in range(1, len(df)):
                    if df['RSI'].iloc[i-1] < 30 and position == 0:  
                        position = capital / df['Close'].iloc[i]
                        capital = 0
                    elif df['RSI'].iloc[i-1] > 70 and position > 0: 
                        capital = position * df['Close'].iloc[i]
                        position = 0
                
                final_value = capital if position == 0 else position * df['Close'].iloc[-1]
                profit_pct = ((final_value - initial_capital) / initial_capital) * 100
                buy_hold_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("רווח האלגוריתם", f"{profit_pct:.2f}%")
                col_b2.metric("רווח קנייה והחזקה (פסיבי)", f"{buy_hold_pct:.2f}%")
                col_b3.metric("הון סופי מקופת $10,000", f"${final_value:,.2f}")

    with tab_settings:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("📱 חיבור שרת טלגרם")
            tg_token = st.text_input("Telegram Bot Token:", value="8979601396:AAFQjLLDf81HJPh8RjkpcpzQYxYAAHd8jpw", type="password")
            tg_chat_id = st.text_input("Telegram Chat ID:", value="5117812191")
        
        with col_s2:
            st.subheader("🔎 סורק הזדמנויות לפי רשימת מעקב")
            # התיבה הזו מאפשרת לבחור כמה מניות שרוצים מתוך המאגר, וכברירת מחדל מושכת את רשימת המעקב שלך
            tickers_to_scan = st.multiselect("בחר מניות לסריקה כעת:", all_tickers, default=st.session_state.watchlist)
            
            if st.button("🚀 הפעל סריקת שוק עכשיו"):
                found_signals = []
                with st.spinner("סורק את השוק ומנתח אינדיקטורים..."):
                    for t in tickers_to_scan:
                        data = yf.Ticker(t).history(period="1mo")
                        if not data.empty and len(data) > 15:
                            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
                            rsi_val = data['RSI'].iloc[-1]
                            if rsi_val < 30:
                                found_signals.append(f"🟢 איתות קנייה: {t} (RSI: {rsi_val:.1f})")
                            elif rsi_val > 70:
                                found_signals.append(f"🔴 איתות מכירה: {t} (RSI: {rsi_val:.1f})")
                
                if found_signals:
                    msg = "📊 התראות סורק אלגוריתמי:\n\n" + "\n".join(found_signals)
                    st.success("נמצאו הזדמנויות! שולח התראה...")
                    send_telegram_message(tg_token, tg_chat_id, msg)
                    st.info("ההתראה נשלחה ישירות לטלגרם שלך.")
                else:
                    st.write("הסריקה הסתיימה. לא נמצאו איתותים קיצוניים כרגע.")

if __name__ == "__main__":
    main()
