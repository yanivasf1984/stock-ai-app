import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import requests

# 1. הגדרות תצוגה
st.set_page_config(page_title="AI Quant Fund", layout="wide", initial_sidebar_state="expanded")

def send_telegram_message(token, chat_id, text):
    """פונקציה קטנה ששולחת את ההתראה לטלגרם מאחורי הקלעים"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text})
    except Exception as e:
        pass

def main():
    st.title("🤖 AI Quant Fund - פלטפורמת מסחר וסריקה")
    
    # חלוקה ל-3 כרטיסיות העבודה שביקשת
    tab_trade, tab_backtest, tab_settings = st.tabs(["📊 מסוף מסחר ואינדיקטורים", "🔄 סימולציית אסטרטגיות (Backtest)", "⚙️ סורק אוטומטי וטלגרם"])
    
    # --- מנוע 1: מסוף המסחר (הגרפים הישנים + בינה מלאכותית) ---
    with tab_trade:
        col1, col2 = st.columns([4, 1])
        with col2:
            ticker = st.text_input("סימול מניה:", "SPY")
            period = st.selectbox("תקופת זמן:", ["3mo", "6mo", "1y", "2y", "5y"], index=1)
            
        if ticker:
            df = yf.Ticker(ticker).history(period=period)
            if not df.empty:
                # החזרת האינדיקטורים המוכרים מהגרסה הקודמת
                df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
                df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
                df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                
                with col1:
                    # יצירת גרף מפוצל: נרות למעלה, RSI למטה
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                        vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    
                    # הוספת הנרות היפניים והממוצעים הנעים למסך העליון
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                                 low=df['Low'], close=df['Close'], name='מחיר'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='SMA 20'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
                    
                    # הוספת מדד ה-RSI למסך התחתון עם קווי אזהרה
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
                    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
                    
                    fig.update_layout(height=650, xaxis_rangeslider_visible=False, margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 🧠 מודל החלטות")
                    current_rsi = df['RSI'].iloc[-1]
                    # לוגיקת מסחר בסיסית מבוססת מומנטום
                    if current_rsi < 30:
                        st.success("🟢 איתות קנייה (Oversold)")
                    elif current_rsi > 70:
                        st.error("🔴 איתות מכירה (Overbought)")
                    else:
                        st.warning("🟡 המתנה (נטרלי)")
                        
                    st.metric("RSI נוכחי", f"{current_rsi:.1f}")
                    st.metric("מחיר סגירה אחרון", f"${df['Close'].iloc[-1]:.2f}")

    # --- מנוע 2: סימולציה ובדיקה לאחור ---
    with tab_backtest:
        st.subheader("מנוע בדיקת אסטרטגיה היסטורית (Backtest)")
        st.write("מנוע זה בודק מה היה קורה אילו סחרת במניה זו לפי חוקי ה-RSI, לעומת החזקה פסיבית שלה.")
        
        if 'df' in locals() and not df.empty:
            if st.button("▶️ הרץ סימולציה על נתוני העבר"):
                initial_capital = 10000 # קופת התחלה של 10,000 דולר
                capital = initial_capital
                position = 0
                
                # מעבר על כל ימי המסחר בהיסטוריה שהורדנו
                for i in range(1, len(df)):
                    if df['RSI'].iloc[i-1] < 30 and position == 0:  # קנייה בתחתית
                        position = capital / df['Close'].iloc[i]
                        capital = 0
                    elif df['RSI'].iloc[i-1] > 70 and position > 0: # מכירה בשיא
                        capital = position * df['Close'].iloc[i]
                        position = 0
                
                # חישוב ערך סופי
                final_value = capital if position == 0 else position * df['Close'].iloc[-1]
                profit_pct = ((final_value - initial_capital) / initial_capital) * 100
                buy_hold_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("רווח האלגוריתם", f"{profit_pct:.2f}%")
                col_b2.metric("רווח קנייה והחזקה (פסיבי)", f"{buy_hold_pct:.2f}%")
                col_b3.metric("הון סופי מקופת $10,000", f"${final_value:,.2f}")

    # --- מנוע 3: סורק שוק אוטומטי והתראות טלגרם ---
    with tab_settings:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("📱 חיבור שרת טלגרם")
            tg_token = st.text_input("Telegram Bot Token:", value="8979601396:AAFQjLLDf81HJPh8RjkpcpzQYxYAAHd8jpw", type="password")
            tg_chat_id = st.text_input("Telegram Chat ID:", value="5117812191")
        
        with col_s2:
            st.subheader("🔎 סורק הזדמנויות בשוק")
            scan_list = st.text_input("רשימת מניות לסריקה (מופרדות בפסיק):", "SPY,QQQ,AAPL,TSLA,MSFT,NVDA")
            
            if st.button("🚀 הפעל סריקת שוק עכשיו"):
                tickers_to_scan = [x.strip() for x in scan_list.split(",")]
                found_signals = []
                
                with st.spinner("סורק את השוק ומנתח אינדיקטורים..."):
                    for t in tickers_to_scan:
                        data = yf.Ticker(t).history(period="1mo")
                        if not data.empty and len(data) > 15:
                            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
                            rsi_val = data['RSI'].iloc[-1]
                            # תנאי מציאת הזדמנות
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
