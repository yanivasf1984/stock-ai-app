import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# 1. הגדרות תצוגה מקצועית (מסך רחב)
st.set_page_config(page_title="AI Quant Fund", layout="wide", initial_sidebar_state="expanded")

def main():
    st.title("🤖 AI Quant Fund - מערכת מסחר אלגוריתמית")
    
    # 2. חלוקה לכרטיסיות (Tabs) כמו בפלטפורמת קוואנט
    tab_trade, tab_backtest, tab_settings = st.tabs(["📊 מסוף מסחר", "🔄 אסטרטגיות ו-Backtest", "⚙️ הגדרות והתראות טלגרם"])
    
    with tab_trade:
        st.subheader("ניתוח פוזיציות וגרפים אינטראקטיביים")
        ticker = st.text_input("הכנס סימול מניה (למשל SPY, TSLA, AAPL):", "SPY")
        
        if ticker:
            # משיכת נתוני שוק
          df = yf.Ticker(ticker).history(period="6mo")
            ifnot df.empty:
                # ציור גרף נרות (Candlesticks) מקצועי במקום גרף קווים פשוט
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'])])
                fig.update_layout(title=f'מגמת מחירים ותנודתיות - {ticker}', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("לא נמצאו נתונים. בדוק את סימול המניה.")
                
    with tab_backtest:
        st.subheader("מנוע בדיקת אסטרטגיות (Backtesting)")
        st.info("כאן יוטמע מנוע ה-Machine Learning שיציג אחוזי הצלחה היסטוריים, Sharpe Ratio וניהול סיכונים.")
        
    with tab_settings:
        st.subheader("ניהול סיכונים והתראות טלגרם")
        
        # פרטי הטלגרם שלך (כבר מוזנים אוטומטית לפי מה שעבד לנו)
        tg_token = st.text_input("Telegram Bot Token:", value="8979601396:AAFQjLLDf81HJPh8RjkpcpzQYxYAAHd8jpw", type="password")
        tg_chat_id = st.text_input("Telegram Chat ID:", value="5117812191")
        
        st.markdown("---")
        st.write("מנוע סריקת מניות ברקע - בקרוב יתווסף כאן כפתור להפעלת סריקה יומית אוטומטית.")

if __name__ == "__main__":
    main()
