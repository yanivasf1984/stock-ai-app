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

st.set_page_config(page_title="AI Stock Analytics Pro - Diamond Edition", page_icon="💎", layout="wide")

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
    
    sp500_hardcoded = [
        'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 
        'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 
        'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'AAP', 'AMAT', 'APTV', 'ACGL', 
        'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 
        'BAC', 'BK', 'BBWI', 'BAX', 'BDX', 'BRK-B', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BX', 'BA', 'BKNG', 
        'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF-B', 'BG', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 
        'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'COR', 'CNC', 'CNP', 
        'CDAY', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTASגישה של צלף. עדיף לקבל 2-3 התראות מדויקות בשבוע של מניות שעומדות להתפוצץ, מאשר רשימה של 20 מניות בינוניות שיבזבזו לך את הזמן והכסף. 

כדי להפוך את הסורק ל"מסננת אכזרית" לטווח הקצר, הוספתי למנוע את ה-**MACD**. מעכשיו, כדי שמניה בכלל תופיע אצלך בדוח, היא חייבת לעבור מסלול מכשולים משולש:
1. **כיוון וחוזק:** מומנטום חזק מאוד (ADX מעל 25).
2. **דלק:** כסף מוסדי אמיתי שזורם פנימה בדיוק עכשיו (CMF חיובי).
3. **תזמון כניסה מדויק (השדרוג החדש):** ה-MACD חייב להיות חיובי, וההיסטוגרמה שלו חייבת להיות **במגמת עלייה ביחס לאתמול**. זה אומר שאנחנו תופסים את המומנטום בזמן שהוא מאיץ, ולא כשהוא מתחיל להתעייף.
4. **חותמת גומי:** מנוע ה-AI חייב לתת הסתברות של מעל 54% לעלייה.

בנוסף, הוספתי את ה-MACD כגרף חמישי ואינדיקטור ברור במסך של "ניתוח מניה בודדת", כדי שתוכל לראות את הפריצה בעיניים.

העתק את הקוד המלא הבא והחלף את כל התוכן בקובץ **`app.py`**:

```python
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

st.set_page_config(page_title="AI Stock Analytics Pro - Master Edition", page_icon="📈", layout="wide")

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
    "
