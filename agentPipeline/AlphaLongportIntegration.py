# 🔧 MCP-style AI Agent 整合 Alpha Vantage + LongPort API（美加股技術分析 + 即時行情）

# 0️⃣ 安裝必要套件
# pip install alpha_vantage longbridge pandas ta pandas-ta notion-client

import os
import time
import pandas as pd
import numpy as np
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from longbridge.openapi import QuoteContext, Config
from notion_client import Client
import talib
import pandas_ta as ta

# 🔑 API 金鑰（請自行設置環境變數）
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DB_ID = os.getenv("NOTION_DB_ID")

# ⚙️ LongPort 初始化
quote_ctx = QuoteContext(Config.from_env())

# 1️⃣ 抓歷史價格與技術指標（Alpha Vantage）
def fetch_technical_indicators(symbol="AAPL"):
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
    data = data.sort_index()
    close = data['4. close']

    df = pd.DataFrame()
    df['close'] = close
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
    df['rsi'] = talib.RSI(close, timeperiod=14)
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close)
    df = df.dropna()
    return df

# 2️⃣ 圖形形態分析（範例：MACD 金叉 + RSI 低）
def detect_entry_signal(df):
    last = df.iloc[-1]
    if last['macd'] > last['macd_signal'] and last['rsi'] < 40:
        return True, f"技術指標出現金叉 + 超賣 RSI ({last['rsi']:.2f})，可能反彈"
    return False, "無明顯入場信號"

# 3️⃣ 取得即時行情（LongPort）
def get_realtime_price(symbol="AAPL.US"):
    quote = quote_ctx.quote(symbol)
    return quote.latest_price

# 4️⃣ 推送到 Notion 筆記
notion = Client(auth=NOTION_TOKEN)

def push_to_notion(symbol, signal_desc, price):
    notion.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties={"Name": {"title": [{"text": {"content": f"{symbol} 分析報告"}}]}},
        children=[
            {"object": "block", "type": "paragraph", "paragraph": {"text": [{"type": "text", "text": {"content": signal_desc}}]}},
            {"object": "block", "type": "paragraph", "paragraph": {"text": [{"type": "text", "text": {"content": f"即時價格：{price}"}}]}}
        ]
    )

# 5️⃣ 主邏輯
if __name__ == '__main__':
    symbol = "AAPL"       # 可以替換成 TSLA, NVDA 等美股代碼
    lpb_symbol = symbol + ".US"  # LongPort 使用後綴

    print("📥 擷取技術指標...")
    df = fetch_technical_indicators(symbol)

    print("🔍 檢測入場信號...")
    signal, desc = detect_entry_signal(df)

    print("💬 分析結果：", desc)

    print("📡 取得即時價格...")
    price = get_realtime_price(lpb_symbol)

    print(f"📊 {symbol} 現價為 ${price:.2f}")

    print("📝 寫入 Notion 筆記...")
    push_to_notion(symbol, desc, price)

    print("✅ 完成分析與記錄。")
