#!/usr/bin/env python3
"""
00981A ETF 每日持股明細爬取腳本
資料來源: CMoney / 口袋證券
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def fetch_cmoney_holdings(etf_code="00981A"):
    """
    從 CMoney 爬取 ETF 持股明細
    """
    url = f"https://www.cmoney.tw/etf/tw/{etf_code}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        # 實際上需要解析 HTML 或呼叫其 API endpoint
        # 以下為結構示意
        print(f"[{datetime.now()}] 成功取得 {etf_code} 頁面")
        return resp.text
    except requests.RequestException as e:
        print(f"[ERROR] 爬取失敗: {e}")
        return None

def fetch_pocket_holdings(etf_code="00981A"):
    """
    從口袋證券爬取 ETF 每日持股明細
    API endpoint 格式 (需實測驗證):
    https://www.pocket.tw/etf/tw/{etf_code}/fundholding
    """
    url = f"https://www.pocket.tw/etf/tw/{etf_code}/fundholding"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        print(f"[{datetime.now()}] 成功取得 {etf_code} 持股明細")
        return resp.text
    except requests.RequestException as e:
        print(f"[ERROR] 爬取失敗: {e}")
        return None

def save_holdings(df: pd.DataFrame, date: str, output_dir="../data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"holdings_{date}.csv")
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {filepath}")

if __name__ == "__main__":
    # 爬取今日持股
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"開始爬取 00981A 持股明細 ({today})")
    
    html = fetch_pocket_holdings()
    if html:
        print("爬取成功，下一步請執行 feature_engineering.py")
    else:
        print("爬取失敗，請檢查網路連線或目標網站結構")
