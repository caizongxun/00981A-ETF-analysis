#!/usr/bin/env python3
"""
診斷 MOPS 營收 API 實際回傳的內容
執行: python data/debug_mops.py
"""
import requests
from io import StringIO
import pandas as pd

def test_mops(stock_id="2383", year_month="2025-12"):
    roc_year = int(year_month[:4]) - 1911
    month    = int(year_month[5:])
    print(f"\n測試: {stock_id} {year_month} (民國 {roc_year}年 {month}月)")
    print("="*60)

    for market in ("sii", "otc"):
        url  = "https://mops.twse.com.tw/mops/web/ajax_t21sc03"
        data = {
            "encodeURIComponent": "1",
            "step": "1",
            "firstin": "1",
            "off": "1",
            "keyword4": "",
            "code1": "",
            "TYPEK2": "",
            "checkbtn": "",
            "queryName": "co_id",
            "inpuType": "co_id",
            "TYPEK": market,
            "isnew": "false",
            "co_id": stock_id,
            "year": str(roc_year),
            "month": str(month).zfill(2),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://mops.twse.com.tw/mops/web/t21sc03",
        }
        print(f"\n[{market}] POST {url}")
        try:
            r = requests.post(url, data=data, headers=headers, timeout=20)
            print(f"  HTTP {r.status_code}")
            print(f"  Content-Type: {r.headers.get('Content-Type', 'N/A')}")
            print(f"  回傳內容前 500 字:")
            # 嘗試 utf-8 和 big5
            for enc in ("utf-8", "big5", "cp950"):
                try:
                    text = r.content.decode(enc)
                    print(f"  [{enc}] {text[:500]}")
                    print(f"  ...(total {len(text)} chars)")
                    # 嘗試解析表格
                    try:
                        tbls = pd.read_html(StringIO(text), thousands=",")
                        print(f"  解析到 {len(tbls)} 張表")
                        for i, tbl in enumerate(tbls):
                            print(f"  表[{i}] shape={tbl.shape}, columns={list(tbl.columns[:5])}")
                            print(f"       第一行: {tbl.iloc[0].tolist()[:5]}" if len(tbl) > 0 else "       空表")
                    except Exception as e:
                        print(f"  pd.read_html 失敗: {e}")
                    break
                except:
                    continue
        except Exception as e:
            print(f"  請求失敗: {e}")

if __name__ == "__main__":
    # 測試已知上市股
    test_mops("2383", "2025-12")  # 光磊，上市
    # 測試已知上櫃股
    test_mops("6223", "2025-12")  # 永光補聽，上櫃
