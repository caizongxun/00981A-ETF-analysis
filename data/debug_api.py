#!/usr/bin/env python3
"""
API 診斷工具 - 印出 pocket.tw 真實回傳內容
執行： python data/debug_api.py
"""
import requests
import json
from pathlib import Path

COOKIE_FILE = Path(__file__).resolve().parent.parent / ".cookie"

cookie = COOKIE_FILE.read_text(encoding="utf-8").strip() if COOKIE_FILE.exists() else ""
if not cookie:
    print("[ERROR] 找不到 .cookie 檔，請先執行 python data/input_cookie.py")
    exit(1)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-TW,zh;q=0.9",
    "Referer": "https://www.pocket.tw/etf/tw/00981A/fundholding",
    "Origin": "https://www.pocket.tw",
    "Cookie": cookie,
}

# 所有要測試的 URL
TEST_URLS = [
    # 原始抓到的兩個 URL
    (
        "M722 持股 (offset=0)",
        "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
        "?action=getdtnodata&DtNo=59449513"
        "&ParamStr=AssignID%3D00981A%3BMTPeriod%3D0%3BDTMode%3D0%3BDTRange%3D1%3BDTOrder%3D1%3BMajorTable%3DM722%3B"
        "&FilterNo=0"
    ),
    (
        "M066 持股 (offset=0)",
        "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
        "?action=getdtnodata&DtNo=61495191"
        "&ParamStr=AssignID%3D98642180%3BMTPeriod%3D0%3BDTMode%3D0%3BDTRange%3D1%3BDTOrder%3D1%3BMajorTable%3DM066%3B"
        "&FilterNo=0"
    ),
    # 嘗試不同 MTPeriod
    (
        "M722 持股 (offset=1)",
        "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
        "?action=getdtnodata&DtNo=59449513"
        "&ParamStr=AssignID%3D00981A%3BMTPeriod%3D1%3BDTMode%3D0%3BDTRange%3D1%3BDTOrder%3D1%3BMajorTable%3DM722%3B"
        "&FilterNo=0"
    ),
    # 嘗試不帶 ParamStr 的簡化版
    (
        "M722 簡化版",
        "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
        "?action=getdtnodata&DtNo=59449513&ParamStr=AssignID%3D00981A%3B&FilterNo=0"
    ),
]

for name, url in TEST_URLS:
    print(f"\n{'='*60}")
    print(f"[{name}]")
    print(f"URL: {url[:100]}...")
    try:
        r = requests.get(url, headers=headers, timeout=15)
        print(f"HTTP Status: {r.status_code}")
        print(f"Content-Type: {r.headers.get('Content-Type', 'unknown')}")
        body = r.text.strip()
        print(f"Response (前 500 字元):")
        print(body[:500])
        print()
        # 嘗試解析 JSON
        if body.startswith(("[", "{")):
            try:
                data = json.loads(body)
                print(f"JSON 解析成功！ 類型={type(data).__name__}")
                if isinstance(data, list):
                    print(f"  元素數: {len(data)}")
                    if data:
                        print(f"  第一筆欄位: {list(data[0].keys()) if isinstance(data[0], dict) else data[0]}")
                elif isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")
            except Exception as e:
                print(f"  JSON 解析失敗: {e}")
    except Exception as e:
        print(f"  請求失敗: {e}")

print(f"\n{'='*60}")
print("完成！請將上面的輸出複製專給我看")
