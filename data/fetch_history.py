#!/usr/bin/env python3
"""
歷史持股資料查詢工具

來源 1： 公開資訊觀測站 (MOPS) - 基金每月公布
來源 2： Wayback Machine - 對 pocket.tw 登錄項快照
來源 3： 統一投信 API 片段日期查詢

執行： python data/fetch_history.py
"""
import requests
import pandas as pd
import json
import re
import time
from pathlib import Path
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COOKIE_FILE = ROOT / ".cookie"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}


# ===========================================================
# 來源 1： 公開資訊觀測站 MOPS - 基金持股公告
# 00981A 每季公布前10大持股，公開資訊為公開信訊
# ===========================================================
def fetch_mops_holdings(year: int, season: int) -> list[dict]:
    """
    year: 民國年份（ex: 113 = 2024）
    season: 1~4
    回傳 [{stock_id, stock_name, weight}, ...]
    """
    url = "https://mops.twse.com.tw/mops/web/ajax_t138sb05"
    payload = {
        "encodeURIComponent": "1",
        "step": "1",
        "firstin": "1",
        "off": "1",
        "queryName": "co_id",
        "inpuType": "co_id",
        "TYPEK": "all",
        "isnew": "false",
        "co_id": "00981A",
        "year": str(year),
        "season": str(season),
    }
    headers = {
        **HEADERS,
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://mops.twse.com.tw/mops/web/t138sb05",
    }
    try:
        r = requests.post(url, data=payload, headers=headers, timeout=20)
        tables = pd.read_html(r.text)
        for t in tables:
            cols = list(t.columns)
            # 找含股票代號的表
            if any("代號" in str(c) or "證券" in str(c) for c in cols):
                print(f"  [MOPS] {year}Q{season} 找到表格: {cols[:5]}")
                return t.to_dict("records")
    except Exception as e:
        print(f"  [MOPS] {year}Q{season} 失敗: {e}")
    return []


# ===========================================================
# 來源 2： Wayback Machine CDX API
# 查詢 pocket.tw 是否有對應日期的快照
# ===========================================================
def list_wayback_snapshots(url_pattern: str, start: str, end: str) -> list[dict]:
    """
    回傳指定期間內所有快照列表
    start/end: 'YYYYMMDD'
    """
    cdx_url = (
        f"https://web.archive.org/cdx/search/cdx"
        f"?url={url_pattern}&output=json&fl=timestamp,statuscode"
        f"&from={start}&to={end}&limit=50&collapse=timestamp:6"
    )
    try:
        r = requests.get(cdx_url, timeout=20)
        rows = json.loads(r.text)
        if len(rows) <= 1:
            return []
        return [{"timestamp": row[0], "status": row[1]} for row in rows[1:]]
    except Exception as e:
        print(f"  [Wayback CDX] 失敗: {e}")
        return []


def fetch_wayback_snapshot(timestamp: str, original_url: str, cookie: str = "") -> str:
    """抓取指定時間點的快照內容"""
    url = f"https://web.archive.org/web/{timestamp}/{original_url}"
    h = {**HEADERS}
    if cookie:
        h["Cookie"] = cookie
    try:
        r = requests.get(url, headers=h, timeout=30)
        return r.text
    except Exception as e:
        print(f"  [Wayback] {timestamp} 失敗: {e}")
        return ""


def try_wayback_pocket(cookie: str = "") -> list[dict]:
    """
    嘗試從 Wayback Machine 抓取 pocket.tw 的持股快照
    由於 pocket.tw 為動態頁面，成功率低但尝試看看
    """
    TARGET = "https://www.pocket.tw/etf/tw/00981A/fundholding"
    API_TARGET = (
        "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
        "?action=getdtnodata&DtNo=59449513*"
    )
    print("  查詢 Wayback Machine 快照...")

    # 查詢 API 端點快照（更可能有用）
    snapshots = list_wayback_snapshots(API_TARGET, "20250501", "20260319")
    print(f"  API 快照數: {len(snapshots)}")
    for s in snapshots[:5]:
        print(f"    {s['timestamp']} status={s['status']}")

    return snapshots


# ===========================================================
# 來源 3： 統一投信官方 - 嘗試帶日期參數查詢
# ===========================================================
def fetch_upamc_by_date(query_date: str, cookie: str = "") -> list[dict]:
    """
    query_date: 'YYYYMMDD'
    統一投信內部可能支援指定日期查詢
    """
    # 嘗試將日期夸入 MTPeriod 參數
    # MTPeriod 對應的可能是 Unix timestamp 或序號
    year  = query_date[:4]
    month = query_date[4:6]
    day   = query_date[6:8]

    urls = [
        # 嘗試從統一投信官方 API 帶日期查詢
        f"https://www.upamc.com.tw/api/etf/holding?code=00981A&date={year}{month}{day}",
        f"https://api.upamc.com.tw/ETF/GetETFHolding?FundCode=00981A&Date={year}{month}{day}",
        # pocket.tw 嘗試帶日期
        (
            f"https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
            f"?action=getdtnodata&DtNo=59449513"
            f"&ParamStr=AssignID%3D00981A%3BQueryDate%3D{year}{month}{day}%3B"
            f"&FilterNo=0"
        ),
    ]
    h = {**HEADERS}
    if cookie:
        h["Cookie"] = cookie
        h["Referer"] = "https://www.pocket.tw/etf/tw/00981A/fundholding"

    for url in urls:
        try:
            r = requests.get(url, headers=h, timeout=15)
            if r.status_code == 200 and len(r.text) > 50:
                body = r.text.strip()
                if body.startswith(("[", "{")):
                    data = json.loads(body)
                    rows = data if isinstance(data, list) else data.get("Data", [])
                    if rows:
                        print(f"  [upamc/pocket] {query_date} 成功！ {len(rows)} 筆")
                        return rows
        except Exception as e:
            pass
    return []


# ===========================================================
# 主流程
# ===========================================================
if __name__ == "__main__":
    cookie = COOKIE_FILE.read_text(encoding="utf-8").strip() if COOKIE_FILE.exists() else ""

    print("=" * 60)
    print("歷史持股資料查詢")
    print("=" * 60)

    # ─ 1. MOPS 公開資訊 ─
    print("\n[來源 1] MOPS 公開資訊觀測站")
    # 00981A 上市 2025-05，所以從民國 114 年開始
    for year_roc, season in [(114, 2), (114, 3), (114, 4), (115, 1)]:
        rows = fetch_mops_holdings(year_roc, season)
        if rows:
            print(f"  MOPS {year_roc}Q{season} 找到 {len(rows)} 筆")
        time.sleep(1)

    # ─ 2. Wayback Machine ─
    print("\n[來源 2] Wayback Machine")
    snapshots = try_wayback_pocket(cookie)

    # ─ 3. pocket.tw 嘗試指定日期 ─
    print("\n[來源 3] pocket.tw 帶日期參數")
    # 每個月第一個持股公布日對應日期
    test_dates = [
        "20251001", "20251101", "20251201",
        "20260101", "20260201",
    ]
    for d in test_dates:
        rows = fetch_upamc_by_date(d, cookie)
        if rows:
            print(f"  {d} 找到資料!")
        else:
            print(f"  {d} 無資料")
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("結論：")
    print("• 若 MOPS 有資料 → 每季前10大持股，樣本稍少")
    print("• 若 Wayback 有 API 快照 → 可以拿到歷史資料")
    print("• 若全部無效 → 建議每月初手動執行 download_real_holdings.py 累積")
