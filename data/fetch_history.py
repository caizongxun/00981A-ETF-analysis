#!/usr/bin/env python3
"""
00981A 全期歷史持股下載

宇宙設計：
  - 00981A 實際持股（約 49 支）為 in_etf=1
  - 台灣電子股宇宙（約 120 支）中未入選者為 in_etf=0
  - 這樣分類器才能學到區分

執行： python data/fetch_history.py
"""
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import json
import time
import re
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COOKIE_FILE = ROOT / ".cookie"

# 台灣電子股宇宙（市値前大型電子股）
# 00981A 會從這裡選約 49 支，其餘標為 0
UNIVERSE = [
    # 半導體
    "2330","2303","2379","2454","3711","2408","3037","2327","2344","2449",
    # 電子零組件
    "2317","2308","2357","2382","2383","2368","2313","3017","3653","3665",
    "2059","2404","2439","3583","3533","3376","3189","3661","6515","6191",
    "6805","8046","8210","8996","2002","3045",
    # 通訊網路
    "2345","3045","4552","3711",
    # 電腦週邊
    "2382","6669","2317",
    # 光電
    "3008","2439",
    # 上櫃電子
    "5274","6223","8299","6274","3211","3217","3264","3324",
    "4979","5347","5536","6488","6510",
    # 其他大型電子不在持股內的股票（宇宙擴充用）
    "2412","2881","2882","2886","2891",  # 電信/金融（不會入選電子ETF）
    "2002","1301","1303",  # 傳産/原料（不會入選）
    "2823","2880","2884","2885","2887","2888",  # 金融
    "2912","2801",  # 將金融/民生
    # 電子中小型不常入選
    "3714","4958","5269","6116","6271","6278","6409","6414",
    "6449","6456","6533","6550","6592","6770","8299",
]
# 去重
_seen = set()
UNIVERSE_CLEAN = []
for x in UNIVERSE:
    if x not in _seen:
        _seen.add(x)
        UNIVERSE_CLEAN.append(x)
UNIVERSE = UNIVERSE_CLEAN

INDUSTRY_MAP = {
    "2330": "半導體", "2308": "電子零組件", "6669": "電腦週邊",
    "2368": "電子零組件", "2383": "電子零組件", "2454": "半導體",
    "3008": "光電", "3711": "通訊網路", "8299": "半導體",
    "6223": "半導體", "5274": "半導體", "3017": "電子零組件",
    "3653": "電子零組件", "2345": "通訊網路", "3037": "半導體",
    "3665": "電子零組件", "2317": "電腦週邊", "2382": "電腦週邊",
}

HEADERS_BASE = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.pocket.tw/etf/tw/00981A/fundholding",
}

_SUFFIX_CACHE: dict = {}

def get_valid_symbol(stock_id: str):
    if stock_id in _SUFFIX_CACHE:
        s = _SUFFIX_CACHE[stock_id]
        return (f"{stock_id}{s}", s) if s else None
    for suffix in (".TW", ".TWO"):
        try:
            hist = yf.Ticker(f"{stock_id}{suffix}").history(period="5d")
            if not hist.empty:
                _SUFFIX_CACHE[stock_id] = suffix
                return (f"{stock_id}{suffix}", suffix)
        except:
            pass
    _SUFFIX_CACHE[stock_id] = None
    return None


def fetch_holdings_by_date(query_date: str, cookie: str) -> set:
    """\u56de傳該日期 00981A 實際持股的 stock_id set"""
    url = (
        "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
        "?action=getdtnodata&DtNo=59449513"
        f"&ParamStr=AssignID%3D00981A%3BQueryDate%3D{query_date}%3B"
        "&FilterNo=0"
    )
    r = requests.get(url, headers={**HEADERS_BASE, "Cookie": cookie}, timeout=20)
    r.raise_for_status()
    rows = json.loads(r.text).get("Data", [])
    ids = set()
    for row in rows:
        sid = str(row[1]).strip() if len(row) > 1 else ""
        if re.match(r'^\d{4,6}$', sid):
            ids.add(sid)
    return ids


def get_features_one(tk: str) -> dict | None:
    result = get_valid_symbol(tk)
    if result is None:
        return None
    sym, _ = result
    try:
        t      = yf.Ticker(sym)
        info   = t.info
        hist   = t.history(period="9mo")
        if hist.empty:
            return None
        closes = hist["Close"]
        mom3  = (closes.iloc[-1] / closes.iloc[-63]  - 1) if len(closes) >= 63  else np.nan
        mom6  = (closes.iloc[-1] / closes.iloc[-126] - 1) if len(closes) >= 126 else np.nan
        vol60 = closes.pct_change().tail(60).std() * np.sqrt(252)
        return {
            "stock_id":         tk,
            "industry":         INDUSTRY_MAP.get(tk, "不明"),
            "roe":              info.get("returnOnEquity",   np.nan),
            "pe_ratio":         info.get("trailingPE",       np.nan),
            "pb_ratio":         info.get("priceToBook",      np.nan),
            "gross_margin":     info.get("grossMargins",     np.nan),
            "operating_margin": info.get("operatingMargins", np.nan),
            "debt_to_equity":   info.get("debtToEquity",     np.nan),
            "market_cap":       info.get("marketCap",        np.nan),
            "price_mom_3m":     mom3,
            "price_mom_6m":     mom6,
            "volatility_60d":   vol60,
        }
    except:
        return None


if __name__ == "__main__":
    if not COOKIE_FILE.exists():
        print("[ERROR] 請先執行: python data/input_cookie.py")
        exit(1)
    cookie = COOKIE_FILE.read_text(encoding="utf-8").strip()

    # Step 1: 下載實際持股
    START = date(2025, 6, 1)
    END   = date.today().replace(day=1)
    period_holdings = {}  # period -> set of actual holdings

    cur = START
    while cur <= END:
        period   = cur.strftime("%Y-%m")
        date_str = cur.strftime("%Y%m%d")
        print(f"[{period}] 查詢...", end=" ", flush=True)
        try:
            ids = fetch_holdings_by_date(date_str, cookie)
            if not ids:
                ids = fetch_holdings_by_date(cur.replace(day=15).strftime("%Y%m%d"), cookie)
            if ids:
                print(f"實際持股 {len(ids)} 支")
                period_holdings[period] = ids
            else:
                print("無資料")
        except Exception as e:
            print(f"[ERR] {e}")
        cur += relativedelta(months=1)
        time.sleep(0.5)

    if not period_holdings:
        print("[ERROR] 沒有資料")
        exit(1)

    # Step 2: 確認宇宙内股票後綴
    print(f"\n宇宙大小: {len(UNIVERSE)} 支")
    print("確認宇宙內股票後綴...")
    valid_universe = set()
    for i, tk in enumerate(UNIVERSE, 1):
        result = get_valid_symbol(tk)
        if result:
            valid_universe.add(tk)
            print(f"  [{i}/{len(UNIVERSE)}] {tk} -> {result[0]} [OK]")
        else:
            print(f"  [{i}/{len(UNIVERSE)}] {tk} -> [SKIP]")
        time.sleep(0.2)
    print(f"有效宇宙: {len(valid_universe)} 支")

    # Step 3: 取得特徵
    print("\n取得 yfinance 特徵...")
    feat_base = {}
    total = len(valid_universe)
    for i, tk in enumerate(sorted(valid_universe), 1):
        print(f"  [{i}/{total}] {tk}", end=" ", flush=True)
        feat = get_features_one(tk)
        if feat:
            feat_base[tk] = feat
            print("OK")
        else:
            print("[SKIP]")
        time.sleep(0.3)

    # Step 4: 每期組合持股標籤
    all_rows = []
    for period, holding_ids in sorted(period_holdings.items()):
        n1 = n0 = 0
        for tk, feat in feat_base.items():
            in_etf = int(tk in holding_ids)
            all_rows.append({**feat, "in_etf": in_etf, "period": period})
            if in_etf: n1 += 1
            else:      n0 += 1
        print(f"  [{period}] in_etf=1: {n1} 支, in_etf=0: {n0} 支")

    full = pd.DataFrame(all_rows)
    full["market_cap_rank"] = full.groupby("period")["market_cap"].rank(ascending=False)

    hist_path = DATA_DIR / "holdings_history.csv"
    full.to_csv(hist_path, index=False, encoding="utf-8-sig")
    print(f"\n完成! holdings_history.csv ({len(full)} 筆, {full['period'].nunique()} 期)")
    print(f"in_etf 分布: {full['in_etf'].value_counts().to_dict()}")
    print("接下來執行: python backtest/walk_forward.py")
