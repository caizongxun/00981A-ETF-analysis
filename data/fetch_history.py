#!/usr/bin/env python3
"""
00981A 全期歷史持股下載

重點：每期同時儲存
  - 候選池內全部股票 (in_etf=0 或 in_etf=1)
  - 讓分類器同時看到兩個類別

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

INDUSTRY_MAP = {
    "2330": "半導體", "4552": "通訊網路", "2308": "電子零組件", "6669": "電腦週邊",
    "2368": "電子零組件", "2383": "電子零組件", "2454": "半導體", "3008": "光電",
    "2317": "電腦週邊", "2382": "電腦週邊", "3711": "通訊網路", "2357": "電腦週邊",
    "2379": "半導體", "3691": "電子零組件", "8299": "半導體",
    "6274": "電子零組件", "6223": "半導體", "5274": "半導體",
    "3017": "電子零組件", "3653": "電子零組件", "2345": "通訊網路",
    "3037": "半導體", "3665": "電子零組件",
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


def fetch_holdings_by_date(query_date: str, cookie: str) -> list[dict]:
    url = (
        "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
        "?action=getdtnodata&DtNo=59449513"
        f"&ParamStr=AssignID%3D00981A%3BQueryDate%3D{query_date}%3B"
        "&FilterNo=0"
    )
    r = requests.get(url, headers={**HEADERS_BASE, "Cookie": cookie}, timeout=20)
    r.raise_for_status()
    rows = json.loads(r.text).get("Data", [])
    result = []
    for row in rows:
        sid = str(row[1]).strip() if len(row) > 1 else ""
        if not re.match(r'^[2-9]\d{3}$', sid):
            continue
        result.append({
            "stock_id":   sid,
            "stock_name": str(row[2]).strip() if len(row) > 2 else "",
            "weight":     float(row[3]) if len(row) > 3 else 0.0,
        })
    return result


def get_features_one(tk: str) -> dict | None:
    """\u53d6得單一股票特徵，失敗回傳 None"""
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

    # Step 1: 下載所有月份持股
    START = date(2025, 6, 1)
    END   = date.today().replace(day=1)
    period_holdings = {}   # period -> set of stock_id (in_etf=1)

    cur = START
    while cur <= END:
        period   = cur.strftime("%Y-%m")
        date_str = cur.strftime("%Y%m%d")
        print(f"[{period}] 查詢...", end=" ", flush=True)
        try:
            rows = fetch_holdings_by_date(date_str, cookie)
            if not rows:
                rows = fetch_holdings_by_date(cur.replace(day=15).strftime("%Y%m%d"), cookie)
            if rows:
                ids = {r["stock_id"] for r in rows}
                print(f"{len(ids)} 支入選")
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

    # Step 2: 建候選池 (所有期入選過的聯集)
    candidate_pool = set()
    for ids in period_holdings.values():
        candidate_pool.update(ids)
    print(f"\n候選池: {len(candidate_pool)} 支")

    # Step 3: 確認 suffix cache
    print("\n確認股票後綴...")
    valid_pool = set()
    for tk in sorted(candidate_pool):
        result = get_valid_symbol(tk)
        if result:
            valid_pool.add(tk)
            print(f"  {tk} -> {result[0]} [OK]")
        else:
            print(f"  {tk} -> [SKIP]")
        time.sleep(0.2)
    print(f"有效候選池: {len(valid_pool)} 支")

    # Step 4: 每期取得對候選池內所有股票的特徵 (in_etf=0 AND 1)
    # 特徵只取一次，各期共用（因為 yfinance 只有最新資料）
    print("\n取得 yfinance 特徵（各股票取一次）...")
    feat_base = {}  # stock_id -> feature dict
    total = len(valid_pool)
    for i, tk in enumerate(sorted(valid_pool), 1):
        print(f"  [{i}/{total}] {tk}", end=" ", flush=True)
        feat = get_features_one(tk)
        if feat:
            feat_base[tk] = feat
            print("OK")
        else:
            print("[SKIP]")
        time.sleep(0.3)

    # Step 5: 對每期組合持股標籤 (in_etf)
    all_rows = []
    for period, holding_ids in sorted(period_holdings.items()):
        valid_holding = holding_ids & set(feat_base.keys())
        for tk, feat in feat_base.items():
            row = {
                **feat,
                "in_etf": int(tk in valid_holding),
                "period":  period,
            }
            all_rows.append(row)

        n1 = sum(1 for tk in feat_base if tk in valid_holding)
        n0 = len(feat_base) - n1
        print(f"  [{period}] in_etf=1: {n1} 支, in_etf=0: {n0} 支")

    full = pd.DataFrame(all_rows)
    full["market_cap_rank"] = full.groupby("period")["market_cap"].rank(ascending=False)

    # 儲存原始持股
    raw_rows = []
    for period, ids in period_holdings.items():
        for sid in ids:
            raw_rows.append({"period": period, "stock_id": sid})
    pd.DataFrame(raw_rows).to_csv(DATA_DIR / "real_holdings_raw.csv",
                                  index=False, encoding="utf-8-sig")

    hist_path = DATA_DIR / "holdings_history.csv"
    full.to_csv(hist_path, index=False, encoding="utf-8-sig")
    print(f"\n完成! holdings_history.csv ({len(full)} 筆, {full['period'].nunique()} 期)")
    print(f"in_etf 分布: {full['in_etf'].value_counts().to_dict()}")
    print("接下來執行: python backtest/walk_forward.py")
