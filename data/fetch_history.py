#!/usr/bin/env python3
"""
00981A 全期歷史持股下載
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

# ── 直接嘗試 .TW 和 .TWO，找到有股價的那個 ──
_SUFFIX_CACHE: dict[str, str] = {}

def get_valid_symbol(stock_id: str) -> tuple[str, str] | None:
    """
    回傳 (symbol, suffix) 或 None
    """
    if stock_id in _SUFFIX_CACHE:
        s = _SUFFIX_CACHE[stock_id]
        return (f"{stock_id}{s}", s)

    for suffix in (".TW", ".TWO"):
        sym = f"{stock_id}{suffix}"
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if not hist.empty:
                _SUFFIX_CACHE[stock_id] = suffix
                return (sym, suffix)
        except:
            pass
    # 兩個都失敗
    _SUFFIX_CACHE[stock_id] = None
    return None


def fetch_holdings_by_date(query_date: str, cookie: str) -> list[dict]:
    url = (
        "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
        "?action=getdtnodata&DtNo=59449513"
        f"&ParamStr=AssignID%3D00981A%3BQueryDate%3D{query_date}%3B"
        "&FilterNo=0"
    )
    h = {**HEADERS_BASE, "Cookie": cookie}
    r = requests.get(url, headers=h, timeout=20)
    r.raise_for_status()
    data = json.loads(r.text)
    rows = data.get("Data", [])
    result = []
    for row in rows:
        if len(row) < 3:
            continue
        sid = str(row[1]).strip()
        if not re.match(r'^\d{4,6}$', sid):
            continue
        result.append({
            "api_date":   str(row[0]),
            "stock_id":   sid,
            "stock_name": str(row[2]).strip(),
            "weight":     float(row[3]) if len(row) > 3 else 0.0,
            "shares":     str(row[4]) if len(row) > 4 else "",
        })
    return result


def get_features_for_period(candidate_pool: set, holding_ids: set, period: str) -> pd.DataFrame:
    records = []
    total   = len(candidate_pool)
    for i, tk in enumerate(sorted(candidate_pool), 1):
        result = get_valid_symbol(tk)
        if result is None:
            print(f"    [{i}/{total}] {tk} [SKIP] 兩個後綴都無股價")
            continue
        sym, suffix = result
        print(f"    [{i}/{total}] {sym}", end=" ", flush=True)
        try:
            t      = yf.Ticker(sym)
            info   = t.info
            hist   = t.history(period="9mo")
            if hist.empty:
                print("[SKIP]")
                continue
            closes = hist["Close"]
            mom3  = (closes.iloc[-1] / closes.iloc[-63]  - 1) if len(closes) >= 63  else np.nan
            mom6  = (closes.iloc[-1] / closes.iloc[-126] - 1) if len(closes) >= 126 else np.nan
            vol60 = closes.pct_change().tail(60).std() * np.sqrt(252)
            records.append({
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
                "in_etf":           int(tk in holding_ids),
                "period":           period,
            })
            print("OK")
        except Exception as e:
            print(f"[ERR] {e}")
    df = pd.DataFrame(records)
    if not df.empty:
        df["market_cap_rank"] = df["market_cap"].rank(ascending=False)
    return df


if __name__ == "__main__":
    if not COOKIE_FILE.exists():
        print("[ERROR] 請先執行: python data/input_cookie.py")
        exit(1)
    cookie = COOKIE_FILE.read_text(encoding="utf-8").strip()

    # ── Step 1: 下載所有月份持股 ──
    START = date(2025, 6, 1)
    END   = date.today().replace(day=1)

    raw_records     = []
    period_holdings = {}

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
                # 只保留電子/科技相關股（2xxx~8xxx）
                ids = {
                    r["stock_id"] for r in rows
                    if re.match(r'^[2-9]\d{3}$', r["stock_id"])
                }
                print(f"{len(ids)} 支")
                period_holdings[period] = ids
                for r in rows:
                    if r["stock_id"] in ids:
                        raw_records.append({"period": period, **r})
            else:
                print("無資料")
        except Exception as e:
            print(f"[ERR] {e}")
        cur += relativedelta(months=1)
        time.sleep(0.5)

    raw_df = pd.DataFrame(raw_records)
    raw_path = DATA_DIR / "real_holdings_raw.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\n原始持股已存: {raw_path} ({len(raw_df)} 筆, {raw_df['period'].nunique()} 期)")

    if not period_holdings:
        print("[ERROR] 沒有資料")
        exit(1)

    # 候選池 = 所有期出現過
    candidate_pool = set()
    for ids in period_holdings.values():
        candidate_pool.update(ids)
    print(f"\n候選池: {len(candidate_pool)} 支")

    # 預先建立 suffix cache（會自動嘗試 .TW 再試 .TWO）
    print("\n確認候選池股票後綴...(TW/TWO)")
    valid_pool = set()
    for tk in sorted(candidate_pool):
        result = get_valid_symbol(tk)
        if result:
            print(f"  {tk} -> {result[0]} [OK]")
            valid_pool.add(tk)
        else:
            print(f"  {tk} -> [SKIP] Yahoo 無資料")
        time.sleep(0.3)

    print(f"\n有效候選池: {len(valid_pool)} 支 ({len(candidate_pool)-len(valid_pool)} 支無股價已移除)")

    # ── Step 2: 特徵 ──
    print("\n取得 yfinance 特徵...")
    all_dfs = []
    for period, holding_ids in sorted(period_holdings.items()):
        # 將 holding_ids 也過濾為 valid_pool 內的
        valid_holding = holding_ids & valid_pool
        print(f"\n  [{period}] 入選 {len(valid_holding)} 支: {sorted(valid_holding)}")
        df = get_features_for_period(valid_pool, valid_holding, period)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1)

    if all_dfs:
        full = pd.concat(all_dfs, ignore_index=True)
        hist_path = DATA_DIR / "holdings_history.csv"
        full.to_csv(hist_path, index=False, encoding="utf-8-sig")
        print(f"\n完成! holdings_history.csv ({len(full)} 筆, {full['period'].nunique()} 期)")
        print("接下來執行: python backtest/walk_forward.py")
    else:
        print("[ERROR] 特徵取得失敗")
