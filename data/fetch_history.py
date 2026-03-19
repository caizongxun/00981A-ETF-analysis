#!/usr/bin/env python3
"""
00981A 全期歷史持股下載 - Point-in-Time 版

特徵設計：全部使用股價類，各期特徵在該期最後一個交易日計算
  - mom_1m:  該期最後一個月内報酬
  - mom_3m:  最後 3 個月報酬
  - mom_6m:  最後 6 個月報酬
  - vol_20d: 最後 20 日年化波動率
  - vol_60d: 最後 60 日年化波動率
  - rs_vs_market: 相對強度（股票報酬 - 0050 報酬）
  - above_ma60:  是否在 60 日均線上方
  - price_range: (high-low)/low，該期價格震盪幅度

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
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COOKIE_FILE = ROOT / ".cookie"

# 宇宙：台灣主要電子股 + 非電子股（供 in_etf=0 使用）
UNIVERSE_TW = [
    # 電子（大概率會入選 00981A）
    "2330","2303","2379","2454","3711","2408","3037","2327","2344","2449",
    "2317","2308","2357","2382","2383","2368","2313","3017","3653","3665",
    "2059","2404","2439","3583","3533","3376","3189","3661","6515","6191",
    "6805","8046","8210","8996","3045","2345","3376",
    # 上櫃電子
    "5274","6223","8299","6274","3211","3217","3264","3324",
    "4979","5347","5536","6488","6510","2002",
    # 非電子（約 15 支，保證有足夠 in_etf=0）
    "2412","2882","2886","2891","1301","1303",
    "2881","2884","2885","2912","2002","1326",
]
# 去重
_s = set()
UNIVERSE = [x for x in UNIVERSE_TW if not (_s.add(x) or x in _s - {x})]

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
            h = yf.Ticker(f"{stock_id}{suffix}").history(period="5d")
            if not h.empty:
                _SUFFIX_CACHE[stock_id] = suffix
                return (f"{stock_id}{suffix}", suffix)
        except:
            pass
    _SUFFIX_CACHE[stock_id] = None
    return None


def fetch_holdings_by_date(query_date: str, cookie: str) -> set:
    url = (
        "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
        "?action=getdtnodata&DtNo=59449513"
        f"&ParamStr=AssignID%3D00981A%3BQueryDate%3D{query_date}%3B"
        "&FilterNo=0"
    )
    r = requests.get(url, headers={**HEADERS_BASE, "Cookie": cookie}, timeout=20)
    r.raise_for_status()
    rows = json.loads(r.text).get("Data", [])
    return {str(row[1]).strip() for row in rows
            if len(row) > 1 and re.match(r'^\d{4,6}$', str(row[1]).strip())}


def calc_pit_features(closes: pd.Series, market: pd.Series,
                      as_of: pd.Timestamp) -> dict | None:
    """
    在 as_of 日期以前的資料計算 point-in-time 特徵
    """
    c = closes[closes.index <= as_of].dropna()
    m = market[market.index <= as_of].dropna()
    if len(c) < 20:
        return None

    p0   = float(c.iloc[-1])
    def mom(n):
        return float(c.iloc[-1] / c.iloc[-n] - 1) if len(c) >= n else np.nan

    vol20 = float(c.pct_change().tail(20).std() * np.sqrt(252)) if len(c) >= 20 else np.nan
    vol60 = float(c.pct_change().tail(60).std() * np.sqrt(252)) if len(c) >= 60 else np.nan
    ma60  = float(c.tail(60).mean()) if len(c) >= 60 else np.nan

    # 相對強度 vs 0050
    if len(m) >= 20 and len(c) >= 20:
        rs = float(c.iloc[-1]/c.iloc[-20] - m.iloc[-1]/m.iloc[-20])
    else:
        rs = np.nan

    # 該期內最高/最低價 range
    period_c = c.tail(21)  # 約一個月
    pr = float((period_c.max() - period_c.min()) / period_c.min()) if len(period_c) >= 5 else np.nan

    return {
        "mom_1m":       mom(21),
        "mom_3m":       mom(63),
        "mom_6m":       mom(126),
        "vol_20d":      vol20,
        "vol_60d":      vol60,
        "rs_vs_market": rs,
        "above_ma60":   float(p0 > ma60) if not np.isnan(ma60) else np.nan,
        "price_range":  pr,
    }


if __name__ == "__main__":
    if not COOKIE_FILE.exists():
        print("[ERROR] 請先執行: python data/input_cookie.py")
        exit(1)
    cookie = COOKIE_FILE.read_text(encoding="utf-8").strip()

    # Step 1: 下載實際持股
    START = date(2025, 6, 1)
    END   = date.today().replace(day=1)
    period_holdings = {}
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
        print("[ERROR] 沒有持股資料")
        exit(1)

    # Step 2: 確認宇宙後綴
    print(f"\n宇宙: {len(UNIVERSE)} 支，確認後綴...")
    valid_universe = {}
    for tk in UNIVERSE:
        r = get_valid_symbol(tk)
        if r:
            valid_universe[tk] = r[0]  # tk -> symbol
        time.sleep(0.15)
    print(f"有效宇宙: {len(valid_universe)} 支")

    # Step 3: 下載宇宙內所有股票完整股價及 0050 (2 年)
    print("\n下載股價資料...")
    symbols = list(valid_universe.values())
    price_data = {}
    for sym in symbols:
        try:
            h = yf.download(sym, start="2024-01-01",
                            progress=False, auto_adjust=True)
            if not h.empty:
                close = h["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.index = pd.to_datetime(close.index).tz_localize(None)
                price_data[sym] = close
                print(f"  {sym} OK ({len(close)} 日)")
        except Exception as e:
            print(f"  {sym} [ERR] {e}")
        time.sleep(0.1)

    # 0050 市場基準
    mkt = yf.download("0050.TW", start="2024-01-01",
                      progress=False, auto_adjust=True)["Close"]
    if isinstance(mkt, pd.DataFrame):
        mkt = mkt.iloc[:, 0]
    mkt.index = pd.to_datetime(mkt.index).tz_localize(None)
    print(f"  0050 OK")

    # Step 4: 各期計算 PIT 特徵
    print("\n計算各期 Point-in-Time 特徵...")
    all_rows = []
    for period, holding_ids in sorted(period_holdings.items()):
        # as_of = 該期最後一個交易日
        period_end = pd.Timestamp(period + "-01") + relativedelta(months=1) - timedelta(days=1)
        as_of = pd.Timestamp(period_end)

        n1 = n0 = 0
        for tk, sym in valid_universe.items():
            if sym not in price_data:
                continue
            feat = calc_pit_features(price_data[sym], mkt, as_of)
            if feat is None:
                continue
            in_etf = int(tk in holding_ids)
            all_rows.append({
                "stock_id": tk,
                "period":   period,
                "in_etf":   in_etf,
                **feat,
            })
            if in_etf: n1 += 1
            else:      n0 += 1
        print(f"  [{period}] in_etf=1: {n1}, in_etf=0: {n0}")

    full = pd.DataFrame(all_rows)
    hist_path = DATA_DIR / "holdings_history.csv"
    full.to_csv(hist_path, index=False, encoding="utf-8-sig")
    print(f"\n完成! {hist_path} ({len(full)} 筆, {full['period'].nunique()} 期)")
    print(f"in_etf 分布: {full['in_etf'].value_counts().to_dict()}")
    print("接下來執行: python backtest/walk_forward.py")
