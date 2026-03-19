#!/usr/bin/env python3
"""
00981A 全期歷史持股下載 - Point-in-Time 版 v3

特徵清單：
  股價簻: mom_1m/3m/6m, vol_20d/60d, rs_vs_market, above_ma60, price_range
  成交量簻: vol_ratio_20d (近 20 日均量 vs 60 日均量), turnover_20d_avg, price_52w_pct
  營收簻: rev_mom_1m (月營收月増率), rev_mom_3m (營收 3M平均年化成長)

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
from io import StringIO

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COOKIE_FILE = ROOT / ".cookie"
CACHE_FILE  = DATA_DIR / "period_holdings.json"
REV_CACHE   = DATA_DIR / "revenue_cache.json"

UNIVERSE_RAW = [
    # 上市電子 (大市値)
    "2330","2303","2379","2454","3711","2408","3037","2327","2344","2449",
    "2317","2308","2357","2382","2383","2368","2313","3017","3653","3665",
    "2059","2404","2439","3583","3533","3376","3189","3661","6515","6191",
    "6805","8046","8210","8996","3045","2345","2002",
    # 上市電子 (中小市値)
    "2337","2338","2340","2342","2351","2352","2353","2354","2355","2356",
    "2359","2360","2362","2363","2364","2365","2367","2369",
    "2371","2373","2374","2375","2376","2377","2380",
    "2385","2387","2388","2390","2392","2393","2395","2397",
    "2399","2401","2402","2405","2406","2409",
    # 上櫃電子
    "5274","6223","8299","6274","3211","3217","3264","3324",
    "4979","5347","5536","6488","6510",
    "3014","3016","3018","3019","3021","3022","3023","3024","3026",
    "3027","3028","3029","3030","3032","3033","3034","3035","3036","3038",
    # 非電子
    "2412","2882","2886","2891","1301","1303",
    "2881","2884","2885","2912","1326",
    "2801","2880","2883","2887",
]
_seen = set()
UNIVERSE = [x for x in UNIVERSE_RAW if x not in _seen and not _seen.add(x)]

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


def fetch_holdings_by_date(query_date, cookie, retries=4, backoff=5.0):
    url = (
        "https://www.pocket.tw/api/cm/MobileService/ashx/GetDtnoData.ashx"
        "?action=getdtnodata&DtNo=59449513"
        f"&ParamStr=AssignID%3D00981A%3BQueryDate%3D{query_date}%3B&FilterNo=0"
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={**HEADERS_BASE, "Cookie": cookie}, timeout=25)
            r.raise_for_status()
            rows = json.loads(r.text).get("Data", [])
            return {str(row[1]).strip() for row in rows
                    if len(row) > 1 and re.match(r'^\d{4,6}$', str(row[1]).strip())}
        except Exception as e:
            wait = backoff * (attempt + 1)
            print(f"[重試 {attempt+1}/{retries}] {e} -> 等待 {wait:.0f}s")
            time.sleep(wait)
    return set()


# ── 月營收特徵 ──
def fetch_revenue_mops(stock_id: str, year_month: str) -> float | None:
    """
    從 MOPS 取得某股某月營收 (TWD 千元)
    year_month: 'YYYY-MM'
    回傳營收千元，失敗回 None
    """
    # 民國年
    y, m = int(year_month[:4]) - 1911, int(year_month[5:])
    url = "https://mops.twse.com.tw/nas/t21/sii/t21sc03_{y}_{m}_0.html".format(y=y, m=m)
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.encoding = "big5"
        tables = pd.read_html(StringIO(r.text))
        for tbl in tables:
            # 找包含公司代號的表
            if tbl.shape[1] < 4:
                continue
            tbl.columns = tbl.columns.droplevel(0) if isinstance(tbl.columns, pd.MultiIndex) else tbl.columns
            for col in tbl.columns:
                tbl[col] = tbl[col].astype(str)
            # 尋找該股記錄
            mask = tbl.apply(lambda row: any(stock_id in str(v) for v in row), axis=1)
            if mask.any():
                row = tbl[mask].iloc[0]
                # 營收在第 3 或第 4 欄（當月營收）
                for idx in [2, 3, 4]:
                    try:
                        val = str(row.iloc[idx]).replace(",", "").strip()
                        return float(val)
                    except:
                        pass
    except:
        pass
    return None


def get_revenue_series(stock_id: str, rev_cache: dict) -> dict:
    """
    回傳 {'YYYY-MM': revenue_float} 字典，利用快取
    """
    if stock_id not in rev_cache:
        rev_cache[stock_id] = {}
    return rev_cache[stock_id]


def calc_rev_features(stock_id: str, as_of_period: str, rev_cache: dict) -> dict:
    """
    計算 as_of_period 期末可知的營收特徵
    營收在每月 10 日公布，所以 as_of_period 2025-06 可知的最新營收是 2025-05
    """
    # 最新可知的營收月 = as_of_period 的上一個月
    as_of_dt = pd.Timestamp(as_of_period + "-01")
    rev_month   = (as_of_dt - relativedelta(months=1)).strftime("%Y-%m")
    rev_month_2 = (as_of_dt - relativedelta(months=2)).strftime("%Y-%m")
    rev_month_3 = (as_of_dt - relativedelta(months=3)).strftime("%Y-%m")
    rev_month_yoy = (as_of_dt - relativedelta(months=13)).strftime("%Y-%m")  # 去年同期

    cache = get_revenue_series(stock_id, rev_cache)

    def get_rev(ym):
        if ym not in cache:
            val = fetch_revenue_mops(stock_id, ym)
            cache[ym] = val
            time.sleep(0.3)
        return cache[ym]

    r0  = get_rev(rev_month)
    r1  = get_rev(rev_month_2)
    r2  = get_rev(rev_month_3)
    r_yoy = get_rev(rev_month_yoy)

    rev_mom_1m = float(r0 / r1 - 1) if r0 and r1 and r1 > 0 else np.nan
    rev_mom_3m = float((r0 / r2) ** (1/3) - 1) if r0 and r2 and r2 > 0 else np.nan
    rev_yoy    = float(r0 / r_yoy - 1) if r0 and r_yoy and r_yoy > 0 else np.nan

    return {
        "rev_mom_1m": rev_mom_1m,
        "rev_mom_3m": rev_mom_3m,
        "rev_yoy":    rev_yoy,
    }


def calc_pit_features(closes: pd.Series, volumes: pd.Series,
                      market: pd.Series, as_of: pd.Timestamp) -> dict | None:
    c = closes[closes.index <= as_of].dropna()
    v = volumes[volumes.index <= as_of].dropna() if volumes is not None else pd.Series(dtype=float)
    m = market[market.index <= as_of].dropna()
    if len(c) < 20:
        return None

    p0 = float(c.iloc[-1])
    def mom(n): return float(c.iloc[-1]/c.iloc[-n]-1) if len(c) >= n else np.nan

    vol20 = float(c.pct_change().tail(20).std()*np.sqrt(252)) if len(c) >= 20 else np.nan
    vol60 = float(c.pct_change().tail(60).std()*np.sqrt(252)) if len(c) >= 60 else np.nan
    ma60  = float(c.tail(60).mean()) if len(c) >= 60 else np.nan
    rs    = float(c.iloc[-1]/c.iloc[-20] - m.iloc[-1]/m.iloc[-20]) if len(m) >= 20 else np.nan
    pc    = c.tail(21)
    pr    = float((pc.max()-pc.min())/pc.min()) if len(pc) >= 5 else np.nan

    # 52 週高位百分比
    c52 = c.tail(252)
    p52h = float(c52.max())
    p52l = float(c52.min())
    price_52w_pct = float((p0 - p52l) / (p52h - p52l)) if p52h > p52l else np.nan

    # 成交量特徵
    vol_ratio_20d = np.nan
    turnover_20d  = np.nan
    if len(v) >= 60:
        avg20 = float(v.tail(20).mean())
        avg60 = float(v.tail(60).mean())
        vol_ratio_20d = avg20 / avg60 if avg60 > 0 else np.nan
        turnover_20d  = avg20  # 即均日成交量（對比不同股票使用 log）

    return {
        "mom_1m": mom(21), "mom_3m": mom(63), "mom_6m": mom(126),
        "vol_20d": vol20, "vol_60d": vol60,
        "rs_vs_market": rs,
        "above_ma60": float(p0 > ma60) if not np.isnan(ma60) else np.nan,
        "price_range": pr,
        "price_52w_pct": price_52w_pct,
        "vol_ratio_20d": vol_ratio_20d,
        "log_turnover": float(np.log1p(turnover_20d)) if not np.isnan(turnover_20d) else np.nan,
    }


if __name__ == "__main__":
    if not COOKIE_FILE.exists():
        print("[ERROR] 請先執行: python data/input_cookie.py")
        exit(1)
    cookie = COOKIE_FILE.read_text(encoding="utf-8").strip()

    # Step 1: 持股快取
    START = date(2025, 6, 1)
    END   = date.today().replace(day=1)
    period_holdings = {k: set(v) for k, v in json.loads(CACHE_FILE.read_text()).items()} \
        if CACHE_FILE.exists() else {}
    if period_holdings:
        print(f"載入快取: 已有 {len(period_holdings)} 期")

    cur = START
    while cur <= END:
        period   = cur.strftime("%Y-%m")
        date_str = cur.strftime("%Y%m%d")
        if period in period_holdings:
            print(f"[{period}] 已有快取，跳過")
            cur += relativedelta(months=1)
            continue
        print(f"[{period}] 查詢...", end=" ", flush=True)
        ids = fetch_holdings_by_date(date_str, cookie)
        if not ids:
            ids = fetch_holdings_by_date(cur.replace(day=15).strftime("%Y%m%d"), cookie)
        if ids:
            print(f"實際持股 {len(ids)} 支")
            period_holdings[period] = ids
            CACHE_FILE.write_text(
                json.dumps({k: list(v) for k, v in period_holdings.items()}, ensure_ascii=False),
                encoding="utf-8")
        else:
            print("無資料")
        cur += relativedelta(months=1)
        time.sleep(2.0)

    if not period_holdings:
        print("[ERROR] 沒有持股資料")
        exit(1)
    print(f"\n共 {len(period_holdings)} 期: {sorted(period_holdings.keys())}")

    # Step 2: 確認宇宙後綴
    print(f"\n宇宙: {len(UNIVERSE)} 支，確認後綴...")
    valid_universe = {}
    for tk in UNIVERSE:
        r = get_valid_symbol(tk)
        if r:
            valid_universe[tk] = r[0]
        time.sleep(0.12)
    print(f"有效宇宙: {len(valid_universe)} 支")

    # Step 3: 下載股價和成交量
    print("\n下載股價資料...")
    price_data  = {}
    volume_data = {}
    for sym in list(valid_universe.values()):
        try:
            h = yf.download(sym, start="2024-01-01", progress=False, auto_adjust=True)
            if not h.empty:
                close = h["Close"]
                vol   = h["Volume"]
                if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                if isinstance(vol,   pd.DataFrame): vol   = vol.iloc[:, 0]
                close.index = pd.to_datetime(close.index).tz_localize(None)
                vol.index   = pd.to_datetime(vol.index).tz_localize(None)
                price_data[sym]  = close
                volume_data[sym] = vol
                print(f"  {sym} OK")
        except Exception as e:
            print(f"  {sym} [ERR] {e}")
        time.sleep(0.08)

    mkt = yf.download("0050.TW", start="2024-01-01", progress=False, auto_adjust=True)["Close"]
    if isinstance(mkt, pd.DataFrame): mkt = mkt.iloc[:, 0]
    mkt.index = pd.to_datetime(mkt.index).tz_localize(None)
    print("  0050 OK")

    # Step 4: 活動營收快取
    rev_cache = {}
    if REV_CACHE.exists():
        rev_cache = json.loads(REV_CACHE.read_text(encoding="utf-8"))
        print(f"\n營收快取已載入: {sum(len(v) for v in rev_cache.values())} 筆")

    # Step 5: PIT 特徵
    print("\n計算各期 Point-in-Time 特徵...")
    all_rows = []
    periods_sorted = sorted(period_holdings.keys())

    for period, holding_ids in sorted(period_holdings.items()):
        period_end = pd.Timestamp(period+"-01") + relativedelta(months=1) - timedelta(days=1)
        as_of = pd.Timestamp(period_end)
        n1 = n0 = 0
        for tk, sym in valid_universe.items():
            if sym not in price_data:
                continue
            feat = calc_pit_features(
                price_data[sym],
                volume_data.get(sym),
                mkt, as_of
            )
            if feat is None:
                continue

            # 月營收特徵
            rev_feat = calc_rev_features(tk, period, rev_cache)

            in_etf = int(tk in holding_ids)
            all_rows.append({
                "stock_id": tk, "period": period, "in_etf": in_etf,
                **feat, **rev_feat
            })
            if in_etf: n1 += 1
            else:      n0 += 1
        print(f"  [{period}] in_etf=1: {n1}, in_etf=0: {n0}")

        # 儲存營收快取
        REV_CACHE.write_text(
            json.dumps(rev_cache, ensure_ascii=False, default=str),
            encoding="utf-8")

    full = pd.DataFrame(all_rows)
    hist_path = DATA_DIR / "holdings_history.csv"
    full.to_csv(hist_path, index=False, encoding="utf-8-sig")
    print(f"\n完成! {hist_path} ({len(full)} 筆, {full['period'].nunique()} 期)")
    print(f"in_etf 分布: {full['in_etf'].value_counts().to_dict()}")
    print(f"特徵欄: {[c for c in full.columns if c not in ['stock_id','period','in_etf']]}")
    print("接下來執行: python backtest/walk_forward.py")
