#!/usr/bin/env python3
"""
00981A 全期歷史持股下載 - Point-in-Time 版 v4

特徵清單：
  股價類: mom_1m/3m/6m, vol_20d/60d, rs_vs_market, above_ma60, price_range
  成交量類: vol_ratio_20d, log_turnover, price_52w_pct
  營收類: rev_mom_1m, rev_mom_3m, rev_yoy  (來源: MOPS 公開資訊觀測站)

執行： python data/fetch_history.py
"""
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import json
import time
import re
import warnings
from pathlib import Path
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from io import StringIO

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COOKIE_FILE = ROOT / ".cookie"
CACHE_FILE  = DATA_DIR / "period_holdings.json"
REV_CACHE   = DATA_DIR / "revenue_cache.json"

UNIVERSE_RAW = [
    "2330","2303","2379","2454","3711","2408","3037","2327","2344","2449",
    "2317","2308","2357","2382","2383","2368","2313","3017","3653","3665",
    "2059","2404","2439","3583","3533","3376","3189","3661","6515","6191",
    "6805","8046","8210","8996","3045","2345","2002",
    "2337","2338","2340","2342","2351","2352","2353","2354","2355","2356",
    "2359","2360","2362","2363","2364","2365","2367","2369",
    "2371","2373","2374","2375","2376","2377","2380",
    "2385","2387","2388","2390","2392","2393","2395","2397",
    "2399","2401","2402","2405","2406","2409",
    "5274","6223","8299","6274","3211","3217","3264","3324",
    "4979","5347","5536","6488","6510",
    "3014","3016","3018","3019","3021","3022","3023","3024","3026",
    "3027","3028","3029","3030","3032","3033","3034","3035","3036","3038",
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


def get_valid_symbol(stock_id):
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


def fetch_revenue_mops(stock_id: str, year_month: str) -> float | None:
    """
    從 MOPS 公開資訊觀測站下載月營收 (TWD 千元)
    使用 POST API: https://mops.twse.com.tw/mops/web/t21sc03
    """
    roc_year = int(year_month[:4]) - 1911
    month    = int(year_month[5:])

    # 先嘗試上市 (sii)，失敗再試上櫃 (otc)
    for market in ("sii", "otc"):
        try:
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
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": "https://mops.twse.com.tw/mops/web/t21sc03",
            }
            r = requests.post(url, data=data, headers=headers, timeout=20)
            r.encoding = "utf-8"
            tbls = pd.read_html(StringIO(r.text), thousands=",")
            for tbl in tbls:
                # 橏選包含營收數字的表（至少 3 欄）
                if tbl.shape[1] < 3 or tbl.shape[0] < 1:
                    continue
                # 尋找包含該股代號的行
                strid = str(stock_id)
                mask  = tbl.astype(str).apply(
                    lambda row: row.str.contains(strid, regex=False).any(), axis=1
                )
                if not mask.any():
                    continue
                row = tbl[mask].iloc[0]
                # 營收通常在最後幾欄，或標題含「營收」
                for col in tbl.columns:
                    if "營收" in str(col) and "當月" in str(col):
                        try:
                            val = float(str(row[col]).replace(",",""))
                            if val > 0:
                                return val
                        except:
                            pass
                # 如果標題沒有「營收」，嘗試第 3~5 欄
                for idx in range(2, min(6, tbl.shape[1])):
                    try:
                        val = float(str(row.iloc[idx]).replace(",",""))
                        if val > 0:
                            return val
                    except:
                        pass
        except:
            pass
    return None


def calc_rev_features(stock_id: str, as_of_period: str, rev_cache: dict) -> dict:
    as_of_dt    = pd.Timestamp(as_of_period + "-01")
    rev_month   = (as_of_dt - relativedelta(months=1)).strftime("%Y-%m")
    rev_month_2 = (as_of_dt - relativedelta(months=2)).strftime("%Y-%m")
    rev_month_3 = (as_of_dt - relativedelta(months=3)).strftime("%Y-%m")
    rev_yoy_m   = (as_of_dt - relativedelta(months=13)).strftime("%Y-%m")

    if stock_id not in rev_cache:
        rev_cache[stock_id] = {}
    cache = rev_cache[stock_id]

    def get_rev(ym):
        if ym not in cache:
            cache[ym] = fetch_revenue_mops(stock_id, ym)
            time.sleep(0.4)
        return cache[ym]

    r0, r1, r2, ry = get_rev(rev_month), get_rev(rev_month_2), \
                     get_rev(rev_month_3), get_rev(rev_yoy_m)

    return {
        "rev_mom_1m": float(r0/r1 - 1)      if r0 and r1 and r1 > 0 else np.nan,
        "rev_mom_3m": float((r0/r2)**(1/3)-1) if r0 and r2 and r2 > 0 else np.nan,
        "rev_yoy":    float(r0/ry - 1)       if r0 and ry and ry > 0  else np.nan,
    }


def calc_pit_features(closes, volumes, market, as_of):
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
    c52   = c.tail(252)
    p52h, p52l = float(c52.max()), float(c52.min())
    p52pct = float((p0-p52l)/(p52h-p52l)) if p52h > p52l else np.nan
    vr, lt = np.nan, np.nan
    if len(v) >= 60:
        a20 = float(v.tail(20).mean())
        a60 = float(v.tail(60).mean())
        vr  = a20/a60 if a60 > 0 else np.nan
        lt  = float(np.log1p(a20))
    return {
        "mom_1m": mom(21), "mom_3m": mom(63), "mom_6m": mom(126),
        "vol_20d": vol20, "vol_60d": vol60, "rs_vs_market": rs,
        "above_ma60": float(p0 > ma60) if not np.isnan(ma60) else np.nan,
        "price_range": pr, "price_52w_pct": p52pct,
        "vol_ratio_20d": vr, "log_turnover": lt,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not COOKIE_FILE.exists():
        print("[ERROR] 請先執行: python data/input_cookie.py")
        exit(1)
    cookie = COOKIE_FILE.read_text(encoding="utf-8").strip()

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

    print(f"\n宇宙: {len(UNIVERSE)} 支，確認後綴...")
    valid_universe = {}
    for tk in UNIVERSE:
        r = get_valid_symbol(tk)
        if r:
            valid_universe[tk] = r[0]
        time.sleep(0.12)
    print(f"有效宇宙: {len(valid_universe)} 支")

    print("\n下載股價資料...")
    price_data, volume_data = {}, {}
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

    rev_cache = json.loads(REV_CACHE.read_text(encoding="utf-8")) \
        if REV_CACHE.exists() else {}
    cached_cnt = sum(len(v) for v in rev_cache.values())
    if cached_cnt:
        print(f"\n營收快取: {cached_cnt} 筆")

    print("\n計算各期 PIT 特徵 (含 MOPS 營收)...")
    all_rows = []
    for period, holding_ids in sorted(period_holdings.items()):
        period_end = pd.Timestamp(period+"-01") + relativedelta(months=1) - timedelta(days=1)
        as_of = pd.Timestamp(period_end)
        n1 = n0 = rev_ok = rev_fail = 0
        for tk, sym in valid_universe.items():
            if sym not in price_data:
                continue
            feat = calc_pit_features(price_data[sym], volume_data.get(sym), mkt, as_of)
            if feat is None:
                continue
            rev_feat = calc_rev_features(tk, period, rev_cache)
            if not np.isnan(rev_feat.get("rev_mom_1m", np.nan)):
                rev_ok += 1
            else:
                rev_fail += 1
            in_etf = int(tk in holding_ids)
            all_rows.append({"stock_id": tk, "period": period, "in_etf": in_etf,
                             **feat, **rev_feat})
            if in_etf: n1 += 1
            else:      n0 += 1
        print(f"  [{period}] in=1:{n1} in=0:{n0} | 營收OK:{rev_ok} 失敗:{rev_fail}")
        REV_CACHE.write_text(
            json.dumps(rev_cache, ensure_ascii=False, default=str), encoding="utf-8")

    full = pd.DataFrame(all_rows)
    hist_path = DATA_DIR / "holdings_history.csv"
    full.to_csv(hist_path, index=False, encoding="utf-8-sig")
    print(f"\n完成! {hist_path} ({len(full)} 筆)")
    nan_rates = full[[c for c in full.columns if c not in ["stock_id","period","in_etf"]]].isna().mean()
    for col, r in nan_rates[nan_rates > 0.1].items():
        print(f"  {col} NaN: {r*100:.0f}%")
    print("接下來執行: python backtest/walk_forward.py")
